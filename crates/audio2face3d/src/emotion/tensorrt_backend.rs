use crate::common::{BindingSchema, Dimension, ElementType, Error, IoMode, Result};
use crate::cuda::{CudaStream, DeviceBuffer, GpuDevice};
use crate::emotion::{ClassifierBackend, ClassifierContract};
use crate::tensorrt::{BindingBuffer, DeviceBindings, TensorRtSession};
use std::path::Path;
use std::sync::Arc;

pub(crate) struct TensorRtClassifierBackend {
    contract: ClassifierContract,
    max_batch_size: usize,
    device: Arc<GpuDevice>,
    session: TensorRtSession,
    stream: CudaStream,
}

impl TensorRtClassifierBackend {
    pub(crate) fn load_interactive(
        device: Arc<GpuDevice>,
        engine: &Path,
        sample_rate: usize,
        emotion_length: usize,
        frame_rate_numerator: usize,
        frame_rate_denominator: usize,
        inferences_to_skip: usize,
    ) -> Result<(Self, ClassifierContract)> {
        let session =
            TensorRtSession::load(Arc::clone(&device), engine).map_err(inference_error)?;
        let input = session
            .metadata()
            .get("input_values")
            .ok_or_else(|| invalid("classifier input_values binding is missing"))?;
        let buffer_length = match input.shape.dimensions().get(1) {
            Some(Dimension::Fixed(value)) => *value,
            Some(Dimension::Dynamic { min, max }) if min == max => *min,
            Some(Dimension::Dynamic { max, .. }) => *max,
            _ => return Err(invalid("classifier audio buffer dimension is missing")),
        };
        let contract = ClassifierContract::new(
            buffer_length,
            sample_rate,
            emotion_length,
            frame_rate_numerator,
            frame_rate_denominator,
            inferences_to_skip,
        )?;
        drop(session);
        Ok((Self::load(device, engine, contract.clone())?, contract))
    }

    pub fn load(
        device: Arc<GpuDevice>,
        engine: &Path,
        contract: ClassifierContract,
    ) -> Result<Self> {
        let session =
            TensorRtSession::load(Arc::clone(&device), engine).map_err(inference_error)?;
        validate_schema(session.metadata(), &contract)?;
        let max_batch_size = classifier_max_batch_size(session.metadata())?;
        let stream = device.create_stream()?;
        Ok(Self {
            contract,
            max_batch_size,
            device,
            session,
            stream,
        })
    }

    pub const fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn validate_track_count(&self, track_count: usize) -> Result<()> {
        if track_count == 0 || track_count > self.max_batch_size {
            return Err(invalid(format!(
                "emotion track count {track_count} is outside classifier engine batch range 1..={}",
                self.max_batch_size
            )));
        }
        Ok(())
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Runs a packed classifier batch and retains its logits on the device.
    ///
    /// This is the primary path used by the owning facade. The internal
    /// scheduler copies this buffer to the host only for its host-output path.
    pub(crate) fn run_device_batch(
        &mut self,
        inputs: &[(usize, Vec<f32>)],
    ) -> Result<DeviceBuffer<f32>> {
        let inputs = inputs
            .iter()
            .map(|(_, audio)| audio.as_slice())
            .collect::<Vec<_>>();
        self.run_device_slices(&inputs)
    }

    fn run_batch(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let slices = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let output = self.run_device_slices(&slices)?;
        let output_elements = output.len();
        let mut host = vec![0.0; output_elements];
        output.copy_to(&mut host, &self.stream)?;
        Ok(host
            .chunks_exact(self.contract.emotion_length)
            .map(<[f32]>::to_vec)
            .collect())
    }

    fn run_device_slices(&mut self, inputs: &[&[f32]]) -> Result<DeviceBuffer<f32>> {
        if inputs.is_empty() || inputs.len() > self.max_batch_size {
            return Err(invalid(format!(
                "invalid classifier inference batch {}; engine accepts at most {}",
                inputs.len(),
                self.max_batch_size
            )));
        }
        if inputs
            .iter()
            .any(|input| input.len() != self.contract.buffer_length)
        {
            return Err(invalid("invalid classifier inference input dimensions"));
        }
        let batch = inputs.len();
        let audio_elements = batch
            .checked_mul(self.contract.buffer_length)
            .ok_or_else(|| invalid("classifier input allocation overflow"))?;
        let output_elements = batch
            .checked_mul(self.contract.emotion_length)
            .ok_or_else(|| invalid("classifier output allocation overflow"))?;
        let mut audio = self.device.allocate::<f32>(audio_elements)?;
        let output = self.device.allocate::<f32>(output_elements)?;
        let host = inputs
            .iter()
            .flat_map(|values| values.iter().copied())
            .collect::<Vec<_>>();
        audio.copy_from(&host, &self.stream)?;
        let mut bindings = DeviceBindings::new();
        bindings
            .insert(
                "input_values",
                BindingBuffer::from_view(audio.view(), ElementType::F32),
            )
            .map_err(inference_error)?;
        bindings
            .insert(
                "output",
                BindingBuffer::from_view(output.view(), ElementType::F32),
            )
            .map_err(inference_error)?;
        bindings
            .set_input_shape(
                "input_values",
                vec![
                    i64::try_from(batch).map_err(|_| invalid("classifier batch exceeds i64"))?,
                    i64::try_from(self.contract.buffer_length)
                        .map_err(|_| invalid("classifier buffer length exceeds i64"))?,
                ],
            )
            .map_err(inference_error)?;
        self.session
            .enqueue(0, &bindings, &self.stream)
            .map_err(inference_error)?
            .synchronize()
            .map_err(inference_error)?;
        Ok(output)
    }
}

impl ClassifierBackend for TensorRtClassifierBackend {
    fn max_batch_size(&self) -> Option<usize> {
        Some(self.max_batch_size)
    }

    fn infer(&mut self, _track: usize, audio: &[f32]) -> Result<Vec<f32>> {
        self.run_batch(&[audio.to_vec()])?
            .pop()
            .ok_or_else(|| invalid("empty classifier result"))
    }

    fn infer_batch(&mut self, inputs: &[(usize, Vec<f32>)]) -> Result<Vec<Vec<f32>>> {
        self.run_batch(
            &inputs
                .iter()
                .map(|(_, audio)| audio.clone())
                .collect::<Vec<_>>(),
        )
    }
}

fn classifier_max_batch_size(schema: &BindingSchema) -> Result<usize> {
    let input = schema
        .get("input_values")
        .ok_or_else(|| invalid("classifier input_values binding is missing"))?;
    match input.shape.dimensions().first() {
        Some(Dimension::Fixed(value)) => Ok(*value),
        Some(Dimension::Dynamic { max, .. }) => Ok(*max),
        Some(Dimension::Batch) => Err(invalid(
            "classifier engine batch dimension has no finite profile maximum",
        )),
        None => Err(invalid(
            "classifier input_values batch dimension is missing",
        )),
    }
}

fn validate_schema(schema: &BindingSchema, contract: &ClassifierContract) -> Result<()> {
    if schema.bindings().len() != 2 {
        return Err(invalid("classifier engine must have two bindings"));
    }
    let input = schema
        .get("input_values")
        .ok_or_else(|| invalid("classifier input_values binding is missing"))?;
    let output = schema
        .get("output")
        .ok_or_else(|| invalid("classifier output binding is missing"))?;
    if input.mode != IoMode::Input
        || input.element_type != ElementType::F32
        || input.shape.dimensions().len() != 2
        || !accepts_one(&input.shape.dimensions()[0])
        || !accepts_size(&input.shape.dimensions()[1], contract.buffer_length)
        || output.mode != IoMode::Output
        || output.element_type != ElementType::F32
        || output.shape.dimensions().len() != 2
        || !accepts_one(&output.shape.dimensions()[0])
        || output.shape.dimensions()[1] != Dimension::Fixed(contract.emotion_length)
    {
        return Err(invalid("classifier engine binding schema mismatch"));
    }
    Ok(())
}

fn accepts_one(dimension: &Dimension) -> bool {
    matches!(dimension, Dimension::Batch | Dimension::Fixed(1))
        || matches!(dimension, Dimension::Dynamic { min, max } if *min <= 1 && *max >= 1)
}

fn accepts_size(dimension: &Dimension, value: usize) -> bool {
    matches!(dimension, Dimension::Fixed(fixed) if *fixed == value)
        || matches!(dimension, Dimension::Dynamic { min, max } if *min <= value && *max >= value)
}

fn inference_error(error: impl std::fmt::Display) -> Error {
    invalid(format!("TensorRT classifier inference failed: {error}"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Binding, Shape};

    #[test]
    fn reads_maximum_batch_from_engine_profile_schema() {
        let schema = BindingSchema::new(vec![Binding {
            name: "input_values".into(),
            mode: IoMode::Input,
            element_type: ElementType::F32,
            shape: Shape::new(vec![
                Dimension::Dynamic { min: 1, max: 128 },
                Dimension::Dynamic {
                    min: 16000,
                    max: 60000,
                },
            ])
            .unwrap(),
        }])
        .unwrap();
        assert_eq!(classifier_max_batch_size(&schema).unwrap(), 128);
    }

    #[test]
    fn runs_installed_classifier_when_configured() {
        let Some(engine) = std::env::var_os("AUDIO2FACE3D_TEST_EMOTION_ENGINE") else {
            return;
        };
        let buffer_length = std::env::var("AUDIO2FACE3D_TEST_EMOTION_BUFFER_LENGTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(16000);
        let contract = ClassifierContract::new(buffer_length, 16000, 6, 30, 1, 0).unwrap();
        let mut backend = TensorRtClassifierBackend::load(
            GpuDevice::new(0).unwrap(),
            Path::new(&engine),
            contract,
        )
        .unwrap();
        if let Ok(expected) = std::env::var("AUDIO2FACE3D_TEST_EMOTION_MAX_BATCH") {
            assert_eq!(backend.max_batch_size(), expected.parse::<usize>().unwrap());
        }
        let invalid_batch = vec![Vec::new(); backend.max_batch_size() + 1];
        assert!(
            backend
                .run_batch(&invalid_batch)
                .unwrap_err()
                .to_string()
                .contains("accepts at most")
        );
        let output = backend.infer(0, &vec![0.0; buffer_length]).unwrap();
        assert_eq!(output.len(), 6);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
