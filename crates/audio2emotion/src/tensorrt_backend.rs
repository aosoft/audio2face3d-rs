use crate::{ClassifierBackend, ClassifierContract};
use audio2x_core::{Audio2xError, BindingSchema, Dimension, ElementType, IoMode, Result};
use audio2x_cuda::{CudaStream, GpuDevice};
use audio2x_inference::{BindingBuffer, DeviceBindings, TensorRtSession};
use std::path::Path;
use std::rc::Rc;

pub struct TensorRtClassifierBackend {
    contract: ClassifierContract,
    max_batch_size: usize,
    device: Rc<GpuDevice>,
    session: TensorRtSession,
    stream: CudaStream,
}

impl TensorRtClassifierBackend {
    pub fn load(
        device: Rc<GpuDevice>,
        engine: &Path,
        contract: ClassifierContract,
    ) -> Result<Self> {
        let session = TensorRtSession::load(Rc::clone(&device), engine).map_err(inference_error)?;
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

    fn run_batch(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
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
        let host = inputs.iter().flatten().copied().collect::<Vec<_>>();
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
        let mut host = vec![0.0; output_elements];
        output.copy_to(&mut host, &self.stream)?;
        Ok(host
            .chunks_exact(self.contract.emotion_length)
            .map(<[f32]>::to_vec)
            .collect())
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

fn inference_error(error: impl std::fmt::Display) -> Audio2xError {
    invalid(format!("TensorRT classifier inference failed: {error}"))
}

fn invalid(message: impl Into<String>) -> Audio2xError {
    Audio2xError::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio2x_core::{Binding, Shape};

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
        let Some(engine) = std::env::var_os("AUDIO2X_TEST_EMOTION_ENGINE") else {
            return;
        };
        let buffer_length = std::env::var("AUDIO2X_TEST_EMOTION_BUFFER_LENGTH")
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
        if let Ok(expected) = std::env::var("AUDIO2X_TEST_EMOTION_MAX_BATCH") {
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
