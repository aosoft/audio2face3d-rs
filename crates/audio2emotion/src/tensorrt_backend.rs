use crate::{ClassifierBackend, ClassifierContract};
use audio2x_core::{Audio2xError, BindingSchema, Dimension, ElementType, IoMode, Result};
use audio2x_cuda::{CudaStream, GpuDevice};
use audio2x_inference::{BindingBuffer, DeviceBindings, TensorRtSession};
use std::path::Path;
use std::rc::Rc;

pub struct TensorRtClassifierBackend {
    contract: ClassifierContract,
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
        let stream = device.create_stream()?;
        Ok(Self {
            contract,
            device,
            session,
            stream,
        })
    }

    fn run_batch(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty()
            || inputs
                .iter()
                .any(|input| input.len() != self.contract.buffer_length)
        {
            return Err(invalid("invalid classifier inference batch"));
        }
        let batch = inputs.len();
        let mut audio = self
            .device
            .allocate::<f32>(batch * self.contract.buffer_length)?;
        let output = self
            .device
            .allocate::<f32>(batch * self.contract.emotion_length)?;
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
                vec![batch as i64, self.contract.buffer_length as i64],
            )
            .map_err(inference_error)?;
        self.session
            .enqueue(0, &bindings, &self.stream)
            .map_err(inference_error)?
            .synchronize()
            .map_err(inference_error)?;
        let mut host = vec![0.0; batch * self.contract.emotion_length];
        output.copy_to(&mut host, &self.stream)?;
        Ok(host
            .chunks_exact(self.contract.emotion_length)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl ClassifierBackend for TensorRtClassifierBackend {
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
        let output = backend.infer(0, &vec![0.0; buffer_length]).unwrap();
        assert_eq!(output.len(), 6);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
