use crate::{RegressionBackend, RegressionContract, RegressionFrameInput};
use audio2x_core::{Audio2xError, BindingSchema, Dimension, ElementType, Result};
use audio2x_cuda::{CudaStream, GpuDevice};
use audio2x_inference::{BindingBuffer, DeviceBindings, TensorRtSession};
use std::path::Path;
use std::rc::Rc;

pub struct TensorRtRegressionBackend {
    contract: RegressionContract,
    device: Rc<GpuDevice>,
    session: TensorRtSession,
    stream: CudaStream,
}

impl TensorRtRegressionBackend {
    pub fn load(
        device: Rc<GpuDevice>,
        engine: &Path,
        contract: RegressionContract,
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

    fn run(&mut self, input: &RegressionFrameInput) -> Result<Vec<f32>> {
        self.run_batch(std::slice::from_ref(input))?
            .pop()
            .ok_or_else(|| Audio2xError::InvalidSchema("empty regression result".into()))
    }

    fn run_batch(&mut self, inputs: &[RegressionFrameInput]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Err(Audio2xError::InvalidSchema(
                "empty regression inference batch".into(),
            ));
        }
        let batch = inputs.len();
        let result_size = self.contract.result_layout.total()?;
        let mut emotion = self
            .device
            .allocate::<f32>(self.contract.emotion_size * batch)?;
        let mut audio = self
            .device
            .allocate::<f32>(self.contract.audio_size * batch)?;
        let result = self.device.allocate::<f32>(result_size * batch)?;
        let emotion_host: Vec<_> = inputs
            .iter()
            .flat_map(|input| input.emotion.iter().copied())
            .collect();
        let audio_host: Vec<_> = inputs
            .iter()
            .flat_map(|input| input.audio.iter().copied())
            .collect();
        emotion.copy_from(&emotion_host, &self.stream)?;
        audio.copy_from(&audio_host, &self.stream)?;
        let mut bindings = DeviceBindings::new();
        bindings
            .insert(
                "emotion",
                BindingBuffer::from_view(emotion.view(), ElementType::F32),
            )
            .map_err(inference_error)?;
        bindings
            .insert(
                "input",
                BindingBuffer::from_view(audio.view(), ElementType::F32),
            )
            .map_err(inference_error)?;
        bindings
            .insert(
                "result",
                BindingBuffer::from_view(result.view(), ElementType::F32),
            )
            .map_err(inference_error)?;
        bindings
            .set_input_shape(
                "emotion",
                vec![batch as i64, 1, self.contract.emotion_size as i64],
            )
            .map_err(inference_error)?;
        bindings
            .set_input_shape(
                "input",
                vec![batch as i64, 1, self.contract.audio_size as i64],
            )
            .map_err(inference_error)?;
        self.session
            .enqueue(0, &bindings, &self.stream)
            .map_err(inference_error)?
            .synchronize()
            .map_err(inference_error)?;
        let mut host = vec![0.0; result_size * batch];
        result.copy_to(&mut host, &self.stream)?;
        Ok(host
            .chunks_exact(result_size)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl RegressionBackend for TensorRtRegressionBackend {
    type Output = Vec<f32>;
    fn infer(&mut self, _track: usize, input: &RegressionFrameInput) -> Result<Self::Output> {
        self.run(input)
    }

    fn infer_batch(
        &mut self,
        inputs: &[(usize, RegressionFrameInput)],
    ) -> Result<Vec<Self::Output>> {
        let frames: Vec<_> = inputs.iter().map(|(_, input)| input.clone()).collect();
        self.run_batch(&frames)
    }
}

fn inference_error(error: impl std::fmt::Display) -> Audio2xError {
    Audio2xError::InvalidSchema(format!("TensorRT regression inference failed: {error}"))
}

fn validate_schema(schema: &BindingSchema, contract: &RegressionContract) -> Result<()> {
    let expected = [
        (
            "emotion",
            audio2x_core::IoMode::Input,
            contract.emotion_size,
        ),
        ("input", audio2x_core::IoMode::Input, contract.audio_size),
        (
            "result",
            audio2x_core::IoMode::Output,
            contract.result_layout.total()?,
        ),
    ];
    if schema.bindings().len() != expected.len() {
        return Err(Audio2xError::InvalidSchema(
            "regression engine must have three bindings".into(),
        ));
    }
    for (name, mode, width) in expected {
        let binding = schema.get(name).ok_or_else(|| {
            Audio2xError::InvalidSchema(format!("missing regression binding {name}"))
        })?;
        let dimensions = binding.shape.dimensions();
        let batch_accepts_one = match dimensions.first() {
            Some(Dimension::Batch) => true,
            Some(Dimension::Dynamic { min, max }) => *min <= 1 && *max >= 1,
            _ => false,
        };
        if binding.mode != mode
            || binding.element_type != ElementType::F32
            || dimensions.len() != 3
            || !batch_accepts_one
            || dimensions[1] != Dimension::Fixed(1)
            || dimensions[2] != Dimension::Fixed(width)
        {
            return Err(Audio2xError::InvalidSchema(format!(
                "invalid regression binding {name}: {binding:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PumpStatus, RegressionExecutor, RegressionTrack};
    use audio2x_core::{
        GeometryAudioParameters, GeometryParameters, NetworkDocument, load_network,
    };

    #[test]
    fn runs_installed_regression_model_when_configured() {
        let Some(root) = std::env::var_os("AUDIO2X_TEST_REGRESSION_MODEL_DIR") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        let NetworkDocument::Geometry(network) =
            load_network(root.join("network_info.json")).unwrap()
        else {
            panic!("expected geometry model")
        };
        let GeometryParameters::Regression(parameters) = network.params else {
            panic!("expected regression parameters")
        };
        let GeometryAudioParameters::Regression(audio) = network.audio_params else {
            panic!("expected regression audio parameters")
        };
        let contract = RegressionContract::new(&parameters, &audio, 30, 1).unwrap();
        let input = RegressionFrameInput {
            timestamp: 0,
            next_timestamp: 533,
            audio: vec![0.0; contract.audio_size],
            emotion: vec![0.0; contract.emotion_size],
        };
        let engine = std::env::var_os("AUDIO2X_TEST_REGRESSION_ENGINE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| root.join("network.trt"));
        let mut backend =
            TensorRtRegressionBackend::load(GpuDevice::new(0).unwrap(), &engine, contract.clone())
                .unwrap();
        let result = backend.run(&input).unwrap();
        assert_eq!(result.len(), contract.result_layout.total().unwrap());
        assert!(result.iter().all(|value| value.is_finite()));
        let batched = backend.run_batch(&[input.clone(), input.clone()]).unwrap();
        assert_eq!(batched.len(), 2);
        assert!(batched.iter().flatten().all(|value| value.is_finite()));

        let audio_accumulator = audio2x_core::AudioAccumulator::new(audio.buffer_len, 0).unwrap();
        audio_accumulator.accumulate(&input.audio).unwrap();
        audio_accumulator.close().unwrap();
        let explicit_size = parameters.explicit_emotions.len();
        let emotion_accumulator = audio2x_core::EmotionAccumulator::new(explicit_size, 2).unwrap();
        emotion_accumulator
            .accumulate(0, &vec![0.0; explicit_size])
            .unwrap();
        emotion_accumulator.close().unwrap();
        let executor = RegressionExecutor::new(contract, 1).unwrap();
        let implicit = vec![0.0; parameters.implicit_emotion_len];
        let track = RegressionTrack {
            audio: &audio_accumulator,
            emotions: &emotion_accumulator,
            implicit_emotion: &implicit,
            input_strength: 1.0,
        };
        let mut callbacks = 0;
        assert_eq!(
            executor
                .pump(&[track], &mut backend, |metadata, output| {
                    callbacks += 1;
                    assert_eq!(metadata.timestamp, 0);
                    assert!(output.iter().all(|value| value.is_finite()));
                    false
                })
                .unwrap(),
            PumpStatus::Interrupted
        );
        assert_eq!(callbacks, 1);
    }
}
