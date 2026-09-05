use crate::animation::{
    RegressionBackend, RegressionBufferContract, RegressionContract, RegressionFrameInput,
    RegressionInferenceInputBuffers, RegressionInferenceOutputBuffers,
};
use crate::common::{Error, Result};
use crate::cuda::{CudaStream, GpuDevice};
use crate::tensorrt::TensorRtSession;
use std::path::Path;
use std::sync::Arc;

pub struct TensorRtRegressionBackend {
    contract: RegressionContract,
    device: Arc<GpuDevice>,
    session: TensorRtSession,
    stream: CudaStream,
}

impl TensorRtRegressionBackend {
    pub fn load(
        device: Arc<GpuDevice>,
        engine: &Path,
        contract: RegressionContract,
    ) -> Result<Self> {
        let session =
            TensorRtSession::load(Arc::clone(&device), engine).map_err(inference_error)?;
        RegressionBufferContract::new(&contract, 1)?
            .bindings()
            .validate_engine_schema(session.metadata())?;
        let stream = device.create_stream()?;
        Ok(Self {
            contract,
            device,
            session,
            stream,
        })
    }

    pub fn contract(&self) -> &RegressionContract {
        &self.contract
    }

    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    pub fn session(&self) -> &TensorRtSession {
        &self.session
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    fn run(&mut self, input: &RegressionFrameInput) -> Result<Vec<f32>> {
        self.run_batch(std::slice::from_ref(input))?
            .pop()
            .ok_or_else(|| Error::InvalidSchema("empty regression result".into()))
    }

    fn run_batch(&mut self, inputs: &[RegressionFrameInput]) -> Result<Vec<Vec<f32>>> {
        let output = self.run_device_batch(inputs)?;
        let result_size = self.contract.result_layout.total()?;
        let host = output.copy_to_host(&self.stream)?;
        Ok(host
            .chunks_exact(result_size)
            .map(<[f32]>::to_vec)
            .collect())
    }

    pub(crate) fn run_device_batch(
        &mut self,
        inputs: &[RegressionFrameInput],
    ) -> Result<RegressionInferenceOutputBuffers> {
        if inputs.is_empty() {
            return Err(Error::InvalidSchema(
                "empty regression inference batch".into(),
            ));
        }
        let batch = inputs.len();
        let contract = RegressionBufferContract::new(&self.contract, batch)?;
        let mut input = RegressionInferenceInputBuffers::allocate(&self.device, &contract)?;
        let output = RegressionInferenceOutputBuffers::allocate(&self.device, &contract)?;
        input.copy_frames(inputs, &self.stream)?;
        let bindings = contract.device_bindings(&input, &output)?;
        self.session
            .enqueue(0, &bindings, &self.stream)
            .map_err(inference_error)?
            .synchronize()
            .map_err(inference_error)?;
        Ok(output)
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

fn inference_error(error: impl std::fmt::Display) -> Error {
    Error::InvalidSchema(format!("TensorRT regression inference failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{PumpStatus, RegressionExecutor, RegressionTrack};
    use crate::common::{
        GeometryAudioParameters, GeometryParameters, NetworkDocument, load_network,
    };

    #[test]
    fn runs_installed_regression_model_when_configured() {
        let Some(root) = std::env::var_os("AUDIO2FACE3D_TEST_REGRESSION_MODEL_DIR") else {
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
        let engine = std::env::var_os("AUDIO2FACE3D_TEST_REGRESSION_ENGINE")
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

        let audio_accumulator = crate::common::AudioAccumulator::new(audio.buffer_len, 0).unwrap();
        audio_accumulator.accumulate(&input.audio).unwrap();
        audio_accumulator.close().unwrap();
        let explicit_size = parameters.explicit_emotions.len();
        let emotion_accumulator = crate::common::EmotionAccumulator::new(explicit_size, 2).unwrap();
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
