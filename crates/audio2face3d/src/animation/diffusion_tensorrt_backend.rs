use crate::animation::{
    DiffusionBackend, DiffusionBufferContract, DiffusionContract, DiffusionFrameInput,
    DiffusionInferenceInputBuffers, DiffusionInferenceOutput, DiffusionInferenceOutputBuffers,
    DiffusionInferenceStateBuffers,
};
use crate::common::{Error, Result};
use crate::cuda::{CudaStream, GpuDevice};
use crate::tensorrt::TensorRtSession;
use std::path::Path;
use std::sync::Arc;

pub(crate) struct TensorRtDiffusionBackend {
    contract: DiffusionContract,
    device: Arc<GpuDevice>,
    session: TensorRtSession,
    stream: CudaStream,
    constant_noise: Option<Vec<f32>>,
}

pub(crate) struct DiffusionDeviceBatch {
    pub state_output: Vec<Vec<f32>>,
    pub prediction: DiffusionInferenceOutputBuffers,
}

impl TensorRtDiffusionBackend {
    pub fn load(
        device: Arc<GpuDevice>,
        engine: &Path,
        contract: DiffusionContract,
    ) -> Result<Self> {
        let session =
            TensorRtSession::load(Arc::clone(&device), engine).map_err(inference_error)?;
        DiffusionBufferContract::new(&contract, 1)?
            .bindings()
            .validate_engine_schema(session.metadata())?;
        let stream = device.create_stream()?;
        Ok(Self {
            contract,
            device,
            session,
            stream,
            constant_noise: None,
        })
    }

    pub(crate) fn configure_constant_noise(&mut self, enabled: bool, seed: u64) -> Result<()> {
        self.constant_noise = if enabled {
            let size = self.contract.noise_size()?;
            let mut generator = crate::animation::GpuPhiloxNoise::new(&self.stream, 1, size, seed)?;
            let mut device_noise = self.device.allocate(size)?;
            generator
                .generate(0, &mut device_noise, &self.stream)?
                .synchronize()?;
            let mut noise = vec![0.0; size];
            device_noise.copy_to(&mut noise, &self.stream)?;
            Some(noise)
        } else {
            None
        };
        Ok(())
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    fn run_batch(
        &mut self,
        inputs: &[(usize, DiffusionFrameInput)],
    ) -> Result<Vec<DiffusionInferenceOutput>> {
        let device_batch = self.run_device_batch(inputs)?;
        let prediction_output = device_batch.prediction.copy_to_host(&self.stream)?;
        let result_size = self.contract.result_layout.total()?;
        let prediction_size = self
            .contract
            .total_frames()
            .checked_mul(result_size)
            .ok_or_else(|| invalid("diffusion prediction size overflow"))?;
        let mut outputs = Vec::with_capacity(inputs.len());
        for (track, state) in device_batch.state_output.into_iter().enumerate() {
            let offset = track * prediction_size;
            outputs.push(DiffusionInferenceOutput {
                output_latents: state,
                prediction: prediction_output[offset..offset + prediction_size].to_vec(),
            });
        }
        Ok(outputs)
    }

    pub(crate) fn run_device_batch(
        &mut self,
        inputs: &[(usize, DiffusionFrameInput)],
    ) -> Result<DiffusionDeviceBatch> {
        if inputs.is_empty() {
            return Err(invalid("empty diffusion inference batch"));
        }
        let batch = inputs.len();
        let state_size = self.contract.state_size()?;
        for (_, input) in inputs {
            if input.audio.len() != self.contract.audio_size
                || input.emotions.len() != self.contract.center_frames * self.contract.emotion_size
                || input.identity.len() != self.contract.identity_size
                || input.noise.len() != self.contract.noise_size()?
                || input.input_latents.len() != state_size
            {
                return Err(invalid("diffusion input dimensions mismatch"));
            }
        }
        let frames = inputs
            .iter()
            .map(|(_, input)| {
                let mut frame = input.clone();
                // The SDK generates one cuRAND tensor and shares it across
                // all tracks and inferences in constant-noise mode.
                if let Some(noise) = &self.constant_noise {
                    frame.noise.clone_from(noise);
                }
                frame
            })
            .collect::<Vec<_>>();
        let contract = DiffusionBufferContract::new(&self.contract, batch)?;
        let mut input = DiffusionInferenceInputBuffers::allocate(&self.device, &contract)?;
        let mut state = DiffusionInferenceStateBuffers::allocate(&self.device, &contract)?;
        let output = DiffusionInferenceOutputBuffers::allocate(&self.device, &contract)?;
        input.copy_frames(&frames, &self.stream)?;
        state.copy_track_inputs(&frames, &self.stream)?;
        let bindings = contract.device_bindings(&input, &state, &output)?;
        self.session
            .enqueue(0, &bindings, &self.stream)
            .map_err(inference_error)?
            .synchronize()
            .map_err(inference_error)?;

        let state_output = state.copy_output_tracks_to_host(&self.stream)?;
        Ok(DiffusionDeviceBatch {
            state_output,
            prediction: output,
        })
    }
}

impl DiffusionBackend for TensorRtDiffusionBackend {
    fn infer_batch(
        &mut self,
        inputs: &[(usize, DiffusionFrameInput)],
    ) -> Result<Vec<DiffusionInferenceOutput>> {
        self.run_batch(inputs)
    }
}

fn inference_error(error: impl std::fmt::Display) -> Error {
    invalid(format!("TensorRT diffusion inference failed: {error}"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        GeometryAudioParameters, GeometryParameters, NetworkDocument, load_network,
    };

    #[test]
    fn runs_installed_diffusion_model_when_configured() {
        let Some(root) = std::env::var_os("AUDIO2FACE3D_TEST_DIFFUSION_MODEL_DIR") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        let NetworkDocument::Geometry(network) =
            load_network(root.join("network_info.json")).unwrap()
        else {
            panic!("expected geometry model")
        };
        let GeometryParameters::Diffusion(parameters) = network.params else {
            panic!("expected diffusion parameters")
        };
        let GeometryAudioParameters::Diffusion(audio) = network.audio_params else {
            panic!("expected diffusion audio parameters")
        };
        let contract = DiffusionContract::new(&parameters, &audio).unwrap();
        let input = DiffusionFrameInput {
            audio: vec![0.0; contract.audio_size],
            emotions: vec![0.0; contract.center_frames * contract.emotion_size],
            identity: {
                let mut value = vec![0.0; contract.identity_size];
                value[0] = 1.0;
                value
            },
            noise: vec![0.0; contract.noise_size().unwrap()],
            input_latents: vec![0.0; contract.state_size().unwrap()],
        };
        let mut backend = TensorRtDiffusionBackend::load(
            GpuDevice::new(0).unwrap(),
            &root.join("network.trt"),
            contract.clone(),
        )
        .unwrap();
        let outputs = backend.run_batch(&[(0, input.clone())]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].prediction.iter().all(|value| value.is_finite()));
        assert!(
            outputs[0]
                .output_latents
                .iter()
                .all(|value| value.is_finite())
        );

        backend.configure_constant_noise(true, 0).unwrap();
        let fixed = backend.run_batch(&[(0, input.clone())]).unwrap();
        let mut changed_noise = input.clone();
        changed_noise.noise.fill(1.0);
        let replay = backend.run_batch(&[(0, changed_noise.clone())]).unwrap();
        assert_eq!(
            fixed, replay,
            "constant noise must ignore per-inference noise"
        );
        let tracks = backend
            .run_batch(&[(0, input.clone()), (1, changed_noise)])
            .unwrap();
        assert_eq!(
            tracks[0], tracks[1],
            "constant noise must be shared by tracks"
        );
        backend.configure_constant_noise(true, 1).unwrap();
        let other_seed = backend.run_batch(&[(0, input.clone())]).unwrap();
        assert_ne!(fixed[0].prediction, other_seed[0].prediction);
        backend.configure_constant_noise(false, 0).unwrap();
        let restored = backend.run_batch(&[(0, input)]).unwrap();
        assert_eq!(
            outputs, restored,
            "disabling constant noise must restore input noise"
        );
    }
}
