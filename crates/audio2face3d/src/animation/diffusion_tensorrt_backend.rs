use crate::animation::{
    DiffusionBackend, DiffusionContract, DiffusionFrameInput, DiffusionInferenceOutput,
};
use crate::common::{BindingSchema, Dimension, ElementType, Error, Result};
use crate::cuda::{CudaStream, GpuDevice};
use crate::tensorrt::{BindingBuffer, DeviceBindings, TensorRtSession};
use std::path::Path;
use std::rc::Rc;

pub struct TensorRtDiffusionBackend {
    contract: DiffusionContract,
    device: Rc<GpuDevice>,
    session: TensorRtSession,
    stream: CudaStream,
}

impl TensorRtDiffusionBackend {
    pub fn load(device: Rc<GpuDevice>, engine: &Path, contract: DiffusionContract) -> Result<Self> {
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

    fn run_batch(
        &mut self,
        inputs: &[(usize, DiffusionFrameInput)],
    ) -> Result<Vec<DiffusionInferenceOutput>> {
        if inputs.is_empty() {
            return Err(invalid("empty diffusion inference batch"));
        }
        let batch = inputs.len();
        let state_size = self.contract.state_size()?;
        let result_size = self.contract.result_layout.total()?;
        let prediction_size = self
            .contract
            .total_frames()
            .checked_mul(result_size)
            .ok_or_else(|| invalid("diffusion prediction size overflow"))?;
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
        let pack = |select: fn(&DiffusionFrameInput) -> &[f32]| -> Vec<f32> {
            inputs
                .iter()
                .flat_map(|(_, input)| select(input).iter().copied())
                .collect()
        };
        let audio_host = pack(|input| &input.audio);
        let emotion_host = pack(|input| &input.emotions);
        let identity_host = pack(|input| &input.identity);
        let noise_host = pack(|input| &input.noise);
        // GRU tensors put batch between slice and latent dimensions.
        let mut state_host = Vec::with_capacity(state_size * batch);
        let slice_size = self.contract.gru_latent_size;
        let slices = self.contract.diffusion_steps * self.contract.gru_layers;
        for slice in 0..slices {
            for (_, input) in inputs {
                let offset = slice * slice_size;
                state_host.extend_from_slice(&input.input_latents[offset..offset + slice_size]);
            }
        }

        let mut audio = self.device.allocate::<f32>(audio_host.len())?;
        let mut emotion = self.device.allocate::<f32>(emotion_host.len())?;
        let mut identity = self.device.allocate::<f32>(identity_host.len())?;
        let mut noise = self.device.allocate::<f32>(noise_host.len())?;
        let mut input_latents = self.device.allocate::<f32>(state_host.len())?;
        let output_latents = self.device.allocate::<f32>(state_host.len())?;
        let prediction = self.device.allocate::<f32>(
            prediction_size
                .checked_mul(batch)
                .ok_or_else(|| invalid("batched diffusion prediction size overflow"))?,
        )?;
        audio.copy_from(&audio_host, &self.stream)?;
        emotion.copy_from(&emotion_host, &self.stream)?;
        identity.copy_from(&identity_host, &self.stream)?;
        noise.copy_from(&noise_host, &self.stream)?;
        input_latents.copy_from(&state_host, &self.stream)?;

        let mut bindings = DeviceBindings::new();
        for (name, buffer) in [
            ("window", audio.view()),
            ("emotion", emotion.view()),
            ("identity", identity.view()),
            ("noise", noise.view()),
            ("input_latents", input_latents.view()),
            ("output_latents", output_latents.view()),
            ("prediction", prediction.view()),
        ] {
            bindings
                .insert(name, BindingBuffer::from_view(buffer, ElementType::F32))
                .map_err(inference_error)?;
        }
        let batch = i64::try_from(batch).map_err(|_| invalid("diffusion batch exceeds i64"))?;
        for (name, shape) in [
            ("window", vec![batch, self.contract.audio_size as i64]),
            (
                "emotion",
                vec![
                    batch,
                    self.contract.center_frames as i64,
                    self.contract.emotion_size as i64,
                ],
            ),
            ("identity", vec![batch, self.contract.identity_size as i64]),
            (
                "noise",
                vec![
                    batch,
                    (self.contract.diffusion_steps + 1) as i64,
                    self.contract.total_frames() as i64,
                    result_size as i64,
                ],
            ),
            (
                "input_latents",
                vec![
                    self.contract.diffusion_steps as i64,
                    self.contract.gru_layers as i64,
                    batch,
                    self.contract.gru_latent_size as i64,
                ],
            ),
        ] {
            bindings
                .set_input_shape(name, shape)
                .map_err(inference_error)?;
        }
        self.session
            .enqueue(0, &bindings, &self.stream)
            .map_err(inference_error)?
            .synchronize()
            .map_err(inference_error)?;

        let mut state_output = vec![0.0; state_host.len()];
        let mut prediction_output = vec![0.0; prediction_size * batch as usize];
        output_latents.copy_to(&mut state_output, &self.stream)?;
        prediction.copy_to(&mut prediction_output, &self.stream)?;
        let mut outputs = Vec::with_capacity(batch as usize);
        for track in 0..batch as usize {
            let mut state = Vec::with_capacity(state_size);
            for slice in 0..slices {
                let offset = (slice * batch as usize + track) * slice_size;
                state.extend_from_slice(&state_output[offset..offset + slice_size]);
            }
            let offset = track * prediction_size;
            outputs.push(DiffusionInferenceOutput {
                output_latents: state,
                prediction: prediction_output[offset..offset + prediction_size].to_vec(),
            });
        }
        Ok(outputs)
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

fn validate_schema(schema: &BindingSchema, contract: &DiffusionContract) -> Result<()> {
    let expected = contract.schema()?;
    if schema.bindings().len() != expected.bindings().len() {
        return Err(invalid("diffusion engine must have seven bindings"));
    }
    for expected in expected.bindings() {
        let actual = schema
            .get(&expected.name)
            .ok_or_else(|| invalid(format!("missing diffusion binding {}", expected.name)))?;
        if actual.mode != expected.mode || actual.element_type != ElementType::F32 {
            return Err(invalid(format!(
                "invalid diffusion binding {}: {actual:?}",
                expected.name
            )));
        }
        let expected_dimensions = expected.shape.dimensions();
        let actual_dimensions = actual.shape.dimensions();
        if expected_dimensions.len() != actual_dimensions.len()
            || !expected_dimensions
                .iter()
                .zip(actual_dimensions)
                .all(|(expected, actual)| match (expected, actual) {
                    (Dimension::Batch, Dimension::Batch) => true,
                    (Dimension::Batch, Dimension::Dynamic { min, max }) => *min <= 1 && *max >= 1,
                    (Dimension::Fixed(expected), Dimension::Fixed(actual)) => expected == actual,
                    _ => false,
                })
        {
            return Err(invalid(format!(
                "invalid diffusion shape for {}: {actual:?}",
                expected.name
            )));
        }
    }
    Ok(())
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
        let outputs = backend.run_batch(&[(0, input)]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].prediction.iter().all(|value| value.is_finite()));
        assert!(
            outputs[0]
                .output_latents
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
