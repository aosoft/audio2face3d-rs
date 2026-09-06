#![cfg_attr(
    any(not(feature = "cuda"), not(feature = "tensorrt")),
    allow(dead_code)
)]

use crate::common::{EmotionNetwork, EmotionPostProcessingConfig};
use crate::common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmotionPostProcessData {
    pub inference_emotion_length: usize,
    pub output_emotion_length: usize,
    /// One output index per inference emotion; `-1` means unmapped.
    pub emotion_correspondence: Vec<i64>,
}

impl EmotionPostProcessData {
    pub fn validate(&self) -> Result<()> {
        if self.inference_emotion_length == 0
            || self.output_emotion_length == 0
            || self.emotion_correspondence.len() != self.inference_emotion_length
            || self.emotion_correspondence.iter().any(|index| {
                *index < -1
                    || usize::try_from(*index)
                        .is_ok_and(|index| index >= self.output_emotion_length)
            })
        {
            return Err(invalid("invalid emotion post-process correspondence"));
        }
        Ok(())
    }

    pub fn from_model(
        network: &EmotionNetwork,
        config: &EmotionPostProcessingConfig,
    ) -> Result<(Self, EmotionPostProcessParameters)> {
        let correspondence = network
            .emotions
            .iter()
            .map(|name| -> Result<i64> {
                config
                    .emotion_correspondence
                    .get(name)
                    .copied()
                    .ok_or_else(|| invalid(format!("missing correspondence for emotion {name}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let data = Self {
            inference_emotion_length: network.emotions.len(),
            output_emotion_length: config.output_emotion_length,
            emotion_correspondence: correspondence,
        };
        let parameters = EmotionPostProcessParameters {
            emotion_contrast: config.emotion_contrast,
            max_emotions: config.max_emotions,
            beginning_emotion: vec![0.0; config.output_emotion_length],
            preferred_emotion: config.preferred_emotion.clone(),
            live_blend_coefficient: config.live_blend_coef,
            enable_preferred_emotion: config.enable_preferred_emotion,
            preferred_emotion_strength: config.preferred_emotion_strength,
            live_transition_time: config.transition_smoothing,
            fixed_dt: config.fixed_dt,
            emotion_strength: config.emotion_strength,
        };
        data.validate()?;
        parameters.validate(&data)?;
        Ok((data, parameters))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmotionPostProcessParameters {
    pub emotion_contrast: f32,
    pub max_emotions: usize,
    pub beginning_emotion: Vec<f32>,
    pub preferred_emotion: Vec<f32>,
    pub live_blend_coefficient: f32,
    pub enable_preferred_emotion: bool,
    pub preferred_emotion_strength: f32,
    pub live_transition_time: f32,
    pub fixed_dt: f32,
    pub emotion_strength: f32,
}

impl Default for EmotionPostProcessParameters {
    fn default() -> Self {
        Self {
            emotion_contrast: 1.0,
            max_emotions: 0,
            beginning_emotion: Vec::new(),
            preferred_emotion: Vec::new(),
            live_blend_coefficient: 0.7,
            enable_preferred_emotion: false,
            preferred_emotion_strength: 0.5,
            live_transition_time: 0.5,
            fixed_dt: 0.033,
            emotion_strength: 0.6,
        }
    }
}

impl EmotionPostProcessParameters {
    pub(crate) fn validate(&self, data: &EmotionPostProcessData) -> Result<()> {
        if self.beginning_emotion.len() != data.output_emotion_length
            || self.preferred_emotion.len() != data.output_emotion_length
            || self.max_emotions > data.inference_emotion_length
        {
            return Err(invalid("invalid emotion post-process parameter dimensions"));
        }
        if self
            .beginning_emotion
            .iter()
            .chain(&self.preferred_emotion)
            .chain([
                &self.emotion_contrast,
                &self.live_blend_coefficient,
                &self.preferred_emotion_strength,
                &self.live_transition_time,
                &self.fixed_dt,
                &self.emotion_strength,
            ])
            .any(|value| !value.is_finite())
        {
            return Err(invalid("emotion post-process parameters must be finite"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EmotionPostProcessor {
    data: EmotionPostProcessData,
    parameters: EmotionPostProcessParameters,
    first_frame: bool,
    working_output: Vec<f32>,
    previous_emotion: Vec<f32>,
    previous_blended_emotion: Vec<f32>,
}

impl EmotionPostProcessor {
    pub fn new(
        data: EmotionPostProcessData,
        parameters: EmotionPostProcessParameters,
    ) -> Result<Self> {
        data.validate()?;
        parameters.validate(&data)?;
        let output_length = data.output_emotion_length;
        Ok(Self {
            data,
            parameters,
            first_frame: true,
            working_output: vec![0.0; output_length],
            previous_emotion: vec![0.0; output_length],
            previous_blended_emotion: vec![0.0; output_length],
        })
    }

    pub fn data(&self) -> &EmotionPostProcessData {
        &self.data
    }

    pub fn parameters(&self) -> &EmotionPostProcessParameters {
        &self.parameters
    }

    pub fn set_parameters(&mut self, parameters: EmotionPostProcessParameters) -> Result<()> {
        parameters.validate(&self.data)?;
        self.parameters = parameters;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.first_frame = true;
        self.previous_emotion.fill(0.0);
        self.previous_blended_emotion.fill(0.0);
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.process_with_preferred(input, None)
    }

    pub fn process_with_preferred(
        &mut self,
        input: &[f32],
        preferred_emotion: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if input.len() != self.data.inference_emotion_length
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid inference emotion input"));
        }
        let preferred = preferred_emotion.unwrap_or(&self.parameters.preferred_emotion);
        if preferred.len() != self.data.output_emotion_length
            || preferred.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid preferred emotion input"));
        }
        let mut inference = input
            .iter()
            .map(|value| value * self.parameters.emotion_contrast)
            .collect::<Vec<_>>();
        softmax(&mut inference);
        for (value, correspondence) in inference.iter_mut().zip(&self.data.emotion_correspondence) {
            if *correspondence == -1 {
                *value = 0.0;
            }
        }
        keep_largest(&mut inference, self.parameters.max_emotions);

        if self.first_frame {
            self.working_output
                .clone_from(&self.parameters.beginning_emotion);
        }
        for (value, correspondence) in inference.iter().zip(&self.data.emotion_correspondence) {
            if let Ok(index) = usize::try_from(*correspondence) {
                self.working_output[index] = *value;
            }
        }

        let blend_source = if self.first_frame {
            &self.parameters.beginning_emotion
        } else {
            &self.previous_emotion
        };
        blend(
            &mut self.working_output,
            blend_source,
            self.parameters.live_blend_coefficient,
        );
        self.previous_emotion.clone_from(&self.working_output);

        if self.parameters.enable_preferred_emotion {
            blend(
                &mut self.working_output,
                preferred,
                self.parameters.preferred_emotion_strength,
            );
        }
        if !self.first_frame {
            let transition_time = self.parameters.live_transition_time.max(1.0e-3);
            let weight = (self.parameters.fixed_dt / transition_time).min(1.0);
            blend(
                &mut self.working_output,
                &self.previous_blended_emotion,
                1.0 - weight,
            );
        }
        self.previous_blended_emotion
            .clone_from(&self.working_output);
        let result = self
            .working_output
            .iter()
            .map(|value| value * self.parameters.emotion_strength)
            .collect();
        self.first_frame = false;
        Ok(result)
    }
}

fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn keep_largest(values: &mut [f32], maximum_count: usize) {
    if maximum_count >= values.len() {
        return;
    }
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| values[*left].total_cmp(&values[*right]));
    for index in indices.into_iter().take(values.len() - maximum_count) {
        values[index] = 0.0;
    }
}

fn blend(values: &mut [f32], source: &[f32], coefficient: f32) {
    values.iter_mut().zip(source).for_each(|(value, source)| {
        *value = (1.0 - coefficient) * *value + coefficient * *source;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::EmotionAudioParameters;
    use std::collections::HashMap;

    fn data() -> EmotionPostProcessData {
        EmotionPostProcessData {
            inference_emotion_length: 3,
            output_emotion_length: 3,
            emotion_correspondence: vec![2, -1, 0],
        }
    }

    fn parameters() -> EmotionPostProcessParameters {
        EmotionPostProcessParameters {
            emotion_contrast: 1.0,
            max_emotions: 3,
            beginning_emotion: vec![0.1, 0.2, 0.3],
            preferred_emotion: vec![0.9, 0.8, 0.7],
            live_blend_coefficient: 0.0,
            enable_preferred_emotion: false,
            preferred_emotion_strength: 0.5,
            live_transition_time: 0.0,
            fixed_dt: 1.0 / 30.0,
            emotion_strength: 1.0,
        }
    }

    #[test]
    fn applies_contrast_nullify_topk_and_correspondence_in_order() {
        let mut params = parameters();
        params.emotion_contrast = 2.0;
        params.max_emotions = 1;
        let mut processor = EmotionPostProcessor::new(data(), params).unwrap();
        let output = processor.process(&[0.0, 1.0, 2.0]).unwrap();
        assert!((output[0] - 0.866_813_3).abs() < 1.0e-6);
        assert_eq!(output[1], 0.2);
        assert_eq!(output[2], 0.0);
    }

    #[test]
    fn blends_preferred_transition_strength_and_reset_state() {
        let mut params = parameters();
        params.live_blend_coefficient = 0.5;
        params.enable_preferred_emotion = true;
        params.preferred_emotion_strength = 0.25;
        params.live_transition_time = 1.0;
        params.fixed_dt = 0.25;
        params.emotion_strength = 2.0;
        let mut processor = EmotionPostProcessor::new(data(), params).unwrap();
        let first = processor.process(&[1.0, 0.0, 0.0]).unwrap();
        let second = processor.process(&[0.0, 0.0, 1.0]).unwrap();
        assert_ne!(first, second);
        processor.reset();
        assert_eq!(processor.process(&[1.0, 0.0, 0.0]).unwrap(), first);
        let override_preferred = processor
            .process_with_preferred(&[1.0, 0.0, 0.0], Some(&[0.0, 0.0, 1.0]))
            .unwrap();
        assert_ne!(override_preferred, first);
    }

    #[test]
    fn validates_all_dimensions_and_finite_values() {
        let mut invalid_data = data();
        invalid_data.emotion_correspondence[0] = 3;
        assert!(EmotionPostProcessor::new(invalid_data, parameters()).is_err());
        let mut invalid_parameters = parameters();
        invalid_parameters.max_emotions = 4;
        assert!(EmotionPostProcessor::new(data(), invalid_parameters).is_err());
        assert!(
            EmotionPostProcessor::new(data(), parameters())
                .unwrap()
                .process(&[0.0])
                .is_err()
        );
    }

    #[test]
    fn converts_model_names_to_strict_correspondence_order() {
        let network = EmotionNetwork {
            audio_params: EmotionAudioParameters { samplerate: 16_000 },
            emotions: vec!["happy".into(), "neutral".into()],
        };
        let mut config = EmotionPostProcessingConfig {
            output_emotion_length: 2,
            emotion_contrast: 1.0,
            live_blend_coef: 0.7,
            preferred_emotion_strength: 0.5,
            fixed_dt: 0.033,
            transition_smoothing: 0.5,
            enable_preferred_emotion: false,
            preferred_emotion: vec![0.0; 2],
            emotion_strength: 0.6,
            max_emotions: 2,
            emotion_correspondence: HashMap::from([("happy".into(), 1), ("neutral".into(), -1)]),
        };
        let (data, parameters) = EmotionPostProcessData::from_model(&network, &config).unwrap();
        assert_eq!(data.emotion_correspondence, [1, -1]);
        assert_eq!(parameters.beginning_emotion, [0.0, 0.0]);
        config.emotion_correspondence.remove("neutral");
        assert!(EmotionPostProcessData::from_model(&network, &config).is_err());
    }
}
