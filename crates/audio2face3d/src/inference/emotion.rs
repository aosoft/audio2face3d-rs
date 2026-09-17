//! Sparse input timeline and A2E -> A2F dependency ordering, without wire metadata.
use crate::inference::{config::Config, worker::wait};
use crate::types::{
    EmotionKeyframe, EmotionTrace, EmotionValues, MediaTime, RequestOptions, SamplePosition,
};
use crate::types::{Error, ErrorKind};
use crate::{
    Model, ModelKind, ModelParameters,
    audio2emotion::{
        EmotionExecutor, EmotionExecutorCreationParameters, EmotionTrackResources, PostProcessData,
        PostProcessParams,
        classifier::{
            ClassifierEmotionExecutor, ClassifierEmotionExecutorCreationParameters,
            ClassifierEmotionExecutorFactory,
        },
    },
    audio2x::{AudioAccumulator, EmotionAccumulator, ExecutionState, FrameRate},
    common::NetworkDocument,
};
use std::{collections::BTreeMap, ops::ControlFlow, sync::Arc};

fn internal(e: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::Inference, format!("emotion backend: {e}"))
}
fn invalid(e: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, e.into())
}
const NAMES: [&str; 10] = [
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "outofbreath",
    "pain",
    "sadness",
];

struct EmotionRecord {
    input: Vec<f32>,
    mixed: Vec<f32>,
    smooth: Vec<f32>,
}

pub struct Stage {
    classifier: Option<ClassifierEmotionExecutor>,
    audio: Arc<AudioAccumulator>,
    preferred: Arc<EmotionAccumulator>,
    output: Arc<EmotionAccumulator>,
    names: Vec<String>,
    keys: BTreeMap<i64, Vec<f32>>,
    current: Vec<f32>,

    frame: u64,
    fed: u64,
    consumed: i64,
    last_input: i64,
    complete: bool,
    smoothing: f32,
    previous: Option<Vec<f32>>,
    records: BTreeMap<i64, EmotionRecord>,
    max_time: f64,
}
impl Stage {
    pub fn load(
        config: &Config,
        header: &RequestOptions,
        names: &[String],
        defaults: &[f32],
        output: Arc<EmotionAccumulator>,
    ) -> Result<Self, Error> {
        if names.iter().map(String::as_str).collect::<Vec<_>>() != NAMES {
            return Err(invalid("unsupported model emotion ordering"));
        }
        let audio = Arc::new(AudioAccumulator::new(60_000, 0).map_err(internal)?);
        let preferred = Arc::new(EmotionAccumulator::new(names.len(), 30).map_err(internal)?);
        let mut params = PostProcessParams {
            beginning_emotion: defaults.to_vec(),
            preferred_emotion: defaults.to_vec(),
            ..Default::default()
        };
        let mut classifier_model = None;
        if let Some(path) = &config.emotion_model {
            let model = Model::load(path).map_err(internal)?;
            if model.kind() != ModelKind::Emotion || model.sample_rate() != 16000 {
                return Err(invalid("expected a 16000 Hz Audio2Emotion model"));
            }
            let ModelParameters::Emotion(c) = model.parameters(0).map_err(internal)? else {
                return Err(internal("missing emotion config"));
            };
            params = PostProcessParams {
                emotion_contrast: c.emotion_contrast,
                max_emotions: c.max_emotions,
                beginning_emotion: vec![0.0; c.output_emotion_length],
                preferred_emotion: c.preferred_emotion.clone(),
                live_blend_coefficient: c.live_blend_coef,
                enable_preferred_emotion: c.enable_preferred_emotion,
                preferred_emotion_strength: c.preferred_emotion_strength,
                live_transition_time: c.transition_smoothing,
                fixed_dt: c.fixed_dt,
                emotion_strength: c.emotion_strength,
            };
            classifier_model = Some(model);
        }
        if let Some(p) = &header.emotion {
            if let Some(v) = p.transition_time {
                if !v.is_finite() || v <= 0.0 {
                    return Err(invalid("live_transition_time must be positive and finite"));
                }
                params.live_transition_time = v;
            }
            if !p.beginning.is_empty() {
                params.beginning_emotion = values(&p.beginning)?;
            }
        }
        if let Some(p) = &header.emotion_post_processing {
            for (v, lo, hi, name, target) in [
                (
                    p.contrast,
                    0.3,
                    3.0,
                    "emotion_contrast",
                    &mut params.emotion_contrast,
                ),
                (
                    p.smoothing,
                    0.0,
                    1.0,
                    "live_blend_coef",
                    &mut params.live_blend_coefficient,
                ),
                (
                    p.preferred_strength,
                    0.0,
                    1.0,
                    "preferred_emotion_strength",
                    &mut params.preferred_emotion_strength,
                ),
                (
                    p.strength,
                    0.0,
                    1.0,
                    "emotion_strength",
                    &mut params.emotion_strength,
                ),
            ] {
                if let Some(v) = v {
                    if !v.is_finite() || !(lo..=hi).contains(&v) {
                        return Err(invalid(format!("{name} out of range")));
                    }
                    *target = v;
                }
            }
            if let Some(v) = p.use_preferred {
                params.enable_preferred_emotion = v;
            }
            if let Some(v) = p.max_emotions {
                if !(1..=6).contains(&v) {
                    return Err(invalid("max_emotions must be 1..6"));
                }
                params.max_emotions = v as usize;
            }
        }
        // The classifier returns the mixed signal. Apply the final transition
        // separately so metadata can distinguish it from the actual A2F input.
        let smoothing = (params.fixed_dt / params.live_transition_time.max(0.001)).min(1.0);
        params.live_transition_time = 0.0;
        let beginning = if classifier_model.is_some() {
            params.preferred_emotion.clone()
        } else {
            params.beginning_emotion.clone()
        };
        let classifier = if let Some(model) = classifier_model {
            let NetworkDocument::Emotion(network) = model.network() else {
                return Err(internal("missing emotion network"));
            };
            let ModelParameters::Emotion(c) = model.parameters(0).map_err(internal)? else {
                return Err(internal("missing emotion config"));
            };
            if c.output_emotion_length != names.len() {
                return Err(invalid("A2E output dimension differs from A2F"));
            }
            let data = PostProcessData {
                inference_emotion_length: network.emotions.len(),
                output_emotion_length: c.output_emotion_length,
                emotion_correspondence: network
                    .emotions
                    .iter()
                    .map(|n| {
                        c.emotion_correspondence
                            .get(n)
                            .and_then(|&v| i32::try_from(v).ok())
                            .ok_or_else(|| internal("missing emotion correspondence"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            };
            Some(
                wait(ClassifierEmotionExecutorFactory::load(
                    ClassifierEmotionExecutorCreationParameters {
                        model_path: config.emotion_model.clone().unwrap(),
                        common: EmotionExecutorCreationParameters {
                            tracks: vec![EmotionTrackResources {
                                audio: audio.clone(),
                            }],
                            device_ordinal: config.device as i32,
                        },
                        input_strength: 1.0,
                        buffer_length: 60_000,
                        frame_rate: FrameRate::new(30, 1).map_err(internal)?,
                        inferences_to_skip: 0,
                        post_process_data: data,
                        post_process_params: params,
                        preferred_emotions: vec![preferred.clone()],
                    },
                ))
                .map_err(internal)?,
            )
        } else {
            if header.emotion_post_processing.is_some() {
                tracing::warn!(
                    "A2E mixing parameters require --emotion-model; direct input emotion mode"
                );
            }
            None
        };
        Ok(Self {
            classifier,
            audio,
            preferred,
            output,
            names: names.to_vec(),
            keys: BTreeMap::new(),
            current: beginning,

            frame: 0,
            fed: 0,
            consumed: -1,
            last_input: -1,
            complete: false,
            smoothing,
            previous: None,
            records: BTreeMap::new(),
            max_time: f64::from(config.max_audio_seconds),
        })
    }
    pub fn push_keys(&mut self, keys: Vec<EmotionKeyframe>) -> Result<(), Error> {
        for key in keys {
            if !key.time().as_seconds().is_finite()
                || key.time().as_seconds() < 0.0
                || key.time().as_seconds() > self.max_time
            {
                return Err(invalid("emotion time_code outside clip limit"));
            }
            let time = (key.time().as_seconds() * 16000.0).round() as i64;
            if time <= self.last_input {
                return Err(invalid("emotion key targets an already consumed time"));
            }
            let value = values(key.values())?;
            if !self.keys.contains_key(&time) && self.keys.len() >= 4096 {
                return Err(Error::new(
                    ErrorKind::LimitExceeded,
                    "emotion key limit exceeded",
                ));
            }
            self.keys.insert(time, value);
        }
        Ok(())
    }
    pub fn feed(&mut self, samples: &[f32]) -> Result<(), Error> {
        self.fed += samples.len() as u64;
        if self.classifier.is_some() {
            self.audio.accumulate(samples).map_err(internal)?;
        }
        while self.frame * 16000 / 30 < self.fed {
            let time = (self.frame * 16000 / 30) as i64;
            while let Some((&t, _)) = self.keys.first_key_value() {
                if t > time {
                    break;
                }
                self.current = self.keys.pop_first().unwrap().1;
            }
            self.last_input = time;
            if self.classifier.is_some() {
                self.preferred
                    .accumulate(time, &self.current)
                    .map_err(internal)?;
            } else {
                let mixed = self.current.clone();
                self.record(time, mixed)?;
            }
            self.frame += 1;
        }
        Ok(())
    }
    fn record(&mut self, time: i64, mixed: Vec<f32>) -> Result<(), Error> {
        if mixed.len() != self.names.len() || mixed.iter().any(|v| !v.is_finite()) {
            return Err(internal("invalid emotion result"));
        }
        if self.records.len() >= 256 {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "emotion result queue exceeds 256 frames",
            ));
        }
        let input = if self.classifier.is_some() {
            self.preferred.read(time).map_err(internal)?
        } else {
            self.current.clone()
        };
        let smooth = if let Some(previous) = &self.previous {
            mixed
                .iter()
                .zip(previous)
                .map(|(v, p)| p + (v - p) * self.smoothing)
                .collect::<Vec<_>>()
        } else {
            mixed.clone()
        };
        self.previous = Some(smooth.clone());
        self.output.accumulate(time, &smooth).map_err(internal)?;
        self.consumed = time;
        self.records.insert(
            time,
            EmotionRecord {
                input,
                mixed,
                smooth,
            },
        );
        Ok(())
    }
    pub fn finish(&mut self) -> Result<(), Error> {
        if self.classifier.is_some() {
            self.audio.close().map_err(internal)?;
            self.preferred.close().map_err(internal)?;
        } else {
            self.output.close().map_err(internal)?;
            self.complete = true;
        }
        Ok(())
    }
    pub fn tick(&mut self) -> Result<(), Error> {
        if self.complete || self.classifier.is_none() {
            return Ok(());
        }
        let classifier = self.classifier.as_mut().unwrap();
        let mut frames = Vec::new();
        let mut error = None;
        let execution = classifier
            .execute(&mut |r| {
                if frames.len() >= 256 {
                    error = Some(internal("A2E callback batch exceeds 256 frames"));
                    return ControlFlow::Break(());
                }
                let mut values = vec![0.0; r.emotions.values.len()];
                match r.emotions.copy_to(&mut values) {
                    Ok(()) => frames.push((r.metadata.timestamp, values)),
                    Err(e) => error = Some(internal(e)),
                }
                ControlFlow::Continue(())
            })
            .map_err(internal)?;
        let report = wait(execution.wait_all()).map_err(internal)?;
        if let Some(error) = error {
            return Err(error);
        }
        let next = classifier.next_audio_sample_to_read(0).map_err(internal)?;
        self.audio
            .drop_samples_before(next.min(self.audio.nb_accumulated_samples()))
            .map_err(internal)?;
        for (time, values) in frames {
            self.record(time, values)?;
        }
        if self.consumed >= 0 {
            self.preferred
                .drop_before(self.consumed)
                .map_err(internal)?;
        }
        if report.state == ExecutionState::Complete {
            self.output.close().map_err(internal)?;
            self.complete = true;
        }
        Ok(())
    }
    pub fn is_complete(&self) -> bool {
        self.complete
    }
    pub fn metadata(&mut self, time: i64) -> Result<EmotionTrace, Error> {
        let Some(EmotionRecord {
            input,
            mixed,
            smooth,
        }) = self.records.remove(&time)
        else {
            return Err(internal(format!("missing emotion metadata at {time}")));
        };
        self.output.drop_before(time).map_err(internal)?;
        let timestamp = MediaTime::from_samples(SamplePosition(time as u64), 16000)?;
        let message = |v: Vec<f32>| {
            EmotionKeyframe::new(timestamp, self.names.iter().cloned().zip(v).collect())
        };
        Ok(EmotionTrace {
            input: vec![message(input)?],
            mixed: if self.classifier.is_some() {
                vec![message(mixed)?]
            } else {
                vec![]
            },
            smoothed: vec![message(smooth)?],
        })
    }
}
fn values(map: &EmotionValues) -> Result<Vec<f32>, Error> {
    let mut result = vec![0.0; NAMES.len()];
    for (name, &value) in map {
        let index = NAMES
            .iter()
            .position(|n| *n == name)
            .ok_or_else(|| invalid(format!("unknown emotion: {name}")))?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(invalid("emotion strength must be 0..1"));
        }
        result[index] = value;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn future_and_duplicate_keys_do_not_close_or_rewrite_consumed_time() {
        let output = Arc::new(EmotionAccumulator::new(10, 30).unwrap());
        let names = NAMES.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        let mut stage = Stage::load(
            &Config::default(),
            &RequestOptions::default(),
            &names,
            &[0.0; 10],
            output.clone(),
        )
        .unwrap();
        let key = |time, value| {
            EmotionKeyframe::new(
                MediaTime::from_seconds(time).unwrap(),
                BTreeMap::from([("joy".into(), value)]),
            )
            .unwrap()
        };
        stage
            .push_keys(vec![key(1.0, 0.3), key(0.0, 0.5), key(1.0, 0.8)])
            .unwrap();
        stage.feed(&[0.0; 534]).unwrap();
        assert_eq!(output.read(0).unwrap()[6], 0.5);
        assert!(!output.state().closed);
        assert!(stage.push_keys(vec![key(0.0, 0.9)]).is_err());
        stage.metadata(0).unwrap();
        stage.metadata(533).unwrap();
        for _ in 0..30 {
            stage.feed(&[0.0; 534]).unwrap();
            let times = stage.records.keys().copied().collect::<Vec<_>>();
            for t in times {
                stage.metadata(t).unwrap();
            }
        }
        assert_eq!(stage.current[6], 0.8);
        stage.finish().unwrap();
        assert!(output.state().closed);
    }
}
