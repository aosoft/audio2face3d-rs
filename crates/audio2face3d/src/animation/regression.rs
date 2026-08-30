use crate::common::{
    AudioAccumulator, Binding, BindingSchema, Dimension, ElementType, EmotionAccumulator, Error,
    IoMode, RegressionAudioParameters, RegressionParameters, Result, Shape, WindowProgress,
    WindowProgressParameters,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegressionResultLayout {
    pub skin: usize,
    pub tongue: usize,
    pub jaw: usize,
    pub eyes: usize,
}

impl RegressionResultLayout {
    pub fn total(self) -> Result<usize> {
        [self.skin, self.tongue, self.jaw, self.eyes]
            .into_iter()
            .try_fold(0_usize, |total, size| {
                total.checked_add(size).ok_or(Error::IntegerOverflow {
                    field: "regression_result_size",
                    value: size,
                    target: "usize",
                })
            })
    }

    pub fn split<'a>(self, result: &'a [f32]) -> Result<RegressionResultSlices<'a>> {
        if result.len() != self.total()? {
            return Err(Error::InvalidSchema(format!(
                "regression result has {} elements, expected {}",
                result.len(),
                self.total()?
            )));
        }
        let (skin, rest) = result.split_at(self.skin);
        let (tongue, rest) = rest.split_at(self.tongue);
        let (jaw, eyes) = rest.split_at(self.jaw);
        Ok(RegressionResultSlices {
            skin,
            tongue,
            jaw,
            eyes,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RegressionResultSlices<'a> {
    pub skin: &'a [f32],
    pub tongue: &'a [f32],
    pub jaw: &'a [f32],
    pub eyes: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct RegressionContract {
    pub implicit_emotion_size: usize,
    pub explicit_emotion_size: usize,
    pub emotion_size: usize,
    pub audio_size: usize,
    pub result_layout: RegressionResultLayout,
    pub result_skin_size: usize,
    pub result_tongue_size: usize,
    pub progress: WindowProgress,
}

impl RegressionContract {
    pub fn new(
        parameters: &RegressionParameters,
        audio: &RegressionAudioParameters,
        frame_rate_numerator: usize,
        frame_rate_denominator: usize,
    ) -> Result<Self> {
        let emotion_size = parameters
            .implicit_emotion_len
            .checked_add(parameters.explicit_emotions.len())
            .ok_or(Error::IntegerOverflow {
                field: "regression_emotion_size",
                value: parameters.explicit_emotions.len(),
                target: "usize",
            })?;
        let stride_numerator =
            audio
                .samplerate
                .checked_mul(frame_rate_denominator)
                .ok_or(Error::IntegerOverflow {
                    field: "regression_stride_numerator",
                    value: frame_rate_denominator,
                    target: "usize",
                })?;
        let target_offset =
            i64::try_from(audio.buffer_ofs).map_err(|_| Error::IntegerOverflow {
                field: "regression_target_offset",
                value: audio.buffer_ofs,
                target: "i64",
            })?;
        Ok(Self {
            implicit_emotion_size: parameters.implicit_emotion_len,
            explicit_emotion_size: parameters.explicit_emotions.len(),
            emotion_size,
            audio_size: audio.buffer_len,
            result_layout: RegressionResultLayout {
                skin: parameters.num_shapes_skin,
                tongue: parameters.num_shapes_tongue,
                jaw: parameters.result_jaw_size,
                eyes: parameters.result_eyes_size,
            },
            result_skin_size: parameters.num_verts_skin.checked_mul(3).ok_or(
                Error::IntegerOverflow {
                    field: "regression_result_skin_size",
                    value: parameters.num_verts_skin,
                    target: "usize",
                },
            )?,
            result_tongue_size: parameters.num_verts_tongue.checked_mul(3).ok_or(
                Error::IntegerOverflow {
                    field: "regression_result_tongue_size",
                    value: parameters.num_verts_tongue,
                    target: "usize",
                },
            )?,
            progress: WindowProgress::new(WindowProgressParameters {
                window_size: audio.buffer_len,
                start_offset: -target_offset,
                target_offset,
                stride_numerator,
                stride_denominator: frame_rate_numerator,
            })?,
        })
    }

    pub fn binding_schema(&self) -> Result<BindingSchema> {
        let tensor = |name: &str, mode, size| -> Result<Binding> {
            Ok(Binding {
                name: name.into(),
                mode,
                element_type: ElementType::F32,
                shape: Shape::new(vec![
                    Dimension::Batch,
                    Dimension::Fixed(1),
                    Dimension::Fixed(size),
                ])?,
            })
        };
        BindingSchema::new(vec![
            tensor("emotion", IoMode::Input, self.emotion_size)?,
            tensor("input", IoMode::Input, self.audio_size)?,
            tensor("result", IoMode::Output, self.result_layout.total()?)?,
        ])
    }

    pub fn prepare_frame(
        &self,
        frame_index: usize,
        audio: &AudioAccumulator,
        emotions: &EmotionAccumulator,
        implicit_emotion: &[f32],
        input_strength: f32,
    ) -> Result<RegressionFrameInput> {
        if implicit_emotion.len() + emotions.state().emotion_size != self.emotion_size {
            return Err(Error::InvalidSchema(
                "implicit and explicit emotion dimensions do not match binding".into(),
            ));
        }
        let window = self.progress.window(frame_index)?;
        let mut emotion = implicit_emotion.to_vec();
        emotion.extend(
            emotions.read(window.target).map_err(|error| {
                Error::InvalidSchema(format!("emotion is unavailable: {error}"))
            })?,
        );
        Ok(RegressionFrameInput {
            timestamp: window.target,
            next_timestamp: self.progress.window(frame_index + 1)?.target,
            audio: audio.read(window.start, self.audio_size, input_strength)?,
            emotion,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegressionFrameInput {
    pub timestamp: i64,
    pub next_timestamp: i64,
    pub audio: Vec<f32>,
    pub emotion: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> RegressionParameters {
        RegressionParameters {
            implicit_emotion_len: 2,
            explicit_emotions: vec!["joy".into()],
            default_emotion: vec![0.0],
            num_shapes_skin: 3,
            num_shapes_tongue: 2,
            num_verts_skin: 1,
            num_verts_tongue: 1,
            result_jaw_size: 2,
            result_eyes_size: 1,
        }
    }

    #[test]
    fn schema_and_frame_input_follow_regression_contract() {
        let contract = RegressionContract::new(
            &parameters(),
            &RegressionAudioParameters {
                buffer_len: 4,
                buffer_ofs: 2,
                samplerate: 4,
            },
            2,
            1,
        )
        .unwrap();
        let schema = contract.binding_schema().unwrap();
        assert_eq!(
            schema
                .bindings()
                .iter()
                .map(|binding| binding.name.as_str())
                .collect::<Vec<_>>(),
            ["emotion", "input", "result"]
        );
        assert_eq!(contract.result_layout.total().unwrap(), 8);

        let audio = AudioAccumulator::new(2, 0).unwrap();
        audio.accumulate(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        audio.close().unwrap();
        let emotions = EmotionAccumulator::new(1, 2).unwrap();
        emotions.accumulate(0, &[0.25]).unwrap();
        emotions.accumulate(2, &[0.75]).unwrap();
        emotions.close().unwrap();
        let frame = contract
            .prepare_frame(0, &audio, &emotions, &[0.1, 0.2], 2.0)
            .unwrap();
        assert_eq!(frame.audio, [0.0, 0.0, 2.0, 4.0]);
        assert_eq!(frame.emotion, [0.1, 0.2, 0.25]);
        assert_eq!((frame.timestamp, frame.next_timestamp), (0, 2));
    }
}
