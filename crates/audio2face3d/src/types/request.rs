use crate::types::{AudioFormat, EmotionValues, Error, Result};
use std::{collections::BTreeMap, time::Duration};

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct FaceParameters {
    pub upper_face_smoothing: Option<f32>,
    pub lower_face_smoothing: Option<f32>,
    pub upper_face_strength: Option<f32>,
    pub lower_face_strength: Option<f32>,
    pub face_mask_level: Option<f32>,
    pub face_mask_softness: Option<f32>,
    pub skin_strength: Option<f32>,
    pub blink_strength: Option<f32>,
    pub blink_offset: Option<f32>,
    pub eyelid_open_offset: Option<f32>,
    pub lip_open_offset: Option<f32>,
    pub tongue_strength: Option<f32>,
    pub tongue_height_offset: Option<f32>,
    pub tongue_depth_offset: Option<f32>,
}
impl FaceParameters {
    pub fn validate(&self) -> Result<()> {
        for value in [
            self.upper_face_smoothing,
            self.lower_face_smoothing,
            self.upper_face_strength,
            self.lower_face_strength,
            self.face_mask_level,
            self.face_mask_softness,
            self.skin_strength,
            self.blink_strength,
            self.blink_offset,
            self.eyelid_open_offset,
            self.lip_open_offset,
            self.tongue_strength,
            self.tongue_height_offset,
            self.tongue_depth_offset,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() {
                return Err(Error::invalid("face parameter must be finite"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct BlendshapeParameters {
    pub multipliers: BTreeMap<String, f32>,
    pub offsets: BTreeMap<String, f32>,
    pub clamp: Option<bool>,
}
impl BlendshapeParameters {
    pub fn validate(&self) -> Result<()> {
        if self
            .multipliers
            .iter()
            .chain(self.offsets.iter())
            .any(|(name, value)| name.is_empty() || !value.is_finite())
        {
            return Err(Error::invalid(
                "blendshape names/values must be non-empty/finite",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct EmotionParameters {
    pub transition_time: Option<f32>,
    pub beginning: EmotionValues,
}
impl EmotionParameters {
    pub fn validate(&self) -> Result<()> {
        if self
            .transition_time
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            return Err(Error::invalid(
                "emotion transition time must be positive and finite",
            ));
        }
        // Validate directly to avoid cloning a sparse map on each request.
        for (name, value) in &self.beginning {
            if name.is_empty() || !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(Error::invalid("invalid beginning emotion"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct EmotionPostProcessing {
    pub contrast: Option<f32>,
    pub smoothing: Option<f32>,
    pub use_preferred: Option<bool>,
    pub preferred_strength: Option<f32>,
    pub strength: Option<f32>,
    pub max_emotions: Option<u32>,
}
impl EmotionPostProcessing {
    pub fn validate(&self) -> Result<()> {
        for (value, min, max) in [
            (self.contrast, 0.3, 3.0),
            (self.smoothing, 0.0, 1.0),
            (self.preferred_strength, 0.0, 1.0),
            (self.strength, 0.0, 1.0),
        ] {
            if value.is_some_and(|v| !v.is_finite() || !(min..=max).contains(&v)) {
                return Err(Error::invalid(
                    "emotion post-processing value outside supported range",
                ));
            }
        }
        if self.max_emotions.is_some_and(|v| !(1..=6).contains(&v)) {
            return Err(Error::invalid("max emotions must be in 1..=6"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RequestOptions {
    pub(crate) input_format: AudioFormat,
    pub(crate) face: Option<FaceParameters>,
    pub(crate) blendshapes: Option<BlendshapeParameters>,
    pub(crate) emotion: Option<EmotionParameters>,
    pub(crate) emotion_post_processing: Option<EmotionPostProcessing>,
    /// Local request deadline budget, applied separately from the audio header.
    pub(crate) timeout: Option<Duration>,
}
impl RequestOptions {
    pub(crate) fn new(input_format: AudioFormat) -> Self {
        Self {
            input_format,
            face: None,
            blendshapes: None,
            emotion: None,
            emotion_post_processing: None,
            timeout: None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(p) = &self.face {
            p.validate()?;
        }
        if let Some(p) = &self.blendshapes {
            p.validate()?;
        }
        if let Some(p) = &self.emotion {
            p.validate()?;
        }
        if let Some(p) = &self.emotion_post_processing {
            p.validate()?;
        }
        Ok(())
    }
}
#[cfg(any(test, feature = "native"))]
impl RequestOptions {
    pub(crate) fn default() -> Self {
        Self::new(AudioFormat::MONO_16KHZ)
    }
}

/// Consuming builder; validation runs in build before resources are started.
#[derive(Clone, Debug)]
#[must_use]
pub struct RequestOptionsBuilder {
    config: RequestOptions,
}
impl RequestOptions {
    pub fn builder(input_format: AudioFormat) -> RequestOptionsBuilder {
        RequestOptionsBuilder {
            config: RequestOptions::new(input_format),
        }
    }
}
impl RequestOptionsBuilder {
    pub fn input_format(mut self, value: AudioFormat) -> Self {
        self.config.input_format = value;
        self
    }
    pub fn face(mut self, value: FaceParameters) -> Self {
        self.config.face = Some(value);
        self
    }
    pub fn optional_face(mut self, value: Option<FaceParameters>) -> Self {
        self.config.face = value;
        self
    }
    pub fn blendshapes(mut self, value: BlendshapeParameters) -> Self {
        self.config.blendshapes = Some(value);
        self
    }
    pub fn optional_blendshapes(mut self, value: Option<BlendshapeParameters>) -> Self {
        self.config.blendshapes = value;
        self
    }
    pub fn emotion(mut self, value: EmotionParameters) -> Self {
        self.config.emotion = Some(value);
        self
    }
    pub fn optional_emotion(mut self, value: Option<EmotionParameters>) -> Self {
        self.config.emotion = value;
        self
    }
    pub fn emotion_post_processing(mut self, value: EmotionPostProcessing) -> Self {
        self.config.emotion_post_processing = Some(value);
        self
    }
    pub fn optional_emotion_post_processing(
        mut self,
        value: Option<EmotionPostProcessing>,
    ) -> Self {
        self.config.emotion_post_processing = value;
        self
    }
    pub fn timeout(mut self, value: Duration) -> Self {
        self.config.timeout = Some(value);
        self
    }
    pub fn optional_timeout(mut self, value: Option<Duration>) -> Self {
        self.config.timeout = value;
        self
    }
    pub fn build(self) -> Result<RequestOptions> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl RequestOptions {
    pub fn into_builder(self) -> RequestOptionsBuilder {
        RequestOptionsBuilder { config: self }
    }
    pub fn input_format(&self) -> AudioFormat {
        self.input_format
    }
    pub fn face(&self) -> &Option<FaceParameters> {
        &self.face
    }
    pub fn blendshapes(&self) -> &Option<BlendshapeParameters> {
        &self.blendshapes
    }
    pub fn emotion(&self) -> &Option<EmotionParameters> {
        &self.emotion
    }
    pub fn emotion_post_processing(&self) -> &Option<EmotionPostProcessing> {
        &self.emotion_post_processing
    }
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}
