use crate::{AudioFormat, EmotionValues, Error, Result};
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
    pub input_format: AudioFormat,
    pub face: Option<FaceParameters>,
    pub blendshapes: Option<BlendshapeParameters>,
    pub emotion: Option<EmotionParameters>,
    pub emotion_post_processing: Option<EmotionPostProcessing>,
    /// Local request deadline budget, applied separately from the audio header.
    pub timeout: Option<Duration>,
}
impl RequestOptions {
    pub fn new(input_format: AudioFormat) -> Self {
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
impl Default for RequestOptions {
    fn default() -> Self {
        Self::new(AudioFormat::MONO_16KHZ)
    }
}
