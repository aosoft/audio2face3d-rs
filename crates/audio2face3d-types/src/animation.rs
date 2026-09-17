use crate::{Error, MediaTime, Result};
use std::{collections::HashSet, sync::Arc};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LayoutId(pub u64);

#[derive(Debug, PartialEq, Eq)]
pub struct CurveLayout {
    id: LayoutId,
    names: Vec<String>,
}
impl CurveLayout {
    pub fn new(id: LayoutId, names: Vec<String>) -> Result<Self> {
        let mut seen = HashSet::with_capacity(names.len());
        if names
            .iter()
            .any(|name| name.is_empty() || !seen.insert(name.as_str()))
        {
            return Err(Error::invalid("layout names must be non-empty and unique"));
        }
        Ok(Self { id, names })
    }
    pub fn id(&self) -> LayoutId {
        self.id
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn into_names(self) -> Vec<String> {
        self.names
    }
}
/// The same ordered-channel contract can describe emotion output layouts.
pub type EmotionLayout = CurveLayout;

#[derive(Debug, PartialEq)]
pub struct CurveFrame {
    layout: Arc<CurveLayout>,
    time: MediaTime,
    values: Vec<f32>,
}
impl CurveFrame {
    pub fn new(layout: Arc<CurveLayout>, time: MediaTime, values: Vec<f32>) -> Result<Self> {
        if values.len() != layout.names().len() || !values.iter().all(|v| v.is_finite()) {
            return Err(Error::invalid(
                "curve values must be finite and match their layout",
            ));
        }
        Ok(Self {
            layout,
            time,
            values,
        })
    }
    pub fn layout(&self) -> &Arc<CurveLayout> {
        &self.layout
    }
    pub fn time(&self) -> MediaTime {
        self.time
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn into_parts(self) -> (Arc<CurveLayout>, MediaTime, Vec<f32>) {
        (self.layout, self.time, self.values)
    }
}
