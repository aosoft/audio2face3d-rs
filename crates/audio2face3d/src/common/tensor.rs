use crate::common::{Error, Result};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    F32,
    F16,
    I64,
    U64,
    Bool,
    Raw,
}

impl ElementType {
    pub const fn byte_width(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 => Some(2),
            Self::I64 | Self::U64 => Some(8),
            Self::Bool => Some(1),
            Self::Raw => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    Fixed(usize),
    Batch,
    Dynamic { min: usize, max: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<Dimension>);

impl Shape {
    pub fn new(dimensions: impl Into<Vec<Dimension>>) -> Result<Self> {
        let dimensions = dimensions.into();
        for dimension in &dimensions {
            match dimension {
                Dimension::Fixed(0) => {
                    return Err(Error::InvalidSchema(
                        "fixed dimensions must be non-zero".into(),
                    ));
                }
                Dimension::Dynamic { min, max } if min > max => {
                    return Err(Error::InvalidSchema(format!(
                        "dynamic minimum {min} exceeds maximum {max}"
                    )));
                }
                _ => {}
            }
        }
        Ok(Self(dimensions))
    }

    pub fn dimensions(&self) -> &[Dimension] {
        &self.0
    }

    pub fn element_count(&self, batch: usize, dynamic: &[usize]) -> Result<usize> {
        let mut dynamic_index = 0;
        let mut count = 1_usize;
        for dimension in &self.0 {
            let value = match *dimension {
                Dimension::Fixed(value) => value,
                Dimension::Batch => batch,
                Dimension::Dynamic { min, max } => {
                    let value = *dynamic
                        .get(dynamic_index)
                        .ok_or_else(|| Error::InvalidSchema("missing dynamic dimension".into()))?;
                    dynamic_index += 1;
                    if !(min..=max).contains(&value) {
                        return Err(Error::InvalidSchema(format!(
                            "dynamic dimension {value} is outside {min}..={max}"
                        )));
                    }
                    value
                }
            };
            count = count.checked_mul(value).ok_or(Error::IntegerOverflow {
                field: "tensor_element_count",
                value,
                target: "usize",
            })?;
        }
        if dynamic_index != dynamic.len() {
            return Err(Error::InvalidSchema("too many dynamic dimensions".into()));
        }
        Ok(count)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoMode {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub mode: IoMode,
    pub element_type: ElementType,
    pub shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingSchema(Vec<Binding>);

impl BindingSchema {
    pub fn new(bindings: impl Into<Vec<Binding>>) -> Result<Self> {
        let bindings = bindings.into();
        let mut names = HashSet::with_capacity(bindings.len());
        for binding in &bindings {
            if binding.name.is_empty() {
                return Err(Error::InvalidSchema(
                    "binding names must not be empty".into(),
                ));
            }
            if !names.insert(binding.name.clone()) {
                return Err(Error::DuplicateBinding(binding.name.clone()));
            }
        }
        Ok(Self(bindings))
    }

    pub fn bindings(&self) -> &[Binding] {
        &self.0
    }

    pub fn get(&self, name: &str) -> Option<&Binding> {
        self.0.iter().find(|binding| binding.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_validates_dynamic_dimensions() {
        let shape = Shape::new(vec![
            Dimension::Batch,
            Dimension::Dynamic { min: 2, max: 4 },
            Dimension::Fixed(3),
        ])
        .unwrap();
        assert_eq!(shape.element_count(2, &[4]).unwrap(), 24);
        assert!(shape.element_count(2, &[5]).is_err());
    }

    #[test]
    fn schema_rejects_duplicate_names() {
        let shape = Shape::new(vec![Dimension::Fixed(1)]).unwrap();
        let binding = Binding {
            name: "input".into(),
            mode: IoMode::Input,
            element_type: ElementType::F32,
            shape,
        };
        assert!(BindingSchema::new(vec![binding.clone(), binding]).is_err());
    }
}
