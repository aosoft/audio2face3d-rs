#![cfg(feature = "animation")]
#![cfg(feature = "tensorrt")]

use audio2face3d::common::{Dimension, ElementType, IoMode, Shape};
use audio2face3d::cuda::GpuDevice;
use audio2face3d::tensorrt::{BindingBuffer, DeviceBindings, TensorRtSession};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::Arc;

fn read_reference_tensors(path: &Path) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).unwrap();
    let mut offset = 0;
    let read_u32 = |offset: &mut usize| {
        let end = *offset + 4;
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        value as usize
    };
    let count = read_u32(&mut offset);
    let mut tensors = HashMap::new();
    for _ in 0..count {
        let name_len = read_u32(&mut offset);
        let name = String::from_utf8(bytes[offset..offset + name_len].to_vec()).unwrap();
        offset += name_len;
        let len = read_u32(&mut offset);
        let values = bytes[offset..offset + len * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        offset += len * 4;
        tensors.insert(name, values);
    }
    assert_eq!(offset, bytes.len());
    tensors
}

fn reference_input_shape(shape: &Shape, element_count: usize) -> Vec<i64> {
    let mut has_variable_axis = false;
    let mut fixed_product = 1_usize;
    for dimension in shape.dimensions() {
        match dimension {
            Dimension::Fixed(value) => {
                fixed_product = fixed_product.checked_mul(*value).unwrap();
            }
            Dimension::Batch | Dimension::Dynamic { .. } => {
                assert!(
                    !std::mem::replace(&mut has_variable_axis, true),
                    "reference fixture cannot infer multiple dynamic dimensions"
                );
            }
        }
    }

    let variable_value = if has_variable_axis {
        assert_eq!(element_count % fixed_product, 0);
        element_count / fixed_product
    } else {
        assert_eq!(element_count, fixed_product);
        1
    };
    assert!(variable_value > 0);

    shape
        .dimensions()
        .iter()
        .map(|dimension| {
            let value = match dimension {
                Dimension::Fixed(value) => *value,
                Dimension::Batch => variable_value,
                Dimension::Dynamic { min, max } => {
                    assert!((*min..=*max).contains(&variable_value));
                    variable_value
                }
            };
            i64::try_from(value).unwrap()
        })
        .collect()
}

#[derive(Clone)]
struct ReferenceTensorSpec {
    name: String,
    mode: IoMode,
    element_type: ElementType,
    shape: Shape,
    len: usize,
}

#[test]
#[ignore = "requires a local C++ fixture, TensorRT engine, and CUDA device"]
fn cpp_fixture_matches_rust_tensor_rt_engine() {
    // This intentionally remains a low-level TensorRT adapter test. It
    // validates binding metadata and raw engine enqueue independently of the
    // public model-specific executor factories.
    let engine = env::var_os("AUDIO2FACE3D_REFERENCE_ENGINE")
        .expect("AUDIO2FACE3D_REFERENCE_ENGINE must name the generated TensorRT engine");
    let fixture = env::var_os("AUDIO2FACE3D_REFERENCE_TENSORS")
        .expect("AUDIO2FACE3D_REFERENCE_TENSORS must name the matching C++ tensor fixture");
    let expected = read_reference_tensors(Path::new(&fixture));
    let device = GpuDevice::new(0).unwrap();
    let stream = device.create_stream().unwrap();
    let mut session = TensorRtSession::load(Arc::clone(&device), Path::new(&engine)).unwrap();
    let specs = session
        .metadata()
        .bindings()
        .iter()
        .map(|binding| ReferenceTensorSpec {
            name: binding.name.clone(),
            mode: binding.mode,
            element_type: binding.element_type,
            shape: binding.shape.clone(),
            len: expected
                .get(&binding.name)
                .unwrap_or_else(|| panic!("fixture has no tensor `{}`", binding.name))
                .len(),
        })
        .collect::<Vec<_>>();
    let mut buffers = specs
        .iter()
        .map(|spec| device.allocate::<f32>(spec.len).unwrap())
        .collect::<Vec<_>>();
    for (spec, buffer) in specs.iter().zip(&mut buffers) {
        if spec.mode == IoMode::Input {
            buffer.copy_from(&expected[&spec.name], &stream).unwrap();
        }
    }

    let mut bindings = DeviceBindings::new();
    for (spec, buffer) in specs.iter().zip(&buffers) {
        bindings
            .insert(
                &spec.name,
                BindingBuffer::from_view(buffer.view(), spec.element_type),
            )
            .unwrap();
        if spec.mode == IoMode::Input {
            bindings
                .set_input_shape(&spec.name, reference_input_shape(&spec.shape, spec.len))
                .unwrap();
        }
    }
    session
        .enqueue(0, &bindings, &stream)
        .unwrap()
        .synchronize()
        .unwrap();
    drop(bindings);

    for (spec, buffer) in specs.iter().zip(&buffers) {
        if spec.mode == IoMode::Output {
            let mut actual = vec![0.0; spec.len];
            buffer.copy_to(&mut actual, &stream).unwrap();
            for (index, (actual, expected)) in actual.iter().zip(&expected[&spec.name]).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1.0e-3,
                    "{}[{index}]: {actual} != {expected}",
                    spec.name
                );
            }
        }
    }
}
