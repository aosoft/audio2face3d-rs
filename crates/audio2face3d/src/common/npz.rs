use crate::common::{Error, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub(crate) struct NpzArchive {
    archive: zip::ZipArchive<File>,
}

impl NpzArchive {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|error| invalid(format!("unable to open {}: {error}", path.display())))?;
        let archive = zip::ZipArchive::new(file)
            .map_err(|error| invalid(format!("invalid NPZ {}: {error}", path.display())))?;
        Ok(Self { archive })
    }

    pub(crate) fn f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let array = self.array(name)?;
        if !matches!(array.descriptor.as_str(), "<f4" | "=f4" | "|f4") {
            return Err(invalid(format!(
                "NPZ array {name} has {}, expected little-endian f32",
                array.descriptor
            )));
        }
        array
            .bytes
            .chunks_exact(4)
            .map(|bytes| {
                Ok(f32::from_le_bytes(
                    bytes.try_into().expect("four-byte chunk"),
                ))
            })
            .collect()
    }

    pub(crate) fn i32(&mut self, name: &str) -> Result<Vec<i32>> {
        let array = self.array(name)?;
        if !matches!(array.descriptor.as_str(), "<i4" | "=i4" | "|i4") {
            return Err(invalid(format!(
                "NPZ array {name} has {}, expected little-endian i32",
                array.descriptor
            )));
        }
        array
            .bytes
            .chunks_exact(4)
            .map(|bytes| {
                Ok(i32::from_le_bytes(
                    bytes.try_into().expect("four-byte chunk"),
                ))
            })
            .collect()
    }

    pub(crate) fn strings(&mut self, name: &str) -> Result<Vec<String>> {
        let array = self.array(name)?;
        let width = array
            .descriptor
            .strip_prefix("|S")
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|width| *width != 0)
            .ok_or_else(|| invalid(format!("NPZ array {name} is not a byte-string array")))?;
        array
            .bytes
            .chunks_exact(width)
            .map(|bytes| {
                let end = bytes
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(bytes.len());
                std::str::from_utf8(&bytes[..end])
                    .map(str::to_owned)
                    .map_err(|error| {
                        invalid(format!("NPZ array {name} contains invalid UTF-8: {error}"))
                    })
            })
            .collect()
    }

    pub(crate) fn contains(&mut self, name: &str) -> bool {
        self.archive.by_name(&format!("{name}.npy")).is_ok()
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn shape(&mut self, name: &str) -> Result<Vec<usize>> {
        Ok(self.array(name)?.shape)
    }

    fn array(&mut self, name: &str) -> Result<NpyArray> {
        let entry_name = format!("{name}.npy");
        let mut entry = self
            .archive
            .by_name(&entry_name)
            .map_err(|_| invalid(format!("NPZ array {name} is missing")))?;
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|error| invalid(format!("unable to read NPZ array {name}: {error}")))?;
        parse_npy(&bytes, name)
    }
}

struct NpyArray {
    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    shape: Vec<usize>,
    descriptor: String,
    bytes: Vec<u8>,
}

fn parse_npy(bytes: &[u8], name: &str) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(invalid(format!(
            "NPZ array {name} has an invalid NPY header"
        )));
    }
    let major = bytes[6];
    let (header_offset, header_length): (usize, usize) = match major {
        1 => (10, usize::from(u16::from_le_bytes([bytes[8], bytes[9]]))),
        2 | 3 if bytes.len() >= 12 => (
            12,
            usize::try_from(u32::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11],
            ]))
            .map_err(|_| invalid("NPY header length exceeds usize"))?,
        ),
        _ => {
            return Err(invalid(format!(
                "NPZ array {name} uses unsupported NPY version"
            )));
        }
    };
    let data_offset = header_offset
        .checked_add(header_length)
        .ok_or_else(|| invalid("NPY header length overflow"))?;
    let header = std::str::from_utf8(
        bytes
            .get(header_offset..data_offset)
            .ok_or_else(|| invalid(format!("NPZ array {name} has a truncated header")))?,
    )
    .map_err(|error| invalid(format!("NPZ array {name} header is not UTF-8: {error}")))?;
    if !header.contains("'fortran_order': False") && !header.contains("\"fortran_order\": False") {
        return Err(invalid(format!("NPZ array {name} must be C-contiguous")));
    }
    let descriptor = dictionary_string(header, "descr")?;
    let shape = dictionary_shape(header)?;
    let item_size = descriptor_item_size(&descriptor)?;
    let elements = shape.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or_else(|| invalid("NPY element count overflow"))
    })?;
    let expected = elements
        .checked_mul(item_size)
        .ok_or_else(|| invalid("NPY byte size overflow"))?;
    let data = bytes
        .get(data_offset..)
        .ok_or_else(|| invalid(format!("NPZ array {name} data is missing")))?;
    if data.len() != expected {
        return Err(invalid(format!(
            "NPZ array {name} has {} bytes, expected {expected}",
            data.len()
        )));
    }
    Ok(NpyArray {
        shape,
        descriptor,
        bytes: data.to_vec(),
    })
}

fn dictionary_string(header: &str, key: &str) -> Result<String> {
    let key_offset = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))
        .ok_or_else(|| invalid(format!("NPY header is missing {key}")))?;
    let value = header[key_offset..]
        .split_once(':')
        .map(|(_, value)| value.trim_start())
        .ok_or_else(|| invalid(format!("NPY header has invalid {key}")))?;
    let quote = value
        .chars()
        .next()
        .filter(|value| matches!(value, '\'' | '"'))
        .ok_or_else(|| invalid(format!("NPY header {key} is not quoted")))?;
    let rest = &value[quote.len_utf8()..];
    let end = rest
        .find(quote)
        .ok_or_else(|| invalid(format!("NPY header {key} quote is incomplete")))?;
    Ok(rest[..end].to_owned())
}

fn dictionary_shape(header: &str) -> Result<Vec<usize>> {
    let key_offset = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| invalid("NPY header is missing shape"))?;
    let value = header[key_offset..]
        .split_once(':')
        .map(|(_, value)| value)
        .ok_or_else(|| invalid("NPY shape is invalid"))?;
    let start = value
        .find('(')
        .ok_or_else(|| invalid("NPY shape start is missing"))?;
    let end = value[start + 1..]
        .find(')')
        .map(|offset| start + 1 + offset)
        .ok_or_else(|| invalid("NPY shape end is missing"))?;
    let shape = value[start + 1..end]
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| invalid("NPY shape contains a non-integer dimension"))
        })
        .collect::<Result<Vec<_>>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(invalid("NPY shape must contain non-zero dimensions"));
    }
    Ok(shape)
}

fn descriptor_item_size(descriptor: &str) -> Result<usize> {
    let digits = descriptor
        .trim_start_matches(['<', '>', '=', '|'])
        .trim_start_matches(|character: char| character.is_ascii_alphabetic());
    descriptor
        .starts_with(['<', '=', '|'])
        .then(|| digits.parse::<usize>().ok())
        .flatten()
        .filter(|size| *size != 0)
        .ok_or_else(|| invalid(format!("unsupported NPY descriptor {descriptor}")))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v1_little_endian_float_array() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }\n";
        let padding = (64 - (10 + header.len()) % 64) % 64;
        let mut padded = header[..header.len() - 1].to_owned();
        padded.extend(std::iter::repeat_n(' ', padding));
        padded.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend(u16::try_from(padded.len()).unwrap().to_le_bytes());
        bytes.extend(padded.as_bytes());
        bytes.extend(1.5_f32.to_le_bytes());
        bytes.extend((-2.0_f32).to_le_bytes());
        let array = parse_npy(&bytes, "values").unwrap();
        assert_eq!(array.descriptor, "<f4");
        assert_eq!(array.bytes.len(), 8);
    }
}
