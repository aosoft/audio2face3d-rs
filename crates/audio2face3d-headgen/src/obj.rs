//! A deliberately constrained OBJ reader. Original vertex identities are retained.
use crate::error::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

pub const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_TOTAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_LINE_BYTES: u64 = 1024 * 1024;
#[derive(Clone, Debug)]
pub struct Face {
    pub vertices: Vec<u32>,
    pub object: String,
    pub group: String,
    pub material: String,
    pub line: usize,
}
#[derive(Clone, Debug)]
pub struct Obj {
    pub positions: Vec<[f32; 3]>,
    pub faces: Vec<Face>,
    pub texture_coordinates: usize,
    pub supplied_normals: usize,
    pub material_libraries: Vec<String>,
    pub sha256: String,
    pub bytes: u64,
}
/// Canonicalize every explicitly named file before reading any shape.
/// No external MTL, texture, or directory discovery is performed.
pub fn resolve_inputs(
    config: &crate::config::Config,
    root: &Path,
) -> Result<Vec<(String, PathBuf)>> {
    config.validate()?;
    let root = root
        .canonicalize()
        .map_err(|e| Error::Output(format!("{}: {e}", root.display())))?;
    if !root.is_dir() {
        return Err(Error::Input("input-root must be a directory".into()));
    }
    let mut paths = Vec::new();
    let mut identities = BTreeSet::new();
    let mut total = 0u64;
    for name in std::iter::once(config.neutral.as_str()).chain(config.expression_files()) {
        crate::config::relative_path(name)?;
        let path = root
            .join(name)
            .canonicalize()
            .map_err(|e| Error::Output(format!("{name}: {e}")))?;
        if !path.starts_with(&root) || !path.is_file() {
            return Err(Error::Input(format!(
                "{name}: not a regular file inside input-root"
            )));
        }
        if !identities.insert(path.clone()) {
            return Err(Error::Input(format!("{name}: duplicate resolved input")));
        }
        let bytes = path
            .metadata()
            .map_err(|e| Error::Output(format!("{name}: {e}")))?
            .len();
        total = total
            .checked_add(bytes)
            .ok_or_else(|| Error::Output("input size overflow".into()))?;
        if bytes > MAX_FILE_BYTES || total > MAX_TOTAL_BYTES {
            return Err(Error::Output(format!("{name}: input byte limit exceeded")));
        }
        paths.push((name.to_owned(), path));
    }
    Ok(paths)
}
pub fn read(path: &Path) -> Result<Obj> {
    let file = File::open(path).map_err(|e| Error::Output(format!("{}: {e}", path.display())))?;
    parse(BufReader::new(file), &path.display().to_string())
}
pub fn parse(mut reader: impl BufRead, label: &str) -> Result<Obj> {
    let mut obj = Obj {
        positions: vec![],
        faces: vec![],
        texture_coordinates: 0,
        supplied_normals: 0,
        material_libraries: vec![],
        sha256: String::new(),
        bytes: 0,
    };
    let (mut object, mut group, mut material) = (String::new(), String::new(), String::new());
    let mut hash = Sha256::new();
    let mut line = Vec::new();
    let mut number = 0;
    let mut indices = 0usize;
    loop {
        line.clear();
        let count = reader
            .by_ref()
            .take(MAX_LINE_BYTES + 1)
            .read_until(b'\n', &mut line)
            .map_err(|e| Error::Output(format!("{label}: {e}")))?;
        if count == 0 {
            break;
        }
        number += 1;
        let fail = |s: &str| Error::Input(format!("{label}:{number}: {s}"));
        obj.bytes += count as u64;
        if count as u64 > MAX_LINE_BYTES || obj.bytes > MAX_FILE_BYTES {
            return Err(Error::Output(format!(
                "{label}:{number}: byte limit exceeded"
            )));
        }
        hash.update(&line);
        let text = std::str::from_utf8(&line).map_err(|_| fail("invalid UTF-8"))?;
        let fields = text
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect::<Vec<_>>();
        if fields.is_empty() {
            continue;
        }
        let args = &fields[1..];
        let values = || -> Result<Vec<f32>> {
            args.iter()
                .map(|v| {
                    v.parse::<f32>()
                        .ok()
                        .filter(|x| x.is_finite())
                        .ok_or_else(|| fail("invalid finite coordinate"))
                })
                .collect()
        };
        match fields[0] {
            "v" => {
                if args.len() != 3 {
                    return Err(fail("v requires exactly x y z"));
                }
                let v = values()?;
                obj.positions.push([v[0], v[1], v[2]]);
                if obj.positions.len() > 100_000 {
                    return Err(Error::Output(format!("{label}: vertex limit exceeded")));
                }
            }
            "vt" => {
                if !(1..=3).contains(&args.len()) {
                    return Err(fail("vt requires 1..3 coordinates"));
                }
                values()?;
                obj.texture_coordinates += 1;
            }
            "vn" => {
                if args.len() != 3 {
                    return Err(fail("vn requires 3 coordinates"));
                }
                values()?;
                obj.supplied_normals += 1;
            }
            "f" => {
                if !(3..=4).contains(&args.len()) {
                    return Err(fail("only triangle and quad faces are supported"));
                }
                let resolve = |s: &str, len: usize| -> Result<u32> {
                    let i = s.parse::<i64>().map_err(|_| fail("invalid index"))?;
                    let index = if i > 0 {
                        i - 1
                    } else {
                        (len as i64)
                            .checked_add(i)
                            .ok_or_else(|| fail("index overflow"))?
                    };
                    if i == 0 || index < 0 || index >= len as i64 {
                        return Err(fail("index outside its source array"));
                    }
                    Ok(index as u32)
                };
                let mut vertices = Vec::new();
                for corner in args {
                    let parts = corner.split('/').collect::<Vec<_>>();
                    if parts.len() > 3
                        || parts[0].is_empty()
                        || (parts.len() == 2 && parts[1].is_empty())
                        || (parts.len() == 3 && parts[2].is_empty())
                    {
                        return Err(fail("invalid face corner"));
                    }
                    vertices.push(resolve(parts[0], obj.positions.len())?);
                    if parts.len() > 1 && !parts[1].is_empty() {
                        resolve(parts[1], obj.texture_coordinates)?;
                    }
                    if parts.len() == 3 {
                        resolve(parts[2], obj.supplied_normals)?;
                    }
                }
                indices += (vertices.len() - 2) * 3;
                if indices > 600_000 {
                    return Err(Error::Output(format!(
                        "{label}: triangle index limit exceeded"
                    )));
                }
                obj.faces.push(Face {
                    vertices,
                    object: object.clone(),
                    group: group.clone(),
                    material: material.clone(),
                    line: number,
                });
            }
            "o" => object = args.join(" "),
            "g" => group = args.join(" "),
            "usemtl" => {
                if args.is_empty() {
                    return Err(fail("missing material name"));
                }
                material = args.join(" ");
            }
            "mtllib" => {
                if args.is_empty() {
                    return Err(fail("missing material library"));
                }
                obj.material_libraries.push(args.join(" "));
            }
            "s" => {
                if args.len() != 1
                    || !(args[0] == "off" || args[0] == "on" || args[0].parse::<u32>().is_ok())
                {
                    return Err(fail("invalid smoothing group"));
                }
            }
            _ => return Err(fail(&format!("unsupported OBJ instruction {}", fields[0]))),
        }
    }
    if obj.positions.is_empty() || obj.faces.is_empty() {
        return Err(Error::Input(format!("{label}: empty geometry")));
    }
    obj.sha256 = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    Ok(obj)
}
#[cfg(test)]
mod tests {
    use super::*;
    const V: &str = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 0\nvt 0\nvt 1 1\nvn 0 0 1\n";
    #[test]
    fn independent_indices_negative_indices_and_attributes() {
        for f in [
            "1 2 3",
            "1/1 2/2 3/1",
            "1//1 2//1 3//1",
            "-4/-2/-1 -3/-1/-1 -2/-2/-1",
        ] {
            let text = format!("{V}o 日本語\ng a b\nusemtl skin\nmtllib missing.mtl\ns 1\nf {f}\n")
                .replace('\n', "\r\n");
            let obj = parse(text.as_bytes(), "fixture").unwrap();
            assert_eq!(obj.positions.len(), 4);
            assert_eq!(obj.faces[0].vertices, [0, 1, 2]);
            assert_eq!(obj.faces[0].object, "日本語");
            assert_eq!(obj.faces[0].group, "a b");
        }
    }
    #[test]
    fn invalid_inputs_are_errors() {
        for text in [
            "".into(),
            format!("{V}f 0 2 3"),
            format!("{V}f 1/3 2/1 3/1"),
            format!("{V}f 1 2 5"),
            format!("{V}f 1 2 3 4 1"),
            format!("{V}l 1 2"),
            "v NaN 0 0".into(),
            "v 0 0 0 1".into(),
        ] {
            assert!(parse(text.as_bytes(), "fixture").is_err(), "{text}");
        }
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    #[test]
    fn resolves_unicode_paths_and_rejects_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let mut c =
            crate::config::Config::parse(include_str!("../presets/ict-facekit.toml")).unwrap();
        c.neutral = "基準.obj".into();
        for name in std::iter::once(c.neutral.as_str()).chain(c.expression_files()) {
            std::fs::write(dir.path().join(name), b"fixture").unwrap();
        }
        assert_eq!(resolve_inputs(&c, dir.path()).unwrap().len(), 54);
        c.neutral = "../outside.obj".into();
        assert!(resolve_inputs(&c, dir.path()).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn rejects_symlink_outside_root() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let c = crate::config::Config::parse(include_str!("../presets/ict-facekit.toml")).unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join(&c.neutral)).unwrap();
        assert!(resolve_inputs(&c, dir.path()).is_err());
    }
}
