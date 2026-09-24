use crate::{
    error::{Error, Result},
    obj::Obj,
};
/// Compare before exclusion, triangulation, or vertex splitting.
pub fn validate(neutral: &Obj, expression: &Obj, label: &str) -> Result<()> {
    if neutral.positions.len() != expression.positions.len()
        || neutral.faces.len() != expression.faces.len()
    {
        return Err(Error::Input(format!(
            "{label}: vertex/face count mismatch with neutral"
        )));
    }
    for (i, (a, b)) in neutral.faces.iter().zip(&expression.faces).enumerate() {
        if a.vertices != b.vertices {
            return Err(Error::Input(format!(
                "{label}:{}: topology mismatch at face {}",
                b.line,
                i + 1
            )));
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_counts_are_not_sufficient() {
        let v = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        let a = crate::obj::parse(format!("{v}f 1 2 3").as_bytes(), "a").unwrap();
        let b = crate::obj::parse(format!("{v}f 1 3 2").as_bytes(), "b").unwrap();
        assert!(
            validate(&a, &b, "b")
                .unwrap_err()
                .to_string()
                .contains("face 1")
        );
        let c = crate::obj::parse(format!("{v}vt 0\nf 1/1 2/1 3/1").as_bytes(), "c").unwrap();
        validate(&a, &c, "c").unwrap();
    }
}
