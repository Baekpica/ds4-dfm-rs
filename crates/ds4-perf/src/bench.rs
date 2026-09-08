use crate::csv;
use std::io::BufRead;

pub const HEADER: &str =
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,kvcache_bytes";

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub ctx: u64,
    pub prefill_tokens: u64,
    pub prefill_tps: f64,
    pub gen_tokens: u64,
    pub gen_tps: f64,
    pub first_token_sec: f64,
    pub kvcache_bytes: u64,
}

impl Row {
    pub fn csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{}\n",
            self.ctx,
            self.prefill_tokens,
            self.prefill_tps,
            self.gen_tokens,
            self.gen_tps,
            self.first_token_sec,
            self.kvcache_bytes
        )
    }
}

pub fn parse(reader: impl BufRead) -> Result<Vec<Row>, String> {
    let mut header = Vec::new();
    let mut rows = Vec::new();
    csv::records(reader, |row| {
        if csv::column(&row, &["prefill_tps"]).is_some() {
            header = row;
            return Ok(());
        }
        if header.is_empty() || row.iter().all(|v| v.trim().is_empty()) {
            return Ok(());
        }
        let field = |name| {
            csv::value(&row, &header, &[name])
                .ok_or_else(|| format!("missing ds4-bench column: {name}"))
        };
        let integer = |name| {
            field(name)?
                .parse::<u64>()
                .map_err(|_| format!("invalid {name}"))
        };
        rows.push(Row {
            ctx: integer("ctx_tokens")?,
            prefill_tokens: integer("prefill_tokens")?,
            prefill_tps: csv::number(field("prefill_tps")?)?,
            gen_tokens: integer("gen_tokens")?,
            gen_tps: csv::number(field("gen_tps")?)?,
            first_token_sec: csv::number(field("first_token_sec")?)?,
            kvcache_bytes: integer("kvcache_bytes")?,
        });
        Ok(())
    })?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_named_columns() {
        let input = include_str!("../tests/fixtures/bench.csv");
        let rows = parse(input.as_bytes()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].prefill_tps, 1502.8);
        assert_eq!(rows[0].gen_tps, 24.3);
        assert_eq!(rows[0].first_token_sec, 0.0512);
        assert_eq!(rows[0].kvcache_bytes, 4294967296);
        assert_eq!(rows[0].ctx, 8192);
        assert!(parse(b"opaque output\n".as_slice()).unwrap().is_empty());
    }
}
