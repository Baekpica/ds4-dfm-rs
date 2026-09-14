use ds4_kv::{Options, Store};
use std::path::PathBuf;

#[derive(Debug)]
pub struct DiskKvArgs {
    dir: Option<PathBuf>,
    space_mb: u64,
    options: Options,
    reject_different_quant: bool,
}

impl Default for DiskKvArgs {
    fn default() -> Self {
        Self {
            dir: None,
            space_mb: 0,
            options: Options::default(),
            reject_different_quant: false,
        }
    }
}

fn require_value(option: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("ds4-server-rs: missing value for {option}"))
}

fn positive_i32(option: &str, value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("ds4-server-rs: invalid value for {option}: {value}"))
}

fn nonnegative_i32(option: &str, value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|value| *value >= 0)
        .ok_or_else(|| format!("ds4-server-rs: invalid value for {option}: {value}"))
}

impl DiskKvArgs {
    pub fn parse_arg(
        &mut self,
        option: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        let mut value = || require_value(option, args.next());
        match option {
            "--kv-disk-dir" => self.dir = Some(value()?.into()),
            "--kv-disk-space-mb" => self.set_space_mb(&value()?)?,
            "--kv-disk-space" => {
                self.space_mb = ds4_core::parse_disk_space(&value()?)?;
            }
            "--kv-cache-min-tokens" => self.set_min_tokens(&value()?)?,
            "--kv-cache-cold-max-tokens" => {
                self.options.cold_max_tokens = nonnegative_i32(option, &value()?)?
            }
            "--kv-cache-continued-interval-tokens" => {
                self.options.continued_interval_tokens = nonnegative_i32(option, &value()?)?
            }
            "--kv-cache-boundary-trim-tokens" => {
                self.options.boundary_trim_tokens = nonnegative_i32(option, &value()?)?
            }
            "--kv-cache-boundary-align-tokens" => {
                self.options.boundary_align_tokens = nonnegative_i32(option, &value()?)?
            }
            "--kv-cache-reject-different-quant" => self.reject_different_quant = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn set_space_mb(&mut self, value: &str) -> Result<(), String> {
        self.space_mb = positive_i32("--kv-disk-space-mb", value)? as u64;
        Ok(())
    }

    fn set_min_tokens(&mut self, value: &str) -> Result<(), String> {
        self.options.min_tokens = positive_i32("--kv-cache-min-tokens", value)?;
        Ok(())
    }

    pub fn dir(&self) -> Option<&std::path::Path> {
        self.dir.as_deref()
    }

    pub fn space_mb(&self) -> u64 {
        self.space_mb
    }

    pub fn min_tokens(&self) -> i32 {
        self.options.min_tokens
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.options.cold_max_tokens > 0
            && self.options.cold_max_tokens < self.options.min_tokens
        {
            return Err(
                "ds4-server-rs: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens"
                    .into(),
            );
        }
        Ok(())
    }

    pub fn open(&self) -> Option<Store> {
        let dir = self.dir.as_ref()?;
        match Store::open(
            dir,
            self.space_mb,
            self.reject_different_quant,
            self.options,
        ) {
            Ok(store) => {
                eprintln!(
                    "ds4-server-rs: KV disk cache {} (budget={} MiB, cross-quant={}, min={}, cold_max={}, continued={}, trim={}, align={}; ordinary serial cold/continued/evict/load)",
                    dir.display(),
                    store.budget_bytes / (1024 * 1024),
                    if self.reject_different_quant { "reject" } else { "accept" },
                    self.options.min_tokens,
                    self.options.cold_max_tokens,
                    self.options.continued_interval_tokens,
                    self.options.boundary_trim_tokens,
                    self.options.boundary_align_tokens,
                );
                Some(store)
            }
            Err(error) => {
                eprintln!(
                    "ds4-server-rs: failed to create KV cache directory {}: {error}",
                    dir.display()
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DiskKvArgs;
    use ds4_kv::Options;
    use std::fs;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ds4-server-kv-cli-{tag}-{}", std::process::id()))
    }

    #[test]
    fn defaults_disable_disk_cache_and_keep_tuning_inert() {
        let mut kv = DiskKvArgs::default();
        assert!(kv.dir.is_none());
        assert_eq!(kv.space_mb, 0);
        assert_eq!(kv.options.min_tokens, Options::default().min_tokens);
        assert!(!kv.reject_different_quant);

        kv.set_space_mb("64").unwrap();
        kv.set_min_tokens("1024").unwrap();
        kv.reject_different_quant = true;
        assert!(kv.open().is_none());
    }

    #[test]
    fn ordinary_serial_checkpoint_policy_flags_parse() {
        let mut kv = DiskKvArgs::default();
        let cases = [
            ("--kv-disk-dir", "cache"),
            ("--kv-disk-space-mb", "64"),
            ("--kv-cache-min-tokens", "1024"),
            ("--kv-cache-cold-max-tokens", "4096"),
            ("--kv-cache-continued-interval-tokens", "2048"),
            ("--kv-cache-boundary-trim-tokens", "16"),
            ("--kv-cache-boundary-align-tokens", "512"),
        ];
        for (option, value) in cases {
            let mut values = [value.to_string()].into_iter();
            assert!(kv.parse_arg(option, &mut values).unwrap());
        }
        let mut empty = std::iter::empty();
        assert!(kv
            .parse_arg("--kv-cache-reject-different-quant", &mut empty)
            .unwrap());
        assert_eq!(kv.dir.as_deref(), Some(std::path::Path::new("cache")));
        assert_eq!(kv.space_mb, 64);
        assert_eq!(kv.options.min_tokens, 1024);
        assert_eq!(kv.options.cold_max_tokens, 4096);
        assert_eq!(kv.options.continued_interval_tokens, 2048);
        assert_eq!(kv.options.boundary_trim_tokens, 16);
        assert_eq!(kv.options.boundary_align_tokens, 512);
        assert!(kv.reject_different_quant);
    }

    #[test]
    fn cold_policy_allows_zero_and_validates_against_minimum() {
        let mut kv = DiskKvArgs::default();
        for option in [
            "--kv-cache-cold-max-tokens",
            "--kv-cache-continued-interval-tokens",
            "--kv-cache-boundary-trim-tokens",
            "--kv-cache-boundary-align-tokens",
        ] {
            let mut zero = ["0".to_string()].into_iter();
            assert!(kv.parse_arg(option, &mut zero).unwrap());
        }
        assert_eq!(kv.options.cold_max_tokens, 0);
        assert_eq!(kv.options.continued_interval_tokens, 0);
        assert_eq!(kv.options.boundary_trim_tokens, 0);
        assert_eq!(kv.options.boundary_align_tokens, 0);
        kv.validate().unwrap();

        let mut enabled = DiskKvArgs::default();
        enabled.options.min_tokens = 1024;
        enabled.options.cold_max_tokens = 512;
        assert_eq!(
            enabled.validate().unwrap_err(),
            "ds4-server-rs: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens"
        );
    }

    #[test]
    fn positive_values_and_int_max_are_accepted() {
        let mut kv = DiskKvArgs::default();
        kv.set_space_mb("2147483647").unwrap();
        kv.set_min_tokens("2147483647").unwrap();
        for option in [
            "--kv-cache-cold-max-tokens",
            "--kv-cache-continued-interval-tokens",
            "--kv-cache-boundary-trim-tokens",
            "--kv-cache-boundary-align-tokens",
        ] {
            let mut max = ["2147483647".to_string()].into_iter();
            assert!(kv.parse_arg(option, &mut max).unwrap());
        }
        assert_eq!(kv.space_mb, i32::MAX as u64);
        assert_eq!(kv.options.min_tokens, i32::MAX);
        assert_eq!(kv.options.cold_max_tokens, i32::MAX);
        assert_eq!(kv.options.continued_interval_tokens, i32::MAX);
        assert_eq!(kv.options.boundary_trim_tokens, i32::MAX);
        assert_eq!(kv.options.boundary_align_tokens, i32::MAX);
    }

    #[test]
    fn invalid_numeric_values_match_the_c_contract() {
        for value in ["", "0", "-1", "junk", "2147483648"] {
            let mut kv = DiskKvArgs::default();
            assert_eq!(
                kv.set_space_mb(value).unwrap_err(),
                format!("ds4-server-rs: invalid value for --kv-disk-space-mb: {value}")
            );
            assert_eq!(
                kv.set_min_tokens(value).unwrap_err(),
                format!("ds4-server-rs: invalid value for --kv-cache-min-tokens: {value}")
            );
        }
        for option in [
            "--kv-cache-cold-max-tokens",
            "--kv-cache-continued-interval-tokens",
            "--kv-cache-boundary-trim-tokens",
            "--kv-cache-boundary-align-tokens",
        ] {
            for value in ["", "-1", "junk", "2147483648"] {
                let mut kv = DiskKvArgs::default();
                let mut values = [value.to_string()].into_iter();
                assert_eq!(
                    kv.parse_arg(option, &mut values).unwrap_err(),
                    format!("ds4-server-rs: invalid value for {option}: {value}")
                );
            }
        }
    }

    #[test]
    fn missing_value_is_reported() {
        let mut missing = DiskKvArgs::default();
        for option in [
            "--kv-disk-dir",
            "--kv-cache-cold-max-tokens",
            "--kv-cache-continued-interval-tokens",
            "--kv-cache-boundary-trim-tokens",
            "--kv-cache-boundary-align-tokens",
        ] {
            let mut empty = std::iter::empty();
            assert_eq!(
                missing.parse_arg(option, &mut empty).unwrap_err(),
                format!("ds4-server-rs: missing value for {option}")
            );
        }
    }

    #[test]
    fn omitted_budget_opens_with_the_c_default() {
        let dir = temp_path("default-budget");
        let _ = fs::remove_dir_all(&dir);
        let mut kv = DiskKvArgs::default();
        kv.dir = Some(dir.clone());

        let store = kv.open().unwrap();

        assert_eq!(store.budget_bytes, 4096 * 1024 * 1024);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_open_failure_is_returned_for_nonfatal_disable() {
        let parent = temp_path("open-failure");
        let _ = fs::remove_file(&parent);
        let _ = fs::remove_dir_all(&parent);
        fs::write(&parent, b"not a directory").unwrap();
        let mut kv = DiskKvArgs::default();
        kv.dir = Some(parent.join("child"));

        assert!(kv.open().is_none());

        let _ = fs::remove_file(parent);
    }

    #[test]
    fn disk_space_alias_accepts_gib() {
        let mut kv = DiskKvArgs::default();
        assert!(kv.dir().is_none());
        let min_tokens = kv.min_tokens();
        let mut values = ["32G".to_string()].into_iter();
        assert!(kv.parse_arg("--kv-disk-space", &mut values).unwrap());
        assert_eq!(kv.space_mb(), 32 * 1024);
        assert_eq!(kv.min_tokens(), min_tokens);
    }

    #[test]
    fn server_bin_parses_serving_flags() {
        let src = include_str!("bin/ds4-server-rs.rs");
        for flag in [
            "--prefix-reuse",
            "--mtp-mode",
            "--max-seqs",
            "--print-plan",
            "--check-config",
            "--prefill-chunk",
            "--prefill-chunk-live",
            "--mem-floor-gb",
            "--kv-disk-space",
        ] {
            assert!(src.contains(flag), "ds4-server-rs must parse {flag}");
        }
    }
}
