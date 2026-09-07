use std::ffi::OsString;
use std::path::PathBuf;

pub const USAGE: &str = "ds4-perf doctor [--bench PATH]
ds4-perf scout --out DIRECTORY -- COMMAND [ARGS...]

Runs a fresh baseline, then a fresh cuda+nvtx Nsight Systems capture.
Use one ds4-bench frontier per invocation; emit benchmark CSV on stdout.
NCU is suggested only. Existing output directories are never overwritten.
";

#[derive(Debug, PartialEq)]
pub enum Args {
    Help,
    Doctor {
        bench: String,
    },
    Scout {
        out: PathBuf,
        command: Vec<OsString>,
    },
}

pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Args, String> {
    let mut args = args.into_iter();
    match args.next().as_deref().and_then(|s| s.to_str()) {
        None | Some("--help" | "-h") => Ok(Args::Help),
        Some("doctor") => {
            let bench = match args.next() {
                None => "./ds4-bench".into(),
                Some(flag) if flag == "--bench" => args
                    .next()
                    .filter(|s| !s.is_empty())
                    .ok_or("doctor --bench requires a path")?
                    .into_string()
                    .map_err(|_| "doctor path must be UTF-8")?,
                _ => return Err("doctor accepts only --bench PATH".into()),
            };
            if args.next().is_some() {
                return Err("doctor accepts only --bench PATH".into());
            }
            Ok(Args::Doctor { bench })
        }
        Some("scout") => {
            if args.next().as_deref() != Some("--out".as_ref()) {
                return Err("scout requires --out DIRECTORY".into());
            }
            let out = args
                .next()
                .filter(|s| !s.is_empty() && s != "--")
                .ok_or("scout requires an output directory")?;
            if args.next().as_deref() != Some("--".as_ref()) {
                return Err("put the benchmark command after --".into());
            }
            let command: Vec<_> = args.collect();
            if command.first().is_none_or(|s| s.is_empty()) {
                return Err("missing benchmark command after --".into());
            }
            Ok(Args::Scout {
                out: out.into(),
                command,
            })
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_arguments() {
        let args = [
            "scout",
            "--out",
            "a b",
            "--",
            "./ds4-bench",
            "--out",
            "$(x)",
            "--",
        ];
        assert_eq!(
            parse(args.map(Into::into)).unwrap(),
            Args::Scout {
                out: "a b".into(),
                command: args[4..].iter().map(OsString::from).collect(),
            }
        );
        for args in [
            vec!["doctor", "extra"],
            vec!["scout", "--out", "x", "--"],
            vec!["scout"],
        ] {
            assert!(parse(args.into_iter().map(Into::into)).is_err());
        }
    }
}
