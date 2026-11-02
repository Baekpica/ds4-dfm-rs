//! CLI + option validation matching `ds4_dist_parse_cli_arg` / `dist_validate_options`.

use crate::activation::bits_valid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    None = 0,
    Coordinator = 1,
    Worker = 2,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::None => "none",
            Role::Coordinator => "coordinator",
            Role::Worker => "worker",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Layers {
    pub start: u32,
    pub end: u32,
    pub has_output: bool,
    pub set: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Options {
    pub role: Role,
    pub layers: Layers,
    pub listen_host: Option<String>,
    pub listen_port: i32,
    pub coordinator_host: Option<String>,
    pub coordinator_port: i32,
    pub prefill_chunk: u32,
    pub prefill_window: u32,
    pub activation_bits: u32,
    pub debug: bool,
}

impl Default for Role {
    fn default() -> Self {
        Role::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliResult {
    Error,
    NotMatched,
    Matched,
}

pub fn parse_role(s: &str) -> Option<Role> {
    match s {
        "none" => Some(Role::None),
        "coordinator" => Some(Role::Coordinator),
        "worker" => Some(Role::Worker),
        _ => None,
    }
}

pub fn parse_layers(s: &str) -> Result<Layers, String> {
    let Some(colon) = s.find(':') else {
        return Err("expected A:B or A:output".into());
    };
    if colon == 0 || colon + 1 == s.len() {
        return Err("expected A:B or A:output".into());
    }
    if s[colon + 1..].contains(':') {
        return Err("layer range has too many ':' separators".into());
    }
    let start =
        parse_u32_component(&s[..colon]).ok_or_else(|| format!("invalid start layer in {s}"))?;
    let end_s = &s[colon + 1..];
    if end_s == "output" {
        return Ok(Layers {
            start,
            end: u32::MAX,
            has_output: true,
            set: true,
        });
    }
    let end = parse_u32_component(end_s).ok_or_else(|| format!("invalid end layer in {s}"))?;
    if end < start {
        return Err(format!("layer range end precedes start in {s}"));
    }
    Ok(Layers {
        start,
        end,
        has_output: false,
        set: true,
    })
}

fn parse_u32_component(s: &str) -> Option<u32> {
    if s.is_empty() || s.len() >= 32 {
        return None;
    }
    if s.as_bytes().iter().any(|c| !c.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn parse_port(s: &str, arg: &str) -> Result<i32, String> {
    let v: i64 = s
        .parse()
        .map_err(|_| format!("invalid value for {arg}: {s}"))?;
    if v <= 0 || v > 65535 {
        return Err(format!("invalid value for {arg}: {s}"));
    }
    Ok(v as i32)
}

fn parse_positive_u32(s: &str, name: &str) -> Result<u32, String> {
    if s.is_empty() {
        return Err(format!("{name} requires a positive integer"));
    }
    let v: u64 = s
        .parse()
        .map_err(|_| format!("invalid value for {name}: {s}"))?;
    if v == 0 || v > u64::from(u32::MAX) {
        return Err(format!("invalid value for {name}: {s}"));
    }
    Ok(v as u32)
}

pub fn parse_cli_arg(
    arg: &str,
    rest: &mut impl Iterator<Item = String>,
    opt: &mut Options,
) -> Result<CliResult, String> {
    match arg {
        "--role" => {
            let role = rest
                .next()
                .ok_or_else(|| "--role requires an argument".to_string())?;
            let parsed = parse_role(&role).ok_or_else(|| {
                format!("invalid distributed role: {role} (valid roles: none, coordinator, worker)")
            })?;
            opt.role = parsed;
            Ok(CliResult::Matched)
        }
        "--layers" => {
            let layers = rest
                .next()
                .ok_or_else(|| "--layers requires an argument".to_string())?;
            match parse_layers(&layers) {
                Ok(l) => {
                    opt.layers = l;
                    Ok(CliResult::Matched)
                }
                Err(detail) => Err(format!("invalid --layers {layers}: {detail}")),
            }
        }
        "--listen" => {
            if opt.listen_host.is_some() || opt.listen_port != 0 {
                return Err("specify --listen only once".into());
            }
            let host = rest
                .next()
                .ok_or_else(|| "--listen requires an argument".to_string())?;
            let port = rest
                .next()
                .ok_or_else(|| "--listen requires an argument".to_string())?;
            opt.listen_port = parse_port(&port, "--listen")?;
            opt.listen_host = Some(host);
            Ok(CliResult::Matched)
        }
        "--coordinator" => {
            if opt.coordinator_host.is_some() || opt.coordinator_port != 0 {
                return Err("specify --coordinator only once".into());
            }
            let host = rest
                .next()
                .ok_or_else(|| "--coordinator requires an argument".to_string())?;
            let port = rest
                .next()
                .ok_or_else(|| "--coordinator requires an argument".to_string())?;
            opt.coordinator_port = parse_port(&port, "--coordinator")?;
            opt.coordinator_host = Some(host);
            Ok(CliResult::Matched)
        }
        "--dist-prefill-chunk" => {
            let value = rest
                .next()
                .ok_or_else(|| "--dist-prefill-chunk requires an argument".to_string())?;
            opt.prefill_chunk = parse_positive_u32(&value, "--dist-prefill-chunk")?;
            Ok(CliResult::Matched)
        }
        "--dist-prefill-window" => {
            let value = rest
                .next()
                .ok_or_else(|| "--dist-prefill-window requires an argument".to_string())?;
            opt.prefill_window = parse_positive_u32(&value, "--dist-prefill-window")?;
            if opt.prefill_window > 64 {
                return Err("--dist-prefill-window must be <= 64".into());
            }
            Ok(CliResult::Matched)
        }
        "--dist-activation-bits" => {
            let value = rest
                .next()
                .ok_or_else(|| "--dist-activation-bits requires an argument".to_string())?;
            let bits = parse_positive_u32(&value, "--dist-activation-bits")?;
            if !bits_valid(bits) {
                return Err("--dist-activation-bits must be 32, 16, or 8".into());
            }
            opt.activation_bits = bits;
            Ok(CliResult::Matched)
        }
        "--dist-replay-check" => Err(ERR_REPLAY_CHECK_DEPRECATED.into()),
        "--debug" => {
            opt.debug = true;
            Ok(CliResult::Matched)
        }
        _ => Ok(CliResult::NotMatched),
    }
}

pub fn parse_cli(args: &[String]) -> Result<(Options, Vec<String>), String> {
    let mut opt = Options::default();
    let mut unmatched = Vec::new();
    let mut rest = args.iter().cloned().collect::<Vec<_>>().into_iter();
    while let Some(arg) = rest.next() {
        match parse_cli_arg(&arg, &mut rest, &mut opt)? {
            CliResult::Matched => {}
            CliResult::NotMatched => unmatched.push(arg),
            CliResult::Error => unreachable!(),
        }
    }
    Ok((opt, unmatched))
}

pub fn validate_options(opt: &Options) -> Result<(), String> {
    if opt.role == Role::None {
        if opt.layers.set
            || opt.listen_host.is_some()
            || opt.listen_port != 0
            || opt.coordinator_host.is_some()
            || opt.coordinator_port != 0
            || opt.prefill_chunk != 0
            || opt.prefill_window != 0
            || opt.activation_bits != 0
        {
            return Err("distributed options require --role coordinator or --role worker".into());
        }
        return Ok(());
    }
    if !opt.layers.set {
        return Err(format!("--role {} requires --layers", opt.role.name()));
    }
    if opt.prefill_window > 64 {
        return Err("--dist-prefill-window must be <= 64".into());
    }
    if opt.activation_bits != 0 && !bits_valid(opt.activation_bits) {
        return Err("--dist-activation-bits must be 32, 16, or 8".into());
    }
    match opt.role {
        Role::Coordinator => {
            if opt.listen_host.is_none() || opt.listen_port <= 0 {
                return Err("--role coordinator requires --listen HOST PORT".into());
            }
            if opt.coordinator_host.is_some() || opt.coordinator_port != 0 {
                return Err("--role coordinator must not use --coordinator".into());
            }
            Ok(())
        }
        Role::Worker => {
            if opt.coordinator_host.is_none() || opt.coordinator_port <= 0 {
                return Err("--role worker requires --coordinator HOST PORT".into());
            }
            if opt.prefill_chunk != 0 {
                return Err("--dist-prefill-chunk requires --role coordinator".into());
            }
            if opt.prefill_window != 0 {
                return Err("--dist-prefill-window requires --role coordinator".into());
            }
            if opt.activation_bits != 0 {
                return Err("--dist-activation-bits requires --role coordinator".into());
            }
            Ok(())
        }
        Role::None => Ok(()),
    }
}

pub fn validate_layers_for_model(opt: &Options, n_layers: u32) -> Result<(), String> {
    if opt.role == Role::None || !opt.layers.set {
        return Ok(());
    }
    if n_layers == 0 {
        return Err("model reports no layers".into());
    }
    let last = n_layers - 1;
    if opt.layers.start > last {
        return Err(format!("layer range starts past final model layer {last}"));
    }
    if !opt.layers.has_output && opt.layers.end > last {
        return Err(format!("layer range ends past final model layer {last}"));
    }
    if opt.role == Role::Coordinator && opt.layers.start != 0 {
        return Err("coordinator layer range must start at layer 0".into());
    }
    Ok(())
}

pub fn resolved_layer_end(layers: &Layers, n_layers: u32) -> u32 {
    if layers.has_output {
        n_layers.saturating_sub(1)
    } else {
        layers.end
    }
}

pub fn prepare_engine_options(opt: &Options) -> Result<(), String> {
    validate_options(opt)
}

/// `--dist-replay-check` only ever ran in the C CLI's one-shot coordinator;
/// Rust hosts took it and did nothing, so they reject it now.
const ERR_REPLAY_CHECK_DEPRECATED: &str = "--dist-replay-check is deprecated: Rust hosts never \
     ran it (see docs/rust-migration/C_CONTINUATION.md)";

pub const USAGE: &str = "\
  --role ROLE\n\
      Distributed role: coordinator or worker.\n\
  --layers A:B\n\
      Inclusive distributed layer slice, e.g. 10:20 or 21:output.\n\
  --listen HOST PORT\n\
      Coordinator TCP listen address. Workers may later use it to force their data listener.\n\
  --coordinator HOST PORT\n\
      Coordinator TCP address for --role worker.\n\
  --dist-prefill-chunk N\n\
      Coordinator prefill pipeline chunk size. Default: session cap, normally 4096.\n\
      Non-default values are experimental and can change logits unless validated.\n\
  --dist-prefill-window N\n\
      Coordinator max end-to-end prefill chunks in flight. Default: workers+2, capped at 8.\n\
  --dist-activation-bits N\n\
      Coordinator hidden-state transport width: 32, 16, or 8. Default: 32.\n\
  --debug\n\
      Print coordinator route/debug logs. Workers keep their normal logs without this.\n";
