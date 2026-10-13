//! DeepSeek V4.1 zchain: the ② anti-repair / ③ post-train sidecar session
//! tables, merged the way the engine merges them at load.
//!
//! Source of truth is the C engine at `/data/YoungAi` (commit 3946dbc):
//!
//! - `core_v41_amp.c:123-152` `v41_gr_load`: per layer, `acc` starts at 1 and
//!   every present `gr_Lnn.bin` multiplies into it (② then ③); a layer with at
//!   least one source gets one override table. Two directories set separately
//!   would silently keep only the second - the product must be formed on the
//!   host first.
//! - `core_v41_amp.c:108-121` `v41_rb_load`: router biases come from the ②
//!   directory only, and the device table is process-wide: every load starts
//!   by unloading all of it (`core_v41_state.c:76-78`).
//! - `core_v41_amp.c:227-270` `v41_amp_load`: the `base.fnv` gate first, then
//!   rb, then the gains, then the low-rank pair. β (`--zchain-scale`, default
//!   1.0, non-positive clamped to 1.0 by `core_v41_api.c:83`) scales the ② A
//!   matrix only - it is the anti-repair step knob, ③ is a different thing.
//! - `core_v41_amp.c:231-238`: with both directories carrying an amp file for
//!   the same layer the two are concatenated by rank, ② rows first:
//!   `y += x·B₂·A₂ + x·B₃·A₃ = x·[B₂|B₃]·[A₂;A₃]`. A rank sum above 8192 is
//!   refused (`core_v41_amp.c:242`).
//! - `core_v41_amp.c:258-263`: gains or biases without any amp file are a
//!   legal plugin shape; nothing at all in either directory is a refusal.
//! - `core_v41_state.c:76-83`: no directory at all is the naked base - no
//!   refusal, but the process-wide rb table is still cleared.

use std::path::{Path, PathBuf};

use crate::sidecar::{
    check_base_fnv, AmpSidecar, BaseFingerprint, GrSidecar, RbSidecar, SidecarError, AMP_MAX_RANK,
};

/// The model geometry the sidecar headers are checked against
/// (`v41_gr_accum_layer`'s header test, `core_v41_amp.c:46-48`).
#[derive(Debug, Clone, Copy)]
pub struct ZchainGeom {
    pub n_layer: u32,
    pub n_expert: u32,
    pub n_embd: u32,
}

/// One layer's merged gain factors: the product of every present source.
#[derive(Debug)]
pub struct GrMerge {
    pub from2: bool,
    pub from3: bool,
    /// `n_expert * n_embd` factors, the table the device override wants.
    pub factor: Vec<f32>,
}

/// One layer's merged low-rank correction, rows concatenated ② then ③.
#[derive(Debug)]
pub struct AmpMerge {
    /// Rows contributed by ②; the first `k2` rows of `a`/`b`.
    pub k2: u32,
    /// Rows contributed by ③.
    pub k3: u32,
    /// Storage type of each source (0 = that directory had no file).
    pub typ2: i32,
    pub typ3: i32,
    /// `[k2 + k3][n_embd]` row-major, ② rows first. β scaled ②'s rows only.
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl AmpMerge {
    pub fn k(&self) -> u32 {
        self.k2 + self.k3
    }
}

/// The merged session tables of one load. Load-time data: the caller hands
/// `gr`/`rb`/`amp` to the device and can drop the struct.
#[derive(Debug)]
pub struct V41Zchain {
    pub geom: ZchainGeom,
    /// The effective β after the non-positive clamp.
    pub beta: f32,
    /// The `base.fnv` verdict (`Absent` when there is no ③ directory).
    pub base: BaseFingerprint,
    pub gr: Vec<Option<GrMerge>>,
    pub rb: Vec<Option<Vec<f32>>>,
    pub amp: Vec<Option<AmpMerge>>,
}

impl V41Zchain {
    /// C `v41_state_plugins` + `v41_amp_load`: `Ok(None)` is the naked base
    /// (no directory given; the caller must still clear the process-wide rb
    /// table, `core_v41_state.c:78`). With at least one directory, the
    /// `base.fnv` gate runs before anything is read, a broken file refuses
    /// the whole load, and an empty plugin is refused at the end.
    pub fn load(
        zchain: Option<&Path>,
        posttrain: Option<&Path>,
        geom: ZchainGeom,
        beta: f32,
    ) -> Result<Option<Self>, SidecarError> {
        if zchain.is_none() && posttrain.is_none() {
            return Ok(None);
        }
        // The ③-alone note comes from the state attach, before the gate
        // (core_v41_state.c:79-83), so a refusal still warns.
        warn_posttrain_alone(zchain, posttrain);
        let beta = if beta > 0.0 { beta } else { 1.0 };
        let base = match posttrain {
            Some(pt) => check_base_fnv(pt, zchain, geom.n_layer)?,
            None => BaseFingerprint::Absent,
        };

        let nl = geom.n_layer as usize;
        let mut gr = Vec::with_capacity(nl);
        let mut rb = Vec::with_capacity(nl);
        let mut amp = Vec::with_capacity(nl);
        for il in 0..geom.n_layer {
            gr.push(merge_gr(zchain, posttrain, geom, il)?);
            rb.push(match zchain {
                Some(d) => RbSidecar::read(&d.join(format!("rb_L{il:02}.bin")), geom.n_expert)?
                    .map(|r| r.bias),
                None => None,
            });
            amp.push(merge_amp(zchain, posttrain, geom, il, beta)?);
        }

        let n_gr = gr.iter().filter(|g| g.is_some()).count();
        let n_rb = rb.iter().filter(|r| r.is_some()).count();
        let n_amp = amp.iter().filter(|a| a.is_some()).count();
        if n_amp == 0 && n_gr == 0 && n_rb == 0 {
            return Err(SidecarError::Empty);
        }
        Ok(Some(Self {
            geom,
            beta,
            base,
            gr,
            rb,
            amp,
        }))
    }

    pub fn n_gr_layers(&self) -> u32 {
        self.gr.iter().filter(|g| g.is_some()).count() as u32
    }

    pub fn n_rb_layers(&self) -> u32 {
        self.rb.iter().filter(|r| r.is_some()).count() as u32
    }

    pub fn n_amp_layers(&self) -> u32 {
        self.amp.iter().filter(|a| a.is_some()).count() as u32
    }

    /// The rank range over the amp layers, for the scratch tensor sizing
    /// (`st->ampT`, `core_v41_amp.c:265`).
    pub fn k_range(&self) -> Option<(u32, u32)> {
        let ks: Vec<u32> = self.amp.iter().flatten().map(|a| a.k()).collect();
        Some((*ks.iter().min()?, *ks.iter().max()?))
    }
}

/// The ③-alone note (`core_v41_state.c:79-83`): ③ was solved on the ①+②
/// state, so this combination is not the verdict state. Warned once per
/// process, exactly like the engine.
fn warn_posttrain_alone(zchain: Option<&Path>, posttrain: Option<&Path>) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    if zchain.is_none() && posttrain.is_some() {
        ONCE.call_once(|| {
            eprintln!(
                "ds4: post-train sidecar without the anti-repair directory: it was solved \
                 on that state, this combination is not the verdict state"
            );
        });
    }
}

/// C `v41_gr_load`'s inner loop (`core_v41_amp.c:131-152`): `acc = 1`, then
/// every present source multiplies in. `None` = neither directory has the
/// layer's file.
fn merge_gr(
    zchain: Option<&Path>,
    posttrain: Option<&Path>,
    geom: ZchainGeom,
    il: u32,
) -> Result<Option<GrMerge>, SidecarError> {
    let mut acc: Option<Vec<f32>> = None;
    let mut from2 = false;
    let mut from3 = false;
    let multiply = |acc: &mut Option<Vec<f32>>, factor: Vec<f32>| match acc {
        Some(a) => {
            for (x, f) in a.iter_mut().zip(factor) {
                *x *= f;
            }
        }
        None => *acc = Some(factor),
    };
    if let Some(d) = zchain {
        if let Some(src) = GrSidecar::read(
            &d.join(format!("gr_L{il:02}.bin")),
            geom.n_expert,
            geom.n_embd,
        )? {
            multiply(&mut acc, src.factor);
            from2 = true;
        }
    }
    if let Some(d) = posttrain {
        if let Some(src) = GrSidecar::read(
            &d.join(format!("gr_L{il:02}.bin")),
            geom.n_expert,
            geom.n_embd,
        )? {
            multiply(&mut acc, src.factor);
            from3 = true;
        }
    }
    Ok(acc.map(|factor| GrMerge {
        from2,
        from3,
        factor,
    }))
}

/// C `v41_amp_load`'s amp loop (`core_v41_amp.c:234-256`): ② with β on its A
/// matrix, then ③ with 1.0, rows concatenated ② first; the rank sum is capped
/// at `AMP_MAX_RANK`.
fn merge_amp(
    zchain: Option<&Path>,
    posttrain: Option<&Path>,
    geom: ZchainGeom,
    il: u32,
    beta: f32,
) -> Result<Option<AmpMerge>, SidecarError> {
    let read = |dir: Option<&Path>| -> Result<Option<AmpSidecar>, SidecarError> {
        match dir {
            Some(d) => AmpSidecar::read(&d.join(format!("amp_L{il:02}.bin")), geom.n_embd),
            None => Ok(None),
        }
    };
    // ② is read first: a broken ② file stops before ③ is touched
    // (`core_v41_amp.c:238`).
    let mut a2 = read(zchain)?;
    let a3 = read(posttrain)?;
    let (k2, k3) = (
        a2.as_ref().map_or(0, |s| s.k),
        a3.as_ref().map_or(0, |s| s.k),
    );
    let k = k2 + k3;
    if k == 0 {
        return Ok(None);
    }
    if k > AMP_MAX_RANK {
        return Err(SidecarError::RankOver { layer: il, k2, k3 });
    }
    if beta != 1.0 {
        if let Some(s) = a2.as_mut() {
            for v in s.a.iter_mut() {
                *v *= beta;
            }
        }
    }
    let (typ2, typ3) = (
        a2.as_ref().map_or(0, |s| s.typ),
        a3.as_ref().map_or(0, |s| s.typ),
    );
    let mut a = Vec::with_capacity(k as usize * geom.n_embd as usize);
    let mut b = Vec::with_capacity(k as usize * geom.n_embd as usize);
    for s in [a2, a3].into_iter().flatten() {
        a.extend_from_slice(&s.a);
        b.extend_from_slice(&s.b);
    }
    Ok(Some(AmpMerge {
        k2,
        k3,
        typ2,
        typ3,
        a,
        b,
    }))
}

/// Convenience for the CLI/server flag pair.
pub fn load_dirs(
    zchain: Option<&str>,
    posttrain: Option<&str>,
    geom: ZchainGeom,
    beta: f32,
) -> Result<Option<V41Zchain>, SidecarError> {
    let z = zchain.filter(|d| !d.is_empty()).map(PathBuf::from);
    let p = posttrain.filter(|d| !d.is_empty()).map(PathBuf::from);
    V41Zchain::load(z.as_deref(), p.as_deref(), geom, beta)
}
