//! Gemma-4 26B A4B VQ (`gemma4_vq`) is its own artifact contract, in the shape
//! every VQ family in this tree uses: a GGUF whose routed experts are GVQ1
//! matrices linked to shared GVBC codebook tensors, under a per-family
//! metadata namespace (docs/ds4-dfm-model-families.md "Design contract").
//!
//! The artifact this module is written from carries 12177 tensors - 11520 GVQ1
//! expert matrices (30 layers x 128 experts x gate, up, down), 60 GVBC books
//! (K8192, dim 8, E4M3FN) and 597 ordinary tensors (175 Q4_K, 30 Q8_0, 392
//! BF16). Routed experts are the only VQ part: the trunk checkpoint has no
//! residual stage and no MTP block. The family's MTP drafter is a separate
//! assistant checkpoint (`Gemma4AssistantForCausalLM`: four layers, hidden
//! 1024, `num_kv_shared_layers` 4 so it has no k/v projection of its own, and
//! a pre/post projection to the 2816 backbone width); it is not in this
//! artifact yet and will join as its own part, the way MiMo's MTP blocks do.

mod layout;

use crate::LayoutSpec;

const VECTOR: u8 = 8;
const WORDS: u32 = 8192;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const Q4_K: u32 = 12;
const BF16: u32 = 30;
const Q8_0: u32 = 8;

pub(super) fn layouts() -> Vec<LayoutSpec> {
    layout::layouts()
}
