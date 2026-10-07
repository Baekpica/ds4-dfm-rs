# SSD expert streaming

`--ssd-streaming` keeps GLM-5.3 routed experts in the mmap-backed GGUF and
loads selected gate/up/down triplets into a bounded device cache. Rust owns
options and admission; the native driver owns file I/O, residency and eviction.
The option is shared by `ds4`, `ds4-bench` and `ds4-server`.
GLM tool workflows use an HTTP client; the built-in `ds4-agent` executor
requires DeepSeek DSML and omits these GLM-only options.
Streaming defaults to Off. Omit `--ssd-streaming` for the normal resident
weight path; add it to enable the bounded expert cache. Qualify performance
on both paths. The resident reference uses one VMM owner and a worker with
`--max-seqs 1`, with context and workspace sized by memory admission.

## Options

| Option | Meaning |
|---|---|
| `--ssd-streaming` | Enable the GLM CUDA expert cache |
| `--ssd-streaming-cache-experts N` | Global cache capacity in expert triplets |
| `--ssd-streaming-cache-experts 24GB` | Byte budget in GiB; 24 GiB is the default |
| `--ssd-streaming-cold` | Advise eviction of consumed expert file pages |

Capacity spans all routed layers, including the embedded predictor. It must
hold at least one row's top-8 experts. A byte budget rounds down to complete
slots. The largest supported gate and down units determine each slot's padded
strides; IQ2_XXS, IQ2_XS and Q2_K retain their GGUF quantized bytes.

Each routing batch pins existing hits before choosing LRU victims. The driver
drains earlier work before reusing a slot, reads misses with bounded staging,
and invalidates a partially uploaded slot on failure. Wide prefill batches
are subdivided when their worst-case route union exceeds capacity.

Enable streaming only for one full, single-shard GLM-5.3 CUDA artifact on
coherent pageable-memory hardware such as the DGX Spark/GB10. The native
mapping driver rejects devices that cannot preserve unpinned file mappings.
Distributed loading, weight-server IPC, full-weight warming and anonymous/full
model-copy modes conflict with this residency contract and are rejected.
Other families need their own validated native integration.

## Memory policy

Account for these components separately:

- **Weights:** the virtual GGUF mapping, mandatory embedding/dense/shared/
  attention/control tensors, and the fixed expert cache.
- **Active state:** per-bank KDA/conv state, compact DSA latent history and
  pooled index keys. Embedded MTP adds predictor history and shared trial
  journal storage.
- **Workspace:** projection/routing buffers and bounded host transfer staging.
- **Media and artifacts:** the optional Vision model and its execution storage.
- **Headroom:** admission reserve and remaining host/GPU capacity.

File page cache can retain expert reads. `--ssd-streaming-cold` is advisory;
the expert device cache limit does not impose an OS page-cache limit.
Streaming trades file I/O for resident weight memory; measure its effect on
both prefill and decode for the intended storage and routing workload.

Disk KV checkpoints persist inactive sessions. Active attention/KDA state
remains allocated while a request runs; expert streaming does not offload it.

## Verification boundary

Current development evidence covers mixed-quant CUDA arithmetic, equal outputs
in padded-cache versus mapped-weight fixtures, cache eviction/failure handling,
and balanced source memory accounting. Native v4 resident/SSD arms at ctx 2048
with 266-token input and 1/128-row prefill pass finite logits, greedy/sample
generation, two banks, MTP abort/keep1 and byte-exact snapshot/fork/disk restore;
cross-width arithmetic differs, so broader answer quality remains open.

Fresh V11 raw-owner/worker and SSD 24 GiB frontiers at natural 2048/8192 tokens
match all 154880 F32 logits and 64 greedy IDs byte exactly with MTP Off.
The final SSD Off max_seqs 1 server also passes 24 short API/tool/image requests
with On3, Vision and partial reuse. These gates close their numeric and
serving fixtures; broader quality and matched speed qualification remain open.

Native SSD 24 GiB gates at ctx 8192 complete exactly 6147 synthetic tokens
with MTP Off and On, retrieve three recorded codes and pass byte-exact FILE
restore and continuation. Shared token history, logits and tensor state also
match across those arms; the On gate uses ordinary greedy continuation.

The Rust serial session gate verifies FILE, snapshot and bounded-range restore
with MTP Off and On at ctx 2048. A short HTTP gate covers three APIs, tools,
streaming, two banks and red/blue image fixtures with MTP On/draft 3.
Two concurrent Chat requests also retrieve codes from 6147 cached tokens,
adding 10 input tokens each at ctx 8192. That gate uses a trusted native
checkpoint imported into the Rust store; it does not qualify cold HTTP
prefill or automatic cross-runtime import.

The SSD Off/raw-owner path completes 1048512 input tokens with exact
checkpoint/replay state and three correct retrieval codes. Fresh Rust HTTP
gates restore that trusted prefix with SSD Off/max_seqs 1 and SSD On/max_seqs 2.
All three questions retrieve correct codes; MTP On3 and ordinary top-k1 agree
on decoded answers/counts within each profile. A short actual red image passes
beside the SSD long text banks. These are cached continuations, without a cold
HTTP 1M prefill or 1M image prompt. Broader MTP/Vision quality and SSD speed
remain open; see the [artifact and workload record](glm53-uncensored.md).
A separate SSD Off, 8K two-bank HTTP edit reuses the periodic 4096-token
checkpoint across banks, preserves the source and matches fresh reuse Off
answers/counts. This is a bounded diagnostic; resident performance uses one bank.
Metadata and `--check-config` are preflight only.

Startup reports mandatory tensor bytes, cache capacity and staging. Engine
close reports cache hits, misses, file read bytes and cache bytes. The serving
plan distinguishes requested/effective settings from recorded qualification.

The interface follows the SSD options in
[upstream ds4](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4.c).
This driver's mixed-format slots and family state remain local contracts.
