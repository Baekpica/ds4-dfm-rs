# Qwen3.8 image-input implementation specification

Status: V1 image inference verified on 2026-08-28; V2 cache identity and reuse
verified on 2026-08-29. This is the original implementation contract and
its recorded evidence. For current scope, see the
[Qwen release guide](../README.md#qwen-release-scope),
[September 7 image benchmarks](qwen38-image-2026-09-07.md) and
[current disk-KV limits](ds4-dfm-model-families.md#disk-kv-contract-and-limits).

## Objective

Add real still-image inference for the Qwen3.8/Qwen4Exp path. Chat
Completions, Responses, and Anthropic Messages must normalize their public
image blocks to one request representation, run the embedded 27-layer vision
encoder and merger from the main GGUF, replace `<|image_pad|>` token
embeddings with the projected image features, and use the reference 3-axis
M-RoPE positions through prefill and later decode.

Text-only, tool-calling, partial-prefix reuse, disk-KV, MTP, and two-bank
serving must keep their current behavior.

## V1 boundaries

- Still images only. Video, audio, PDF, and image output are out of scope.
- Accept PNG and JPEG supplied as bounded base64 data URIs.
- Public shapes:
  - Chat Completions: `{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}`
  - Responses: `{"type":"input_image","image_url":"data:image/...;base64,..."}`
  - Anthropic: `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}`
- Reject `http://`, `https://`, `file://`, raw filesystem paths, SVG, GIF,
  WebP, and malformed base64 with a specific 400 response. Server-side URL
  fetching would add an SSRF boundary and is not required for local clients
  that can send standard data URIs.
- Images are legal only in user messages. A system/assistant/tool image is a
  validation error, matching the pinned Qwen chat template's constraints.
- V1 supports multiple image blocks in encounter order, bounded to four
  images, 10 MiB of base64-decoded payload per image, and 20 MiB total per
  request. Each allocation is bounded before image decompression.
- Qwen's pinned `Qwen3VLProcessor`/`Qwen2VLImageProcessorFast` geometry is
  authoritative: RGB, mean/std 0.5,
  patch 16, temporal patch 2, spatial merge 2, factor 32, 65,536 minimum
  pixels, and 16,777,216 maximum pixels. A still frame is duplicated for the
  temporal patch exactly as in the reference processor.
- V1 did not write image-derived state to disk-KV or a live prefix record.
  `<|image_pad|>` token IDs describe only geometry and cannot identify the
  pixels. That cold-only cache boundary is superseded by the V2 identity
  extension below; the public image API and geometry remain unchanged.
- MTP is disabled for the image-bearing prefill and re-enabled only after the
  target graph has reached the text decode frontier. This avoids teaching the
  embedded drafter a second multimodal input contract in the first version.

## Reference contract

The implementation is matched against the pinned
`Qwen4ExpForConditionalGeneration` source and processor configuration, not
inferred from tensor names alone:

- image token 248056, vision start 248053, vision end 248054;
- Conv3D patch embedding `[2,16,16]`, hidden 1152;
- interpolated learned 48x48 position embedding;
- 27 non-causal vision Transformer blocks, 16 heads, head size 72, FFN 4304;
- 2x2 spatial merger, 4608 hidden, 2560 output;
- projected feature count `T * H * W / 4`;
- language M-RoPE sections `[11,11,10]` with text/image token-type IDs;
- continuation positions include the reference rope delta rather than the
  raw token count.

The local llama.cpp `libmtmd` implementation is used for public schema,
bounded decode, media-marker, and embedding-insertion precedent. It is not
linked into ds4: this GGUF embeds a Qwen4Exp-specific vision stack and the
existing mtmd projector does not implement this exact graph.

## Internal data flow

1. API parsers append text and image blocks to a `chat_msg` in encounter
   order. Each image owns decoded bytes only until preprocessing finishes.
2. Qwen prompt rendering emits
   `<|vision_start|><|image_pad|><|vision_end|>` at each image block and
   records the rendered marker ordinal. Other model families reject images.
3. Tokenization expands each single image placeholder to the exact projected
   feature count and builds parallel token-type and three-axis position arrays.
4. The vision graph resizes/normalizes the image, runs patch embedding,
   position interpolation, 27 blocks, and merger on CUDA, producing F32
   `[image_tokens, 2560]` features.
5. `qwen4exp_graph_forward_chunk` gathers normal token embeddings, overwrites
   image-token rows from the precomputed feature tensor, and passes explicit
   M-RoPE positions into QSA. PLE continues to receive the original token IDs.
6. The live graph retains the rope delta through decode so chunked prefill and
   later scalar decode use the same position contract. Image graphs do not
   create partial checkpoints and cannot be fork sources. Cold reset, a text
   fork into the bank, and teardown clear the multimodal fields together.

## Minimal implementation shape

Reuse the existing GGUF tensor descriptors, plain BF16/F32 matmul helpers,
CUDA tensor allocator, request buffers, and Qwen graph lifecycle.

- `ds4_server.c`: public schema parsing, shared media ordering, prompt marker
  insertion, request limits, and HTTP validation errors.
- `ds4.h` / `ds4.c`: one narrow image-input session call, vision weight
  binding/validation, preprocessing metadata, embedding replacement, M-RoPE
  state, and bank/checkpoint rules.
- `ds4_gpu.h` / `ds4_cuda.cu`: only kernels not expressible by existing
  primitives: patch/resample packing, vision position/RoPE, attention softmax,
  merger packing, embedding-row replacement, and explicit M-RoPE.
- `vendor/stb_image.h`: reuse the exact pinned decoder used by local
  llama.cpp; no new image framework or network client.
- `ds4_server.c`'s existing embedded test runner: one focused three-surface
  parser/normalization, PNG geometry, prompt-order, and rejection check;
  existing Qwen batch and real API runs remain the text/KV/bank gates.

Do not introduce a generic multimodal framework, plugin interface, media
registry, or asynchronous fetcher. The first implementation has one model
family and one media kind.

## Verification evidence

- The embedded server test normalizes Chat Completions, Responses, and
  Anthropic image blocks to the same ordered message representation. It checks
  the pinned 1x1-to-256x256 smart-resize geometry and rejects a network URL.
- The CUDA QSA test compares interleaved THW M-RoPE and M-RoPE pooled keys
  against its F32 CPU reference. The observed maxima were 4.77e-7 / 1.09e-7
  relative RMS and 1.19e-6 / 1.52e-7 relative RMS, respectively.
- The CUDA build and `ds4_test` complete every embedded/unit check before the
  pre-existing environment-dependent long-context fixture gate. With the
  worker running, that final gate stops at the expected single-process lock.
- The real `MQ-Q5-SSD-PLE-BF16` artifact answered a 180x180 PNG through Chat
  Completions: 116 prompt tokens, 291.1 prefill tok/s, 751.5 ms TTFT, and 27.3
  decode tok/s. The response's reasoning correctly identified the visible
  black angled lightning-like mark.
- Barrier-synchronized Responses and Anthropic image requests completed as
  one `served=2 fallback=0` continuous batch. All three API surfaces returned
  their native response shape with zero request failures.
- An 8,243-token Chat request placed the 64 image-feature rows across the
  8,192-token prefill boundary and completed at 489.4 prefill tok/s, 17.232 s
  TTFT, and 22.5 decode tok/s.
- After both banks had served images, a 7,040-token text request and its repeat
  completed; the repeat exercised `partial fork src=0 dst=1 cut=7039`, proving
  that a formerly multimodal destination returns to the text lifecycle. A
  subsequent shared-text image request was cold-placed without a live/disk
  prefix match.
- The final guarded 262,144-context, two-bank worker reported zero request,
  continuous-batch, census, and governor failures. The 115 GiB cgroup limit
  and 109 GiB reclaim threshold remained active; the post-gate observation was
  101.12 GiB device-live with 10.89 GiB system memory available.

## V2 decoded-pixel cache identity

V2 keeps the model-visible prompt unchanged and inserts one fixed-width marker
after each image placeholder only in the server's internal replay key. The
marker contains a hash of decoded RGB pixels plus source geometry, the derived
grid, and projected-token count. Container-only PNG/JPEG byte differences that
decode to the same pixels share an identity; different pixels do not, even
when their image-pad tokens and geometry are identical.

Live and disk matching compare image-bearing records only with image-bearing
requests. If a byte-LCP ends inside an image marker, token reuse is capped at
that image's first token so identical image-pad IDs cannot cross a pixel
mismatch. M-RoPE positions are rebuilt from the current request after a state
copy or recurrent-checkpoint restore. The vision tower is skipped only when
every image span lies below the restored frontier.

The 2026-08-29 guarded DGX Spark gate established:

- same-image exact continuation: 90/110 prompt tokens reused;
- same geometry with different pixels: cold placement and vision recompute;
- cross-worker disk restore: 1,890/1,909 prompt tokens reused;
- same-image divergent partial reuse: a 20,087-token request restored the
  16,384-token recurrent checkpoint and computed 3,703 tokens;
- zero request, continuous-batch, memory-census, or governor failures after
  the gate under the external 109/115 GiB reclaim/hard guard.

The disk exact path was exercised across a worker restart. Disk partial lookup
and image/text identity segregation are covered by the focused server/KV-store
test; no cross-restart divergent-partial performance claim is made.

This establishes source-matched operator order, semantic real-image behavior,
chunk/bank/API integration, and bounded serving on the released GGUF. It does
not claim a cross-runtime hidden-state/logit numeric comparison: the exact
original BF16 checkpoint was not loaded alongside ds4 for this gate. Large
images use a correctness-first O(P^2) online attention kernel; replace it with
tiled FlashAttention only if large-image profiling makes that path relevant.

The public interface remains intentionally small: base64 PNG/JPEG only, with
no URL fetch. Large-image attention remains the next vision-specific kernel
target; V2 changes cache identity and state reuse, not that O(P^2) kernel.
