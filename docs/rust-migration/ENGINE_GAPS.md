# Engine gaps (C-shared) — migration record

이 문서는 2026-08-31 마이그레이션 판정의 근거다. 아래 상태는 당시 기록이며,
현재 릴리즈의 해결 여부를 보증하지 않는다. 재검증과 수리 근거는
[v0.1.0 gate ledger](../releases/v0.1.0.md)에 남긴다.

C `v0.6.5-dfm` 계보와 Rust 호스트가 **동일하게** 보이는 엔진 측 결함의
레지스트리다. 호스트 마이그레이션 게이트는 호스트-패리티 계약이므로,
여기 등재된 갭은 게이트 셀에서 `PASS*(engine-gap E-n)` annotation의
근거가 된다 — 단 **같은 명령의 C 대조 실행이 증거로 기록된 항목만**
(`C-control: reproduced`). C-control이 pending인 항목은 annotation
근거로 쓸 수 없다.

규율: 갭 수리는 엔진(C/CUDA) 작업이며 `AGENT.md`를 따른다.
**엔진 수리 커밋과 호스트 마이그레이션 커밋을 혼합하지 않는다.**
이 기록은 genesis와 함께 `ds4-dfm-rs`로 이관되었다. 과거 증거를 보존하고,
추가 수리는 검증된 커밋과 재현 결과로 연결한다.

---

## E-1 — Motif-3 partial-reuse HTTP `cached=0`

- **증상:** partial prefix reuse에서 엔진은 `partial fork cut=6393
  suffix=11` / `lcp=22904`를 찍지만 HTTP `timings.prefill_cached_tokens`
  가 0으로 보고된다. Motif family의 `last_done` stats 배선 갭.
- **C-control:** **reproduced** — C 서버도 HTTP `cached=0`
  (하니스 실수 아님; 2026-08-26 판독).
- **Canonical 경로:** DeepSeek partial 4-way는 정상
  (`PARTIAL_RESTORED`, cached=6656) — 갭은 Motif family stats에 한정.
- **증거:**
  `scratch/rust-host-live/task53/bank-partial-fourway-motif-20260826-114317/`
- **스코프:** 엔진 (C `ds4.c`/`ds4_server.c` Motif stats 경로).
- **상태:** Open.

## E-2 — K-EXAONE standalone boot-prewarm CUDA IMA

- **증상:** in-process aligned-artifacts standalone 부트에서 boot
  prewarm 중 `CUDA tensor read failed: an illegal memory access`
  (exaone prefill logits readback). 이후 CUDA 컨텍스트가 오염되어 모든
  alloc이 IMA로 실패 → 요청은 `lazy session graph alloc failed` 500.
  메모리 용량 문제 아님 (avail ~35 GiB, ctx=512에서도 재현).
- **C-control:** **reproduced** — ① C `ds4-server` 동일 플래그에서 같은
  IMA·같은 500 (2026-08-26 phase B); ② `test-exaone-batch`(순수 C)도
  같은 클래스 IMA로 실패했다. candidate는 "exaone prefill logits
  readback" (`g4-logs/exaone-batch.log`); exact `v0.6.5-dfm` sm_121a
  `test_exaone_batch`도 같은 routed-prefill 뒤 CUDA IMA를 logits
  readback에서 표면화했다. rust-host 무관.
- **Canonical 경로:** VMM owner + worker는 정상 — worker prewarm 2.0s
  PASS, serial 생성 200 (`openai_chat_serial=1`, TTFT 480 ms).
- **증거:**
  `scratch/rust-host-live/task53/exaone-serial-rightsize-20260826-154403/`
  (phase A rust / phase B C-control / phase C owner+worker),
  `v065-full-restamp-20260831-184200/g4-logs/exaone-batch.log`,
  `v065-full-restamp-20260831-184200/g4-logs/exaone-batch-v065-tag-sm121a.log`.
- **스코프:** 엔진 (EXAONE prefill + in-process artifacts 경로, CUDA).
- **상태:** Open.

## E-3 — Solar-Open2 standalone boot-prewarm / session-test IMA

- **증상:** standalone `-c 2048` 부트의 boot_prewarm에서 `cudaFreeAsync`
  IMA (`cuda/mmq/ds4_ggml_stubs.cu:167`); `test-solar-session`(순수 C
  테스트 바이너리)에서도 같은 클래스 IMA (`cudaMallocAsync`
  `ds4_ggml_stubs.cu:139`, `cudaStreamBeginCapture (dense)` 실패).
- **C-control:** **reproduced** — ① C `ds4-server` standalone Solar
  `-c 2048` 부트에서 동일 IMA (2026-08-26 §10 셰이크아웃 G3,
  `v065-full-restamp-20260831-184200/g3-logs/e3-c-control.server.log`);
  candidate session test도 같은 `cudaMallocAsync` IMA를 표면화했다.
- **Canonical 경로:** VMM owner + rust worker `-c 2048` PASS
  (chat `ok` stop, `continuous=1`, TTFT 254 ms).
- **증거:** `scratch/rust-host-live/family-serve-20260826-retry/solar/`
  (workspace-level), `gate-20260826/g3-logs/`, `gate-20260826/golden-logs/`.
- **스코프:** 엔진 (Solar in-process 경로, CUDA).
- **상태:** Open.

## E-4 — Motif-3 batch 테스트 디코드 divergence — Fixed

- **증상:** `test-motif3-batch`가 row 1에서 기대 토큰과 불일치
  (`got=2753,173,122,689 want=2753,173,203024,439`).
- **C-control:** **reproduced** — v0.6.3-dfm golden worktree의
  `test_motif3_batch`가 **바이트 동일한 divergence**로 실패
  (`gate-20260826/golden-logs/motif-batch.log`). rust-host 무관;
  기대 벡터가 현 GGUF/드라이버/커널 상태와 어긋나는 pre-existing.
- **증거:** `gate-20260826/g2-logs/motif-batch.log` (candidate),
  `gate-20260826/golden-logs/motif-batch.log` (golden).
- **수리:** `2bbdc1b`에서 multi-row IQ2 D2R을 exact aligned-MMQ fallback으로
  제한했다. scalar decode의 측정된 D2R tier는 유지한다.
- **재검증:** current `test-motif3-batch` widths 1/2/3/2/1 PASS
  (`v065-full-restamp-20260831-184200/g2-logs/motif-batch-fixed.log`).
- **상태:** Fixed (`2bbdc1b`).

## E-5 — Motif-3 residency smoke RSS 한계 초과 — Fixed

- **증상:** VMM owner 하에 `test-motif3-resident`가 source GGUF map RSS
  370,412 KiB로 한계(262,144 KiB)를 초과해 실패 (VMM import 직후에도
  350,312 KiB).
- **C-control:** **reproduced** — v0.6.3-dfm golden 바이너리도 **동일
  수치**로 실패 (`gate-20260826/golden-logs/motif-resident.log`).
  rust-host 무관; 드라이버/커널(610.43.02 / 6.17.0-1031)의 map fault
  동작 대비 스모크 한계가 낡은 pre-existing.
- **증거:** `gate-20260826/g2-logs/motif-resident-owner.log` (candidate),
  `gate-20260826/golden-logs/motif-resident.log` (golden).
- **수리:** `eb4ba77`에서 성공한 VMM manifest import 뒤 source mapping
  cache를 해제하고, 256K 물리 할당 측정은 eager graph로 고정했다.
- **재검증:** import 직후 source RSS 0 KiB, inference 뒤 21,848 KiB,
  256K allocation 9.687 GiB, cleanup remainder 0
  (`v065-full-restamp-20260831-184200/g2-logs/motif-resident-retry.log`).
- **상태:** Fixed (`eb4ba77`).

## E-6 — dots3 residency smoke chunk/ring logit parity

- **증상:** VMM owner 하에 `test-dots3-resident`가 chunk/ring 세션에서
  `logit parity mismatch`로 실패. 수치
  `first=63594/63594 batch_cos=0.98607464 batch_nrmse=0.201895
  cache_cos=1 cache_nrmse=0 decode=0 tok/s`. residency gate는
  `chunk=0` (나머지 metadata/forward/session/dsa/cache256/release=1).
- **C-control:** **reproduced** — exact `v0.6.5-dfm` sm_121a
  `test_dots3_resident --chunk-only`가 candidate와 **동일 수치·동일
  gate**로 실패. worker SHA-256은
  `3f264c70de24499d5cb24dc4b84dfaeb7032f45336d1570507a433d76769d4ad`.
- **Canonical 경로:** loader/tokenizer PASS. 서빙 스모크는 별도
  (`family-serve` dots3 standalone는 이전에 PASS).
- **증거:**
  `v065-full-restamp-20260831-184200/g5-logs/dots3-resident-owner.log`
  (candidate),
  `v065-full-restamp-20260831-184200/g5-logs/dots3-resident-v065-tag-sm121a-chunk.log`
  (exact tag).
- **스코프:** 엔진/픽스처 (dots3 chunk/ring logit 게이트).
- **상태:** Open.
