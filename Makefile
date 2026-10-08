CC ?= cc
UNAME_S := $(shell uname -s)
IQUEST_NATIVE_INCS := ds4_iquest_ref.h ds4_iquest_bind.inc ds4_iquest_graph.inc ds4_iquest_session.inc ds4_iquest_payload.inc ds4_iquest_batch.inc ds4_iquest_bank_payload.inc
NAIVE_NATIVE_INCS := ds4_naive_plan.h ds4_naive_bind.inc ds4_naive_draft.inc ds4_naive_graph.inc ds4_naive_session.inc ds4_naive_mtp.inc ds4_naive_payload.inc ds4_naive_batch.inc ds4_naive_bank_payload.inc

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
DS4_MOTIF3_MODEL ?=
DS4_MOTIF3_FIXTURES ?= ../motif-3-mixed-ds4/fixtures/official-final
DS4_EXAONE_MODEL ?=
DS4_DOTS3_MODEL ?=
DS4_QWEN4EXP_MODEL ?=
DS4_QWEN4EXP_ROOT ?=
DS4_QWEN4EXP_SOURCE ?=
DS4_QWEN_VISION_TOKENS ?=
DS4_QWEN_VISION_IMAGES ?= tests/fixtures/qwen-images/screen.png
DS4_GLM53_MODEL ?=
DS4_GLM53_VISION_MODEL ?=
CUDA_EXTRA_BINS :=

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_ple.o ds4_distributed.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_ple.o ds4_distributed.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
# Persisted CUDA build configuration: the cuda-spark / cuda-generic / cuda
# targets record their flags here and every invocation includes the record,
# so a stale CUDA object (e.g. after a sync touches ds4_cuda.cu) can only be
# recompiled with the configuration the rest of the tree was built with --
# never silently with the bare nvcc defaults (compute_75 PTX, JIT'd onto the
# device with older codepaths: slower, and a different numeric profile than
# the arch-native SASS the tree's other objects carry). Command-line
# variables still override; switch configurations by running a cuda-*
# target; survives make clean deliberately.
-include .ds4-cuda-config.mk
ifneq ($(strip $(CUDA_ARCH)),)
ifeq ($(strip $(CUDA_ARCH)),sm_121)
# GB10: the v0.5 mxf4 block-scale MMA (indexer rr-selector) needs the
# arch-SPECIFIC target.  -arch=sm_121a alone silently emits .target sm_121
# and ptxas rejects the MMA, so the gencode pair is mandatory; sm_121a
# SASS runs on every sm_121 device.  DS4_CUDA_HAVE_MXF4 gates the kernels
# AND the host engage path so non-121a builds stay coherent.
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
endif
NVCC_EXTRA_FLAGS ?=
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math -std=c++17 $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread $(NVCC_EXTRA_FLAGS)
# deepmem lite-2 (plumbing deleted in D3-3): DS4_CUDA_SPARK_HBM_CACHE is
# retired.  Startup weight promotion is compiled unconditionally and gated
# at runtime (integrated devices only; policy knob DS4_WEIGHT_RESIDENCY,
# legacy opt-out DS4_CUDA_NO_HBM_CACHE), so cuda-spark is purely an
# arch-selection alias for CUDA_ARCH=sm_121 and the 08-05 installer
# plan-overcommit class (forum 378855/65) cannot be built.
# Include path so cuda/mmq/*.cu can find its sibling vendored headers and
# the ds4_ggml_stubs shim. The redirected ggml.h / ggml-impl.h / ggml-cuda.h
# live alongside the vendored common.cuh.
MMQ_INCLUDES := -Icuda/mmq
# -lcuda is required for the in-process VMM weight arena (CUDA driver API).
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcuda
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o cuda/mmq/ds4_fattn.o
QWEN38_PLE_CUDA_OBJ := cuda/qwen38_ple.o
DS4_CUDA_SUPPORT_OBJS := ds4_ple.o ds4_distributed.o ds4_cuda.o $(MMQ_OBJS) $(QWEN38_PLE_CUDA_OBJ)
DS4_CUDA_CORE_OBJS := ds4.o $(DS4_CUDA_SUPPORT_OBJS)
CORE_OBJS = $(DS4_CUDA_CORE_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_ple.o ds4_distributed.o
METAL_LDLIBS := $(LDLIBS)
CUDA_EXTRA_BINS := ds4_weight_server
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression \
        test-cuda-tokentile-ldmatrix \
        proof-cuda-smoke proof-cuda-long proof-cuda-opp-c \
        proof-rust-cuda-opp-c proof-inkling-adopted print-version \
        test-motif3-loader test-motif3-reference test-motif3-tokenizer \
        test-motif3-cuda test-motif3-resident test-motif3-batch \
        test-dots3-loader test-dots3-tokenizer \
        test-dots3-resident \
        test-qwen4exp-loader test-qwen4exp-tokenizer \
        test-qwen4exp-ple test-qwen4exp-ple-reference \
        test-qwen4exp-ple-cuda test-qwen4exp-primitives \
        test-qwen4exp-hc-forward \
        test-qwen4exp-ple-compute test-qwen4exp-ple-forward \
        test-qwen4exp-moe test-qwen4exp-moe-forward test-qwen4exp-gdn \
        test-qwen4exp-gdn-forward test-qwen4exp-qsa \
        test-qwen4exp-qsa-forward test-qwen4exp-batch \
        test-qwen4exp-verify \
        test-qwen-vision-attention test-qwen-vision-model test-qwen-vision-host \
        test-qwen-vision-norm test-qwen-vision-rope \
        test-mmid-fast \
        test-mmq-parity test-qwen35-cuda test-model-family-kernels test-inkling-kernels test-inkling-moe \
        test-inkling-attn-prep test-inkling-attention test-inkling-norm test-inkling-linear test-inkling-batch test-inkling-q8-batch test-inkling-media \
        test-solar-loader test-solar-kda test-solar-kda-prefill \
        test-solar-kda-chunk \
        test-glm53-loader test-glm53-vision-loader test-glm53-image \
        test-glm53-vision test-glm53-dsa test-glm53-session \
        test-glm53-multimodal-session \
        test-solar-gates test-solar-kv test-solar-tokenizer \
        test-solar-forward test-solar-session \
        test-exaone-ref test-exaone-kernels test-exaone-batch \
        pq2-0-test test-qwen35-rows \
        rust-bridge ds4-rs ds4-bench-rs ds4-agent-rs ds4-server-rs test-kv-parity test-web-parity test-dist-parity test-route-parity test-server-parity test-catalog-parity test-tokenizer-parity test-agent-parity test-session-parity

ifeq ($(UNAME_S),Darwin)
all: ds4-c ds4-server-c ds4-bench-c ds4-eval ds4-agent-c ds4 ds4-server ds4-bench ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal C oracles (ds4-*-c) + Rust defaults + ./ds4-eval"
	@echo "  make cpu          Build CPU-only C oracles (ds4-*-c) + ./ds4-eval"
	@echo "  make test         Build and run tests"
	@echo "  make rust-bridge  Compile native/bridge/ds4_bridge.o (Rust FFI skeleton)"
	@echo "  make test-kv-parity  C↔Rust KVC 4-way matrix (Phase 4)"
	@echo "  make test-web-parity C↔Rust web encode/wire + mock CDP (Phase 5)"
	@echo "  make test-dist-parity C↔Rust DS4D codecs + blocking runtime (Phase 6)"
	@echo "  make test-route-parity C↔Rust route_decide reason table (Phase 7)"
	@echo "  make test-server-parity C↔Rust HTTP door + parsers + tools + live tool stream + corrective retry + continuation + memgov /metrics (Phase 7)"
	@echo "  make test-catalog-parity C↔Rust shape catalog + mmap GGUF identify + tensor inventory + bind plan + host bind lookup + host load apply + host validate + host vocab apply + host layout + MTP/DSpark sibling catalogs (Phase 8)"
	@echo "  make test-tokenizer-parity C↔Rust tokenizer encode/decode/stop (Phase 8)"
	@echo "  make test-agent-parity C↔Rust one-turn agent prompt/projector (agent shadow)"
	@echo "  make test-session-parity C↔Rust session ledger / DSV4 prefix (Phase 8)"
	@echo "  make ds4-c        Build C oracle ./ds4-c"
	@echo "  make ds4-rs       Deprecated alias for ./ds4 (Rust default)"
	@echo "  make clean        Remove build outputs"

ds4-c: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server-c: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench-c: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent-c: ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4-c ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server-c ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench-c ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent-c ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"

proof-cuda-smoke proof-cuda-long proof-cuda-opp-c proof-rust-cuda-opp-c proof-inkling-adopted:
	@echo "$@ requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10 (sm_121a arch alias)"
	@echo "  make ds4-bench-perf        Build ./ds4-bench-perf with Rust NVTX (after CUDA build)"
	@echo "  make ds4-perf              Build standalone profiling orchestration"
	@echo "  make ds4-perf-gpu          Build optional CUDA 13.3 calibration/CUPTI helper"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only C oracles (ds4-*-c) + ./ds4-eval"
	@echo "  make test                Build and run tests (reuses the last cuda-* configuration)"
	@echo "  make rust-bridge         Compile native/bridge/ds4_bridge.o (Rust FFI skeleton)"
	@echo "  make ds4 / ds4-server    Build Rust defaults (Cargo bin names stay *-rs)"
	@echo "  make ds4-c / ds4-server-c Build C oracles"
	@echo "  make ds4-rs              Deprecated alias that copies ./ds4"
	@echo "  make test-kv-parity      C↔Rust KVC 4-way matrix (Phase 4)"
	@echo "  make test-web-parity     C↔Rust web encode/wire + mock CDP (Phase 5)"
	@echo "  make test-dist-parity    C↔Rust DS4D codecs + blocking runtime (Phase 6)"
	@echo "  make test-route-parity   C↔Rust route_decide reason table (Phase 7)"
	@echo "  make test-server-parity  C↔Rust HTTP door + parsers + tools + live tool stream + corrective retry + continuation + memgov /metrics (Phase 7)"
	@echo "  make test-catalog-parity C↔Rust shape catalog + mmap GGUF identify + tensor inventory + bind plan + host bind lookup + host load apply + host validate + host vocab apply + host layout + MTP/DSpark sibling catalogs (Phase 8)"
	@echo "  make test-tokenizer-parity C↔Rust tokenizer encode/decode/stop (Phase 8)"
	@echo "  make test-agent-parity   C↔Rust one-turn agent prompt/projector (agent shadow)"
	@echo "  make test-session-parity C↔Rust session ledger / DSV4 prefix (Phase 8)"
	@echo "  make proof-rust-cuda-opp-c C→Rust OPP-C host parity (oracle ./ds4-c, candidate ./ds4)"
	@echo "  make ds4-server-rs       Deprecated alias that copies ./ds4-server"
	@echo "  make clean               Remove build outputs (keeps the recorded cuda configuration)"

# GB10 / DGX Spark is compute capability 12.1. Without an explicit -arch,
# nvcc 13.0 emits compute_75 PTX that the driver JITs onto sm_121 with
# Turing-era codepaths (no cp.async, no Blackwell MMA) — measurably slower
# MMQ prefill. The arch must reach the sub-make as CUDA_ARCH (not inside a
# pre-expanded NVCCFLAGS, where the parent's empty NVCC_ARCH_FLAGS would
# erase it), so spark defines travel via NVCC_EXTRA_FLAGS instead.
cuda-spark:
	@printf '%s\n' '# written by make cuda-spark (see the config include note in Makefile)' 'CUDA_ARCH := sm_121' 'NVCC_EXTRA_FLAGS :=' > .ds4-cuda-config.mk
	$(MAKE) -B ds4-c ds4-server-c ds4-bench-c ds4-eval ds4-agent-c ds4 ds4-server ds4-bench ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH=sm_121 NVCC_EXTRA_FLAGS=""

cuda-generic:
	@printf '%s\n' '# written by make cuda-generic (see the config include note in Makefile)' 'CUDA_ARCH := native' > .ds4-cuda-config.mk
	$(MAKE) ds4-c ds4-server-c ds4-bench-c ds4-eval ds4-agent-c ds4 ds4-server ds4-bench ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	@printf '%s\n' '# written by make cuda (see the config include note in Makefile)' 'CUDA_ARCH := $(strip $(CUDA_ARCH))' > .ds4-cuda-config.mk
	$(MAKE) ds4-c ds4-server-c ds4-bench-c ds4-eval ds4-agent-c ds4 ds4-server ds4-bench ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH="$(CUDA_ARCH)"

ds4-c: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server-c: ds4_server.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench-c: ds4_bench.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent-c: ds4_agent.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4-c ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server-c ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench-c ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent-c ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# Proof-harness scenarios. Each is a thin wrapper around tests/ds4_proof.py
# --scenario <name>. They expect DS4_PROOF_BASE (and, for MTP scenarios,
# DS4_PROOF_MTP) in the environment; ds4 must already be built. The harness
# materializes the (canonical x overlay) matrix, writes work_dir/expanded-plan.json,
# runs every cell, and reports per-cell selected-token-id MD5s with vs-canonical-
# counterpart parity contracts.
#   - smoke / long: capture-vs-eager PARITY (two paths in one build must match).
#   - opp-c: FP8 KV DRIFT gate. Single canonical, no parity contract; each cell
#     is checked against the configured architecture's committed golden so
#     lossy-FP8 numeric drift between builds is caught. Never replace a golden
#     with candidate output; validate it against the frozen C tag first.
#     Generate a newly approved architecture golden with:
#       tests/ds4_proof.py --scenario cuda-opp-c-full \
#         --write-expected <architecture-golden.json> [weight-server flags]
#   - rust opp-c: HOST PARITY gate. The C oracle (./ds4-c) writes an ephemeral
#     snapshot and the Rust binary checks it through the same stable runner
#     path. This complements, and never replaces, the committed native golden.
DS4_PROOF_REQUIRE_BASE = @if [ -z "$$DS4_PROOF_BASE" ]; then echo "$@: set DS4_PROOF_BASE to a base model gguf path" >&2; exit 2; fi
ifeq ($(strip $(CUDA_ARCH)),sm_121)
DS4_PROOF_OPPC_EXPECTED := tests/proof/expected/cuda-opp-c-full-sm121a-v0.6.5-dfm-4d40d97.json
else
DS4_PROOF_OPPC_EXPECTED := tests/proof/expected/cuda-opp-c-full.json
endif
DS4_PROOF_OPPC_RUNNER := /tmp/ds4_proof/proof-cuda-opp-c-bin

proof-cuda-smoke: ds4
	$(DS4_PROOF_REQUIRE_BASE)
	tests/ds4_proof.py --scenario cuda-capture-smoke --work-dir /tmp/ds4_proof/$@

proof-inkling-adopted: ds4
	$(DS4_PROOF_REQUIRE_BASE)
	tests/ds4_proof.py --scenario inkling-adopted-rollback --work-dir /tmp/ds4_proof/$@

proof-cuda-long: ds4
	$(DS4_PROOF_REQUIRE_BASE)
	tests/ds4_proof.py --scenario cuda-long-context-full --work-dir /tmp/ds4_proof/$@

proof-cuda-opp-c: ds4-c
	$(DS4_PROOF_REQUIRE_BASE)
	@set -eu; \
		mkdir -p /tmp/ds4_proof; \
		ln -sfn "$(CURDIR)/ds4-c" "$(DS4_PROOF_OPPC_RUNNER)"; \
		echo "proof_expected=$(DS4_PROOF_OPPC_EXPECTED)"; \
		tests/ds4_proof.py --bin "$(DS4_PROOF_OPPC_RUNNER)" \
			--scenario cuda-opp-c-full --work-dir /tmp/ds4_proof/$@ \
			--check-expected $(DS4_PROOF_OPPC_EXPECTED)

# Candidate ./ds4 (Rust) vs oracle ./ds4-c (C). Same inode or same hash is a
# false-green (Rust-vs-Rust) and must die.
proof-rust-cuda-opp-c: ds4 ds4-c
	$(DS4_PROOF_REQUIRE_BASE)
	@set -eu; \
		if [ "$(CURDIR)/ds4" -ef "$(CURDIR)/ds4-c" ] || \
		   [ "$$(sha256sum ds4 | awk '{print $$1}')" = "$$(sha256sum ds4-c | awk '{print $$1}')" ]; then \
			echo "proof guard: ./ds4 and ./ds4-c are the same binary" >&2; \
			exit 2; \
		fi; \
		mkdir -p /tmp/ds4_proof; \
		root=$$(mktemp -d /tmp/ds4_proof/$@.XXXXXX); \
		runner=$$root/bin; \
		expected=$$root/c-expected.json; \
		mkdir -p "$$root/c" "$$root/rust"; \
		ln -s "$(CURDIR)/ds4-c" "$$runner"; \
		echo "proof_artifacts=$$root"; \
		tests/ds4_proof.py --bin "$$runner" --scenario cuda-opp-c-full \
			--work-dir "$$root/c" --write-expected "$$expected"; \
		ln -sfn "$(CURDIR)/ds4" "$$runner"; \
		tests/ds4_proof.py --bin "$$runner" --scenario cuda-opp-c-full \
			--work-dir "$$root/rust" --check-expected "$$expected"
endif

ds4.o: ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS) ds4_mimo2_bind.inc ds4_mimo2_plan.h ds4_mimo2_graph.inc ds4_mimo2_batch.inc ds4_mimo2_session.inc ds4_mimo2_mtp.inc ds4_mimo2_media.inc ds4_mimo2_payload.inc ds4_mimo2_dflash.inc cuda/mimo2_dflash_host.h ds4_dots3_batch.inc ds4_dots3_mtp.inc ds4_step37_graph.inc ds4_step37_vision.inc ds4_ling3vl_graph.inc ds4_ling3vl_vision.inc ds4_ling3vl_rope.h ds4_ling3vl_batch.inc ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_gpu.h vendor/stb_image.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

# Rust FFI seam: wraps ds4.h so crates/ds4-sys never bindgens the engine header.
# Not linked into the C oracles. Phase 3 shadows link this object plus CORE_OBJS.
rust-bridge: native/bridge/ds4_bridge.o

native/bridge/ds4_bridge.o: native/bridge/ds4_bridge.c native/bridge/ds4_bridge.h native/bridge/ds4_host_load.h ds4.h ds4_distributed.h
	$(CC) $(CFLAGS) -I. -c -o $@ native/bridge/ds4_bridge.c

# Phase 9: Rust host is the default name. Cargo [[bin]] stays *-rs;
# Makefile copies onto ./ds4 ./ds4-server ./ds4-bench ./ds4-agent.
# C oracles are ./ds4-c ./ds4-server-c ./ds4-bench-c ./ds4-agent-c.
# ./ds4-eval stays C.
DS4_RS_ROOT := $(abspath .)
DS4_RS_LINK_OBJS := native/bridge/ds4_bridge.o $(CORE_OBJS)
DS4_RS_SOURCES := Cargo.toml Cargo.lock $(shell find crates -type f -print) \
	vendor/hf-chat-template/Cargo.toml $(shell find vendor/hf-chat-template/src -type f -print)
# Cargo does not fingerprint external link-object contents; include them in rustc metadata.
DS4_RS_LINK_FINGERPRINT = $(shell cksum $(DS4_RS_LINK_OBJS) 2>/dev/null | cksum | awk '{print $$1}')
# Cargo honors an externally supplied CARGO_TARGET_DIR. Copy the binary from
# that same directory so sandboxed/CI builds cannot silently publish a stale
# workspace-local target/release artifact.
DS4_RS_TARGET_DIR := $(if $(CARGO_TARGET_DIR),$(CARGO_TARGET_DIR),target)
ifeq ($(UNAME_S),Darwin)
DS4_RS_LIBS := -C link-arg=-framework -C link-arg=Foundation \
	-C link-arg=-framework -C link-arg=Metal -C link-arg=-lm
else
DS4_RS_GCCLIB := $(dir $(shell gcc -print-libgcc-file-name))
DS4_RS_LIBS := -C link-arg=-L$(CUDA_HOME)/targets/sbsa-linux/lib \
	-C link-arg=-L$(CUDA_HOME)/lib64 \
	-C link-arg=-L$(DS4_RS_GCCLIB) \
	-C link-arg=-lcudart -C link-arg=-lcublas -C link-arg=-lcuda \
	-C link-arg=-lstdc++ -C link-arg=-latomic -C link-arg=-lgcc \
	-C link-arg=-ldl -C link-arg=-lm -C link-arg=-lpthread -C link-arg=-lc
endif

ds4: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --bin ds4-rs --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-rs" $@

ds4-agent: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --bin ds4-agent-rs --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-agent-rs" $@

.PHONY: ds4-perf ds4-perf-gpu
ds4-perf:
	cargo build --release --locked -p ds4-perf
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-perf" $@

# Profiling-only CUDA bindings; inference packages do not depend on this helper.
ds4-perf-gpu:
	cargo build --release --locked -p ds4-perf-gpu --features cuda
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-perf-gpu" $@
	cp -f "$(DS4_RS_TARGET_DIR)/release/libds4_perf_gpu.so" libds4_perf_gpu.so

# NVTX is an optional Rust-host dependency; native linking stays identical.
# The official SDK build requires libclang for bindgen.
ds4-bench: DS4_RS_BENCH_FEATURES = native
ds4-bench-perf: DS4_RS_BENCH_FEATURES = native,perf-nvtx
.PHONY: ds4-bench ds4-bench-perf
ds4-bench ds4-bench-perf: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --bin ds4-bench-rs --release --features $(DS4_RS_BENCH_FEATURES) -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-bench-rs" $@

tests/naive_gate: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example naive_gate --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/naive_gate" $@

# Common residency probes use at most one MiB of actual model-file mapping.
tests/weight_mapping_probe: tests/test_weight_mapping.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcudart

tests/weight_mapping_policy: tests/test_weight_mapping_policy.cu $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -DDS4_USE_CUDA -o $@ $^ $(CUDA_LDLIBS)

tests/test_iquest_primitives: tests/test_iquest_primitives.cu cuda/iquest_primitives.cuh ds4_iquest_ref.h
	$(NVCC) $(NVCCFLAGS) --fmad=false -o $@ $<

tests/iquest_attention_profile: tests/iquest_attention_profile.cu cuda/iquest_primitives.cuh cuda/iquest_prefill.cuh cuda/iquest_decode.cuh ds4_iquest_ref.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/iquest_attn_precision: tests/iquest_attn_precision.cu cuda/iquest_prefill.cuh cuda/iquest_primitives.cuh ds4_iquest_ref.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/iquest_reduce_verify: tests/iquest_reduce_verify.cu cuda/iquest_primitives.cuh ds4_iquest_ref.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/iquest_router_profile: tests/iquest_router_profile.cu cuda/iquest_primitives.cuh cuda/iquest_router.cuh ds4_iquest_ref.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/iquest_router_verify: tests/iquest_router_verify.cu cuda/iquest_router.cuh cuda/iquest_primitives.cuh ds4_iquest_ref.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/test_iquest_dispatch: tests/test_iquest_dispatch.c ds4.c $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -o $@ $< -Wl,--gc-sections $(LDLIBS)

.PHONY: test-iquest-primitives test-iquest-dispatch
test-iquest-primitives: tests/test_iquest_primitives
	./tests/test_iquest_primitives

test-iquest-dispatch: tests/test_iquest_dispatch
	./tests/test_iquest_dispatch

tests/iquest_verify: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example iquest_verify --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/iquest_verify" $@

tests/iquest_bank_verify: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example iquest_bank_verify --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/iquest_bank_verify" $@

tests/iquest_perf_verify: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example iquest_perf_verify --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/iquest_perf_verify" $@

tests/test_iquest_prefix: tests/test_iquest_prefix.c ds4.c $(IQUEST_NATIVE_INCS) $(DS4_CUDA_SUPPORT_OBJS)
	$(CC) $(CFLAGS) -DDS4_USE_CUDA -ffunction-sections -fdata-sections -c -o tests/test_iquest_prefix.o tests/test_iquest_prefix.c
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_iquest_prefix.o $(DS4_CUDA_SUPPORT_OBJS) -Xlinker --gc-sections $(CUDA_LDLIBS)

tests/naive_serve_gate: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example naive_serve_gate --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/naive_serve_gate" $@

# Phase 4: C KVC oracle linked against ds4_kvstore.o (no CUDA engine).
tests/parity/kv_c_oracle: tests/parity/kv_c_oracle.c tests/parity/kv_c_stubs.c ds4_kvstore.o
	$(CC) $(CFLAGS) -I. -o $@ tests/parity/kv_c_oracle.c tests/parity/kv_c_stubs.c ds4_kvstore.o -lm

test-kv-parity: tests/parity/kv_c_oracle
	DS4_KV_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/kv_c_oracle cargo test -p ds4-kv

# Phase 5: C encode/wire oracle + Rust search/visit against a mock CDP.
tests/parity/web_c_oracle: tests/parity/web_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/web_c_oracle.c

test-web-parity: tests/parity/web_c_oracle
	DS4_WEB_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/web_c_oracle cargo test -p ds4-web

# Phase 6: explicit DS4D integer codecs vs C htonl records.
tests/parity/dist_c_oracle: tests/parity/dist_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/dist_c_oracle.c -lm

test-dist-parity: tests/parity/dist_c_oracle
	DS4_DIST_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/dist_c_oracle cargo test -p ds4-dist

tests/parity/route_c_oracle: tests/parity/route_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/route_c_oracle.c

tests/parity/server_c_oracle: tests/parity/server_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/server_c_oracle.c

tests/parity/parse_c_oracle: tests/parity/parse_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/parse_c_oracle.c

tests/parity/stream_c_oracle: tests/parity/stream_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/stream_c_oracle.c

tests/parity/tool_stream_c_oracle: tests/parity/tool_stream_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/tool_stream_c_oracle.c

tests/parity/dsml_c_oracle: tests/parity/dsml_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/dsml_c_oracle.c

tests/parity/retry_c_oracle: tests/parity/retry_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/retry_c_oracle.c

tests/parity/admit_c_oracle: tests/parity/admit_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/admit_c_oracle.c

tests/parity/render_c_oracle: tests/parity/render_c_oracle.c tests/parity/render_tools_inc.c
	$(CC) $(CFLAGS) -o $@ tests/parity/render_c_oracle.c

tests/parity/bridge_null_oracle: tests/parity/bridge_null_oracle.c tests/parity/bridge_null_stubs.c native/bridge/ds4_bridge.c native/bridge/ds4_bridge.h ds4_distributed.h
	$(CC) $(CFLAGS) -I. -o $@ tests/parity/bridge_null_oracle.c tests/parity/bridge_null_stubs.c native/bridge/ds4_bridge.c $(LDLIBS)

tests/parity/cont_c_oracle: tests/parity/cont_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/cont_c_oracle.c

tests/parity/memgov_c_oracle: tests/parity/memgov_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/memgov_c_oracle.c

test-route-parity: tests/parity/route_c_oracle tests/parity/server_c_oracle tests/parity/parse_c_oracle tests/parity/stream_c_oracle tests/parity/tool_stream_c_oracle tests/parity/dsml_c_oracle tests/parity/retry_c_oracle tests/parity/admit_c_oracle tests/parity/render_c_oracle tests/parity/bridge_null_oracle tests/parity/cont_c_oracle tests/parity/memgov_c_oracle tests/parity/kv_c_oracle
	DS4_ROUTE_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/route_c_oracle \
	DS4_SERVER_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/server_c_oracle \
	DS4_PARSE_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/parse_c_oracle \
	DS4_STREAM_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/stream_c_oracle \
	DS4_TOOL_STREAM_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/tool_stream_c_oracle \
	DS4_DSML_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/dsml_c_oracle \
	DS4_RETRY_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/retry_c_oracle \
	DS4_ADMIT_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/admit_c_oracle \
	DS4_RENDER_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/render_c_oracle \
	DS4_BRIDGE_NULL_ORACLE=$(DS4_RS_ROOT)/tests/parity/bridge_null_oracle \
	DS4_CONT_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/cont_c_oracle \
	DS4_MEMGOV_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/memgov_c_oracle \
	DS4_KV_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/kv_c_oracle \
		cargo test -p ds4-server

test-server-parity: test-route-parity
	python3 tests/test_prefill_cancel.py

tests/parity/shape_c_oracle: tests/parity/shape_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/shape_c_oracle.c

tests/parity/catalog_c_oracle: tests/parity/catalog_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/catalog_c_oracle.c

tests/parity/tensor_c_oracle: tests/parity/tensor_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/tensor_c_oracle.c

tests/parity/bind_c_oracle: tests/parity/bind_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/bind_c_oracle.c

tests/parity/bind_lookup_c_oracle: tests/parity/bind_lookup_c_oracle.c native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -I. -o $@ tests/parity/bind_lookup_c_oracle.c

tests/parity/load_c_oracle: tests/parity/load_c_oracle.c native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -I. -o $@ tests/parity/load_c_oracle.c

tests/parity/validate_c_oracle: tests/parity/validate_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/validate_c_oracle.c -lm

tests/parity/layout_c_oracle: tests/parity/layout_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/layout_c_oracle.c

tests/parity/vocab_c_oracle: tests/parity/vocab_c_oracle.c native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -I. -o $@ tests/parity/vocab_c_oracle.c

tests/parity/tokenizer_c_oracle: tests/parity/tokenizer_c_oracle.c ds4.c ds4.h native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

tests/parity/agent_c_oracle: tests/parity/agent_c_oracle.c ds4_agent.c
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-catalog-parity: tests/parity/shape_c_oracle tests/parity/catalog_c_oracle tests/parity/tensor_c_oracle tests/parity/bind_c_oracle tests/parity/bind_lookup_c_oracle tests/parity/load_c_oracle tests/parity/validate_c_oracle tests/parity/layout_c_oracle tests/parity/vocab_c_oracle tests/parity/tokenizer_c_oracle tests/parity/session_c_oracle tests/parity/payload_c_oracle
	DS4_SHAPE_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/shape_c_oracle \
	DS4_CATALOG_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/catalog_c_oracle \
	DS4_TENSOR_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/tensor_c_oracle \
	DS4_BIND_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/bind_c_oracle \
	DS4_BIND_LOOKUP_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/bind_lookup_c_oracle \
	DS4_LOAD_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/load_c_oracle \
	DS4_VALIDATE_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/validate_c_oracle \
	DS4_LAYOUT_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/layout_c_oracle \
	DS4_VOCAB_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/vocab_c_oracle \
	DS4_TOKENIZER_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/tokenizer_c_oracle \
	DS4_SESSION_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/session_c_oracle \
	DS4_PAYLOAD_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/payload_c_oracle \
		cargo test -p ds4-core

test-tokenizer-parity: tests/parity/tokenizer_c_oracle tests/parity/vocab_c_oracle
	DS4_TOKENIZER_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/tokenizer_c_oracle \
	DS4_VOCAB_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/vocab_c_oracle \
		cargo test -p ds4-core --test tokenizer

test-agent-parity: test-tokenizer-parity tests/parity/agent_c_oracle
	DS4_AGENT_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/agent_c_oracle \
		cargo test -p ds4-cli --lib agent::

tests/parity/session_c_oracle: tests/parity/session_c_oracle.c ds4.c ds4.h native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

tests/parity/payload_c_oracle: tests/parity/payload_c_oracle.c
	$(CC) $(CFLAGS) -o $@ tests/parity/payload_c_oracle.c

test-session-parity: tests/parity/session_c_oracle tests/parity/payload_c_oracle
	DS4_SESSION_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/session_c_oracle \
	DS4_PAYLOAD_C_ORACLE=$(DS4_RS_ROOT)/tests/parity/payload_c_oracle \
		cargo test -p ds4-core --test session --test payload

ds4-server: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-server --bin ds4-server-rs --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-server-rs" $@

# Transition aliases. Prefer the default names.
ds4-rs: ds4
	cp -f ds4 $@

ds4-agent-rs: ds4-agent
	cp -f ds4-agent $@

ds4-bench-rs: ds4-bench
	cp -f ds4-bench $@

ds4-server-rs: ds4-server
	cp -f ds4-server $@

ds4_ple.o: ds4_ple.c ds4_ple.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ple.c

ds4_cli.o: ds4_cli.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

# Version stamp: git describe in a checkout; the committed VERSION file
# (bumped at each release cut) covers gitless trees and tag-less clones.
# No --always in the describe tier: on an installer clone without tags it
# "succeeds" with a bare hash, VERSION is never read, and the update check
# treats the unparseable local as older -> daily self-nag on release builds.
DS4_BUILD_VERSION := $(shell git describe --tags --dirty 2>/dev/null || cat VERSION 2>/dev/null || echo unknown)

print-version:
	@echo $(DS4_BUILD_VERSION)

ds4_server.o: ds4_server.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h rax.h Makefile VERSION
	$(CC) $(CFLAGS) -DDS4_BUILD_VERSION='"$(DS4_BUILD_VERSION)"' -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

tests/cuda_tokentile_ldmatrix.o: tests/cuda_tokentile_ldmatrix.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ tests/cuda_tokentile_ldmatrix.c

tests/test_solar_kda.o: tests/test_solar_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_kda_prefill.o: tests/test_solar_kda_prefill.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_gates.o: tests/test_solar_gates.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_kv.o: tests/test_solar_kv.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_fattn.o: tests/test_solar_fattn.c ds4_gpu.h cuda/mmq/ds4_mmq.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_model_family_kernels.o: tests/test_model_family_kernels.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_inkling_kernels.o: tests/test_inkling_kernels.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_inkling_moe.o: tests/test_inkling_moe.c tests/fixtures/inkling/moe-vectors.h ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_inkling_attn_prep.o: tests/test_inkling_attn_prep.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -I. -c -o $@ $<

tests/test_inkling_attention.o: tests/test_inkling_attention.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -I. -c -o $@ $<

tests/test_inkling_norm.o: tests/test_inkling_norm.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -I. -c -o $@ $<

tests/test_inkling_linear.o: tests/test_inkling_linear.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -I. -c -o $@ $<

tests/test_inkling_q8_batch.o: tests/test_inkling_q8_batch.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -I. -c -o $@ $<

tests/test_inkling_media.o: tests/test_inkling_media.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -I. -c -o $@ $<

tests/test_inkling_forward.o: tests/test_inkling_forward.c ds4.c ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_inkling_session.o: tests/test_inkling_session.c ds4.c ds4.h ds4_gpu.h native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_inkling_encoders.o: tests/test_inkling_encoders.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_inkling_mtp.o: tests/test_inkling_mtp.c ds4.c ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_inkling_mtp_shared.o: tests/test_inkling_mtp_shared.c ds4.c ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4exp_primitives.o: tests/test_qwen4exp_primitives.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_hc_forward.o: tests/test_qwen4exp_hc_forward.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_ple_compute.o: tests/test_qwen4exp_ple_compute.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_ple_forward.o: tests/test_qwen4exp_ple_forward.c ds4.c ds4.h ds4_gpu.h ds4_ple.h cuda/qwen38_ple.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_moe.o: tests/test_qwen4exp_moe.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_moe_forward.o: tests/test_qwen4exp_moe_forward.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_gdn.o: tests/test_qwen4exp_gdn.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_gdn_forward.o: tests/test_qwen4exp_gdn_forward.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_qsa.o: tests/test_qwen4exp_qsa.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_qsa_forward.o: tests/test_qwen4exp_qsa_forward.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_forward.o: tests/test_solar_forward.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_session.o: tests/test_solar_session.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c $(IQUEST_NATIVE_INCS) ds4_naive_bind.inc ds4_naive_plan.h ds4_mimo2_bind.inc ds4_mimo2_plan.h ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h rax.h Makefile VERSION
	$(CC) $(CFLAGS) -DDS4_NO_GPU -DDS4_BUILD_VERSION='"$(DS4_BUILD_VERSION)"' -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h ds4_naive_stub.inc ds4_iquest_stub.inc $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_iquest_gpu.cuh ds4_iquest_ref.h cuda/iquest_primitives.cuh cuda/iquest_prefill.cuh cuda/iquest_decode.cuh cuda/iquest_router.cuh ds4_gpu.h ds4_qwen35_gpu.cuh cuda/qwen35_primitives.cuh cuda/qwen35_attn_gdn.cuh ds4_mimo2_gpu.cuh cuda/mimo2_primitives.cuh cuda/mimo2_prefill.cuh cuda/mimo2_media.cuh cuda/mimo2_dflash_attn.cuh cuda/mimo2_dflash_host.h ds4_glm53_vision_gpu.cuh ds4_inkling_gpu.cuh ds4_step37_gpu.cuh cuda/step37_primitives.cuh ds4_step37_vision_gpu.cuh cuda/step37_vision.cuh ds4_ling3vl_gpu.cuh cuda/ling3vl_primitives.cuh ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_repack.h cuda/mmq/ds4_mmq.h ds4_naive_gpu.cuh cuda/naive_primitives.cuh cuda/naive_sparse_tile.cuh cuda/naive_draft.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces. ds4_mmq.cu transitively pulls in mmq.cuh which has
# heavy template instantiation - compile in its own TU and link in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/ds4_mmq_pipe.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/mmvq.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/ds4_mimo2_swiglu.cuh cuda/mmq/ds4_glm_q2.h cuda/mmq/ds4_glm_shared.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/inkling_mmvq.cuh cuda/mmq/inkling_shared_tile.cuh cuda/mmq/inkling_q4.cuh cuda/mmq/inkling_q3.cuh cuda/mmq/common.cuh cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

# Shared aligned-artifact layout library: compiled into the engine (via
# MMQ_OBJS) and linked into ds4_weight_server so both producers build
# bit-identical repack artifacts.
cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

cuda/mmq/ds4_fattn.o: cuda/mmq/ds4_fattn.cu cuda/mmq/common.cuh cuda/mmq/mma.cuh cuda/mmq/ds4_mmq.h cuda/mmq/inkling_attention.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

tests/test_repack_premapped: tests/test_repack_premapped.cu cuda/mmq/ds4_repack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-repack-premapped: tests/test_repack_premapped
	./tests/test_repack_premapped

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/cuda_tokentile_ldmatrix: tests/cuda_tokentile_ldmatrix.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-tokentile-ldmatrix: tests/cuda_tokentile_ldmatrix
	./tests/cuda_tokentile_ldmatrix 0
	./tests/cuda_tokentile_ldmatrix 1
	./tests/cuda_tokentile_ldmatrix 2

tests/test_solar_kda: tests/test_solar_kda.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-kda: tests/test_solar_kda
	./tests/test_solar_kda

tests/test_solar_kda_prefill: tests/test_solar_kda_prefill.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-kda-prefill: tests/test_solar_kda_prefill
	./tests/test_solar_kda_prefill

tests/test_solar_kda_chunk.o: tests/test_solar_kda_chunk.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_kda_chunk: tests/test_solar_kda_chunk.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-kda-chunk: tests/test_solar_kda_chunk
	./tests/test_solar_kda_chunk

tests/test_solar_gates: tests/test_solar_gates.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-gates: tests/test_solar_gates
	./tests/test_solar_gates

tests/test_solar_kv: tests/test_solar_kv.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-kv: tests/test_solar_kv
	./tests/test_solar_kv

.PHONY: test-solar-fattn
tests/test_solar_fattn: tests/test_solar_fattn.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-fattn: tests/test_solar_fattn
	./tests/test_solar_fattn

cuda/mmq/test/test_mmq_parity.o: cuda/mmq/test/test_mmq_parity.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

tests/test_mmq_parity: cuda/mmq/test/test_mmq_parity.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-mmq-parity: tests/test_mmq_parity
	./tests/test_mmq_parity

# Prism Bonsai (qwen35) PQ2_0 and fold parity on CUDA.  The oracle is ds4.c's
# own reference code, reached through the DS4_TEST_HOOKS entry points, so the
# test links ds4.c built with that switch instead of the normal object.
ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_qwen35_cuda: tests/test_qwen35_cuda.cu ds4_cuda_test_hooks.o $(filter-out ds4.o,$(DS4_CUDA_CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -std=c++17 -DDS4_TEST_HOOKS -I. $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-qwen35-cuda
test-qwen35-cuda: tests/test_qwen35_cuda
	./tests/test_qwen35_cuda

# The Rust host (./ds4) is the default binary, and the one the server shares.
# It pins the shape and the tensor directory instead of parsing the GGUF, so its
# load-time configuration is not the C validator's: this gate pins the two
# pieces that do not come from the shape (the rotary table and the
# prism.hadamard fold) against the ids make bonsai-cuda-parity pins on the C
# host.  Both backends, because the fold feeds the reference too.
.PHONY: test-qwen35-rust-host
test-qwen35-rust-host: ds4 ds4-c
	@expect='11751 13 198 760 6511 314 9564 369'; \
	for backend in cuda cpu; do \
	  extra=""; [ $$backend = cuda ] && extra="DS4_CUDA_COPY_MODEL=1"; \
	  got=$$(env $$extra ./ds4 -m "$(DS4_BONSAI_MODEL)" --backend $$backend \
	         --token-ids 760,6511,314,9338,369 --predict 8 --temp 0 2>/dev/null | tail -1); \
	  case "$$got" in "$$expect"*) echo "qwen35 rust host parity ($$backend): PASS";; \
	    *) echo "qwen35 rust host parity ($$backend): FAIL"; \
	       echo "  expected: $$expect"; echo "  got:      $$got"; exit 1;; esac; \
	done

# The Bonsai session path, diffed against the in-process CPU reference on both
# backends.  The CUDA run needs the card (and DS4_CUDA_COPY_MODEL, see
# docs/BONSAI.md); the CPU run needs no card but pays the reference's ~3 s per
# forward, so it runs fewer steps and skips the long-prompt pass.
tests/test_qwen35_session.o: tests/test_qwen35_session.c ds4.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen35_session: tests/test_qwen35_session.o ds4_cuda_test_hooks.o $(filter-out ds4.o,$(DS4_CUDA_CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-qwen35-session test-qwen35-session-multichunk
test-qwen35-session: tests/test_qwen35_session
	DS4_CUDA_COPY_MODEL=1 DS4_TEST_MODEL="$(DS4_BONSAI_MODEL)" DS4_TEST_BACKEND=cuda DS4_TEST_STEPS="$${DS4_TEST_STEPS:-$(DS4_BONSAI_STEPS)}" ./tests/test_qwen35_session
	DS4_TEST_MODEL="$(DS4_BONSAI_MODEL)" DS4_TEST_BACKEND=cpu ./tests/test_qwen35_session

# The same scenarios with a two-token chunk, so the prefill crosses many chunk
# boundaries instead of handing the trunk one wide chunk.
test-qwen35-session-multichunk: tests/test_qwen35_session
	DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_PREFILL_CHUNK=2 DS4_TEST_MODEL="$(DS4_BONSAI_MODEL)" DS4_TEST_BACKEND=cuda DS4_TEST_STEPS="$${DS4_TEST_STEPS:-$(DS4_BONSAI_STEPS)}" ./tests/test_qwen35_session

tests/test_mmid_fast.o: tests/test_mmid_fast.cu cuda/mmq/mmid.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

tests/test_mmid_fast: tests/test_mmid_fast.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-mmid-fast: tests/test_mmid_fast
	./tests/test_mmid_fast

tests/test_model_family_kernels: tests/test_model_family_kernels.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-model-family-kernels: tests/test_model_family_kernels
	./tests/test_model_family_kernels

tests/test_step37_vision_ops: tests/test_step37_vision_ops.cu cuda/step37_vision.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-step37-vision-ops
test-step37-vision-ops: tests/test_step37_vision_ops
	./tests/test_step37_vision_ops

tests/naive_state: tests/naive_state.c ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

.PHONY: test-naive-state
test-naive-state: tests/naive_state
	./tests/naive_state

tests/naive_dense.o: tests/naive_dense.c ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/naive_dense: tests/naive_dense.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-naive-dense
test-naive-dense: tests/naive_dense
	./tests/naive_dense

tests/naive_graph.o: tests/naive_graph.c tests/naive_state_fixture.h ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/naive_graph: tests/naive_graph.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/naive_banks.o: tests/naive_banks.c tests/naive_state_fixture.h ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/naive_banks: tests/naive_banks.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/naive_trial.o: tests/naive_trial.c tests/naive_state_fixture.h ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/naive_trial: tests/naive_trial.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/naive_trial_live.o: tests/naive_trial_live.c ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/naive_trial_live: tests/naive_trial_live.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-naive-trial
test-naive-trial: tests/naive_trial
	./tests/naive_trial

.PHONY: test-naive-banks
test-naive-banks: tests/naive_banks
	./tests/naive_banks

.PHONY: test-naive-graph
test-naive-graph: tests/naive_graph
	./tests/naive_graph

tests/naive_bind: tests/naive_bind.c ds4.c ds4_naive_bind.inc ds4_naive_plan.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

.PHONY: test-naive-bind
test-naive-bind: tests/naive_bind
	NAIVE_NATIVE_BIND=$(DS4_RS_ROOT)/tests/naive_bind \
		cargo test -p ds4-core --test naive --locked native_bind_matches_directory

tests/naive_draft_bind: tests/naive_draft_bind.c ds4.c ds4_naive_bind.inc ds4_naive_plan.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

.PHONY: test-naive-draft-bind
test-naive-draft-bind: tests/naive_draft_bind
	NAIVE_DRAFT_BIND=$(DS4_RS_ROOT)/tests/naive_draft_bind \
		cargo test -p ds4-core --test naive_draft --locked draft_bind_matches_directory

tests/naive_draft_ops: tests/naive_draft_ops.cu cuda/naive_primitives.cuh cuda/naive_draft.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-naive-draft-ops
test-naive-draft-ops: tests/naive_draft_ops
	./tests/naive_draft_ops

tests/naive_draft_graph.o: tests/naive_draft_graph.c ds4.c $(NAIVE_NATIVE_INCS) $(IQUEST_NATIVE_INCS)
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/naive_draft_graph: tests/naive_draft_graph.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -Xlinker --gc-sections $(CUDA_LDLIBS)

tests/naive_memory: tests/naive_memory.c ds4_naive_plan.h
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: test-naive-memory
test-naive-memory: tests/naive_memory
	./tests/naive_memory

tests/naive_primitives: tests/naive_primitives.cu cuda/naive_primitives.cuh cuda/naive_sparse_tile.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-naive-primitives
test-naive-primitives: tests/naive_primitives
	./tests/naive_primitives

tests/naive_attention_profile: tests/naive_attention_profile.cu cuda/naive_primitives.cuh cuda/naive_sparse_tile.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/naive_softmax: tests/naive_softmax.cu cuda/naive_primitives.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/naive_index_profile: tests/naive_index_profile.cu cuda/naive_primitives.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/naive_index_u2_profile: tests/naive_index_u2_profile.cu cuda/naive_primitives.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-naive-index-u2
test-naive-index-u2: tests/naive_index_u2_profile
	@set -e; \
	for args in \
	    "2112 32 1 0 1" \
	    "8192 32 1 0 1" \
	    "2049 32 1 0 1" \
	    "2051 32 1 0 1" \
	    "2111 32 1 0 1" \
	    "2113 32 1 0 1" \
	    "8191 32 1 0 1" \
	    "8193 32 1 0 1" \
	    "2113 8 1 0 1" \
	    "8193 9 1 0 1" \
	    "8191 31 1 0 1" \
	    "2112 32 1 1 1" \
	    "8193 32 1 1 1" \
	    "2112 32 1 2 1" \
	    "8193 32 1 2 1" \
	    "2112 32 1 3 1" \
	    "8193 32 1 3 1" \
	    "2112 32 1 2 1 31" \
	    "2112 32 1 2 1 126" \
	    "2112 32 1 2 1 127" \
	    "2112 32 1 2 1 128" \
	    "2112 32 1 2 1 2047" \
	    "2112 1 1 2 1 0" \
	    "8192 1 2 0 1" \
	    "8193 7 2 0 1" \
	    "2113 8 2 0 1" \
	    "8191 31 2 0 1"; do \
	    ./tests/naive_index_u2_profile $$args; \
	done

tests/naive_router_profile: tests/naive_router_profile.cu cuda/naive_primitives.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

tests/naive_sum_profile: tests/naive_sum_profile.cu cuda/naive_primitives.cuh ds4_naive_plan.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-naive-sum
test-naive-sum: tests/naive_sum_profile
	@set -e; \
	for rows in 1 5 7 8 31 43 127 133 2048 2053 2181; do \
	    for fixture in 0 1 2; do \
	        ./tests/naive_sum_profile $$rows 1 $$fixture; \
	    done; \
	done

tests/naive_swiglu_profile: tests/naive_swiglu_profile.cu cuda/naive_primitives.cuh cuda/mmq/ds4_mimo2_swiglu.cuh ds4_naive_plan.h cuda/mmq/quantize.o cuda/mmq/ds4_ggml_stubs.o
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -o $@ $< cuda/mmq/quantize.o cuda/mmq/ds4_ggml_stubs.o $(CUDA_LDLIBS)

tests/test_step37_primitives: tests/test_step37_primitives.cu cuda/step37_primitives.cuh tests/fixtures/step37/primitives.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-step37-primitives
test-step37-primitives: tests/test_step37_primitives
	./tests/test_step37_primitives

tests/test_ling3vl_primitives: tests/test_ling3vl_primitives.cu cuda/ling3vl_primitives.cuh ds4_ling3vl_rope.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<

.PHONY: test-ling3vl-primitives
test-ling3vl-primitives: tests/test_ling3vl_primitives
	./tests/test_ling3vl_primitives

tests/test_ling3vl_matmul.o: tests/test_ling3vl_matmul.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_ling3vl_matmul.c

tests/test_ling3vl_matmul: tests/test_ling3vl_matmul.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-ling3vl-matmul
test-ling3vl-matmul: tests/test_ling3vl_matmul
	./tests/test_ling3vl_matmul

tests/test_ling3vl_q5pair.o: tests/test_ling3vl_q5pair.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_ling3vl_q5pair.c

tests/test_ling3vl_q5pair: tests/test_ling3vl_q5pair.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-ling3vl-q5pair
test-ling3vl-q5pair: tests/test_ling3vl_q5pair
	./tests/test_ling3vl_q5pair

tests/test_ling3vl_moefuse.o: tests/test_ling3vl_moefuse.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_ling3vl_moefuse.c

tests/test_ling3vl_moefuse: tests/test_ling3vl_moefuse.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-ling3vl-moefuse
test-ling3vl-moefuse: tests/test_ling3vl_moefuse
	./tests/test_ling3vl_moefuse

tests/test_ling3vl_mla: tests/test_ling3vl_mla.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-ling3vl-mla
test-ling3vl-mla: tests/test_ling3vl_mla
	./tests/test_ling3vl_mla

tests/test_ling3vl_mla_expand: tests/test_ling3vl_mla_expand.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-ling3vl-mla-expand
test-ling3vl-mla-expand: tests/test_ling3vl_mla_expand
	./tests/test_ling3vl_mla_expand

tests/test_inkling_kernels: tests/test_inkling_kernels.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-kernels: tests/test_inkling_kernels
	./tests/test_inkling_kernels

tests/test_inkling_moe: tests/test_inkling_moe.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-moe: tests/test_inkling_moe
	./tests/test_inkling_moe

tests/test_inkling_attn_prep: tests/test_inkling_attn_prep.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-attn-prep: tests/test_inkling_attn_prep
	./tests/test_inkling_attn_prep

tests/test_inkling_attention: tests/test_inkling_attention.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-attention: tests/test_inkling_attention
	./tests/test_inkling_attention

tests/test_inkling_norm: tests/test_inkling_norm.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-norm: tests/test_inkling_norm
	./tests/test_inkling_norm

tests/test_inkling_linear: tests/test_inkling_linear.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-linear: tests/test_inkling_linear
	./tests/test_inkling_linear

cuda/mmq/test/test_inkling_batch.o: cuda/mmq/test/test_inkling_batch.cu cuda/mmq/ds4_mmq.h cuda/mmq/mmvq.cuh ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

tests/test_inkling_batch: cuda/mmq/test/test_inkling_batch.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-batch: tests/test_inkling_batch
	./tests/test_inkling_batch

tests/test_inkling_q8_batch: tests/test_inkling_q8_batch.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-q8-batch: tests/test_inkling_q8_batch
	./tests/test_inkling_q8_batch

tests/test_inkling_media: tests/test_inkling_media.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-inkling-media: tests/test_inkling_media
	./tests/test_inkling_media

tests/test_inkling_forward: tests/test_inkling_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_inkling_session: tests/test_inkling_session.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_inkling_payload.o: tests/test_inkling_payload.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_inkling_payload: tests/test_inkling_payload.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_inkling_encoders: tests/test_inkling_encoders.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_inkling_mtp: tests/test_inkling_mtp.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_inkling_mtp_shared: tests/test_inkling_mtp_shared.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_qwen4exp_primitives: tests/test_qwen4exp_primitives.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_qwen_vision_attention.o: tests/test_qwen_vision_attention.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_qwen_vision_norm.o: tests/test_qwen_vision_norm.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_qwen_vision_rope.o: tests/test_qwen_vision_rope.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_qwen_vision_rope: tests/test_qwen_vision_rope.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen-vision-rope: tests/test_qwen_vision_rope
	./tests/test_qwen_vision_rope

tests/test_qwen_vision_norm: tests/test_qwen_vision_norm.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen-vision-norm: tests/test_qwen_vision_norm
	./tests/test_qwen_vision_norm

tests/test_qwen_vision_attention: tests/test_qwen_vision_attention.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen-vision-attention: tests/test_qwen_vision_attention
	./tests/test_qwen_vision_attention

# Full numeric gate. A resident owner may supply weights through the IPC env.
tests/test_qwen_vision_model.o: tests/test_qwen_vision_model.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen_vision_model: tests/test_qwen_vision_model.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen-vision-model: tests/test_qwen_vision_model
	@test -n "$(DS4_QWEN4EXP_MODEL)" || (echo "Set DS4_QWEN4EXP_MODEL to the Qwen GGUF"; exit 1)
	./tests/test_qwen_vision_model "$(DS4_QWEN4EXP_MODEL)" tests/fixtures/qwen-images/screen.png

# Rust-tokenized full-model gate; no retained C tokenizer is invoked.
tests/test_qwen_vision_host.o: tests/test_qwen_vision_host.c ds4.c ds4.h ds4_gpu.h native/bridge/ds4_host_load.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen_vision_host: tests/test_qwen_vision_host.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen-vision-host: tests/test_qwen_vision_host
	@test -n "$(DS4_QWEN4EXP_MODEL)" || (echo "Set DS4_QWEN4EXP_MODEL to the Qwen/Darwin GGUF"; exit 1)
	@test -n "$(DS4_QWEN_VISION_TOKENS)" || (echo "Set DS4_QWEN_VISION_TOKENS to Rust --dump-tokens output"; exit 1)
	./tests/test_qwen_vision_host "$(DS4_QWEN4EXP_MODEL)" "$(DS4_QWEN_VISION_TOKENS)" $(DS4_QWEN_VISION_IMAGES)

test-qwen4exp-primitives: tests/test_qwen4exp_primitives
	./tests/test_qwen4exp_primitives

tests/test_qwen4exp_hc_forward: tests/test_qwen4exp_hc_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-hc-forward: tests/test_qwen4exp_hc_forward
	./tests/test_qwen4exp_hc_forward

tests/test_qwen4exp_ple_compute: tests/test_qwen4exp_ple_compute.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-ple-compute: tests/test_qwen4exp_ple_compute
	./tests/test_qwen4exp_ple_compute

tests/test_qwen4exp_ple_forward: tests/test_qwen4exp_ple_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-ple-forward: tests/test_qwen4exp_ple_forward
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the first SSD-PLE GGUF shard" >&2; exit 2; }
	@test -n "$(DS4_QWEN4EXP_ROOT)" || \
		{ echo "set DS4_QWEN4EXP_ROOT to the SSD-PLE artifact root" >&2; exit 2; }
	@test -n "$(DS4_QWEN4EXP_BF16_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_BF16_MODEL to the first resident BF16 GGUF shard" >&2; exit 2; }
	./tests/test_qwen4exp_ple_forward "$(DS4_QWEN4EXP_MODEL)" \
		"$(DS4_QWEN4EXP_ROOT)" "$(DS4_QWEN4EXP_BF16_MODEL)"

tests/test_qwen4exp_moe: tests/test_qwen4exp_moe.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-moe: tests/test_qwen4exp_moe
	./tests/test_qwen4exp_moe

tests/test_qwen4exp_moe_forward: tests/test_qwen4exp_moe_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_qwen4exp_verify.o: tests/test_qwen4exp_verify.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_draft.o: tests/test_qwen4exp_draft.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_draft: tests/test_qwen4exp_draft.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-qwen4exp-draft
test-qwen4exp-draft: tests/test_qwen4exp_draft
	./tests/test_qwen4exp_draft
	DS4_QWEN_MTP_CPU_ARGMAX=1 ./tests/test_qwen4exp_draft

tests/test_qwen4exp_verify: tests/test_qwen4exp_verify.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-verify: tests/test_qwen4exp_verify
	./tests/test_qwen4exp_verify "$(DS4_QWEN4EXP_MODEL)"

test-qwen4exp-moe-forward: tests/test_qwen4exp_moe_forward
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the first SSD-PLE GGUF shard" >&2; exit 2; }
	./tests/test_qwen4exp_moe_forward "$(DS4_QWEN4EXP_MODEL)"

tests/test_qwen4exp_gdn: tests/test_qwen4exp_gdn.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-gdn: tests/test_qwen4exp_gdn
	./tests/test_qwen4exp_gdn

tests/test_qwen4exp_gdn_forward: tests/test_qwen4exp_gdn_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-gdn-forward: tests/test_qwen4exp_gdn_forward
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the first SSD-PLE GGUF shard" >&2; exit 2; }
	./tests/test_qwen4exp_gdn_forward "$(DS4_QWEN4EXP_MODEL)"

tests/test_qwen4exp_qsa: tests/test_qwen4exp_qsa.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-qsa: tests/test_qwen4exp_qsa
	./tests/test_qwen4exp_qsa

tests/test_qwen4exp_qsa_forward: tests/test_qwen4exp_qsa_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-qsa-forward: tests/test_qwen4exp_qsa_forward
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the first SSD-PLE GGUF shard" >&2; exit 2; }
	./tests/test_qwen4exp_qsa_forward "$(DS4_QWEN4EXP_MODEL)"

tests/test_qwen4exp_batch.o: tests/test_qwen4exp_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_qwen4exp_batch: tests/test_qwen4exp_batch.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-batch: tests/test_qwen4exp_batch
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the first SSD-PLE GGUF shard" >&2; exit 2; }
	./tests/test_qwen4exp_batch "$(DS4_QWEN4EXP_MODEL)"

# The test includes ds4.c directly for its static graph seams, so it must
# not also link ds4.o (duplicate externs); ds4_cuda.o resolves against the
# test object's own copy.
tests/test_solar_forward: tests/test_solar_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-forward: tests/test_solar_forward
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_forward "$(DS4_SOLAR_MODEL)" 128 29497 132 4767

tests/test_solar_session: tests/test_solar_session.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-solar-session: tests/test_solar_session
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_session "$(DS4_SOLAR_MODEL)"

.PHONY: test-solar-session-partial test-solar-disk-kv
test-solar-session-partial: tests/test_solar_session
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_session "$(DS4_SOLAR_MODEL)" --partial-only

test-solar-disk-kv: tests/test_solar_session
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_session "$(DS4_SOLAR_MODEL)" --disk-kv-only

ds4_weight_server: tools/ds4_weight_server.cu cuda/mmq/ds4_repack.o
	$(NVCC) $(NVCCFLAGS) -o $@ tools/ds4_weight_server.cu cuda/mmq/ds4_repack.o $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

ifneq ($(UNAME_S),Darwin)
# EXAONE harnesses include ds4.c to reach the reference and graph builders, so
# link the external distributed/CUDA implementation without a second ds4.o.
tests/test_exaone_ref.o: tests/test_exaone_ref.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_exaone_ref: tests/test_exaone_ref.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-exaone-ref: tests/test_exaone_ref
	@test -n "$(DS4_EXAONE_MODEL)" || \
		{ echo "set DS4_EXAONE_MODEL to the EXAONE GGUF" >&2; exit 2; }
	./tests/test_exaone_ref "$(DS4_EXAONE_MODEL)" 0

tests/test_exaone_kernels.o: tests/test_exaone_kernels.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_exaone_kernels: tests/test_exaone_kernels.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-exaone-kernels: tests/test_exaone_kernels
	./tests/test_exaone_kernels $(DS4_EXAONE_MODEL)

tests/test_dots3_mtp_guards: tests/test_dots3_mtp_guards.c ds4.c ds4_dots3_mtp.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -Wl,--gc-sections -o $@ $< -lm

tests/test_eos_sampling: tests/test_eos_sampling.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -Wl,--gc-sections -o $@ $< -lm

test-eos-sampling: tests/test_eos_sampling
	./tests/test_eos_sampling

tests/test_dots3_mtp.o: tests/test_dots3_mtp.c ds4.c ds4_dots3_mtp.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_dots3_mtp: tests/test_dots3_mtp.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_dots3_batch.o: tests/test_dots3_batch.c ds4.c ds4_dots3_batch.inc ds4_dots3_mtp.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_dots3_batch: tests/test_dots3_batch.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_dots3_checkpoint.o: tests/test_dots3_checkpoint.c ds4.c ds4_dots3_batch.inc ds4_dots3_mtp.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_dots3_checkpoint: tests/test_dots3_checkpoint.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_exaone_checkpoint.o: tests/test_exaone_checkpoint.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_exaone_checkpoint: tests/test_exaone_checkpoint.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_exaone_partial.o: tests/test_exaone_partial.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_exaone_partial: tests/test_exaone_partial.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_exaone_batch.o: tests/test_exaone_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_exaone_batch: tests/test_exaone_batch.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-exaone-batch: tests/test_exaone_batch
	@test -n "$(DS4_EXAONE_MODEL)" || \
		{ echo "set DS4_EXAONE_MODEL to the first EXAONE GGUF shard" >&2; exit 2; }
	./tests/test_exaone_batch "$(DS4_EXAONE_MODEL)"

tests/test_exaone_forward.o: tests/test_exaone_forward.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_exaone_forward: tests/test_exaone_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_exaone_tokenizer.o: tests/test_exaone_tokenizer.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -c -o $@ $<

tests/test_exaone_tokenizer: tests/test_exaone_tokenizer.o
	$(CC) $(CFLAGS) -O0 -o $@ $^ -Wl,--gc-sections $(LDLIBS)
endif

tests/test_split_gguf: tests/test_split_gguf.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

tests/test_inkling_loader: tests/test_inkling_loader.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

tests/test_step37_loader: tests/test_step37_loader.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

tests/test_step37_state: tests/test_step37_state.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

ifeq ($(UNAME_S),Linux)
tests/test_step37_media.o: tests/test_step37_media.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_step37_media: tests/test_step37_media.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_ling3vl_media.o: tests/test_ling3vl_media.c ds4.c ds4_ling3vl_graph.inc ds4_ling3vl_vision.inc ds4_ling3vl_rope.h ds4_ling3vl_batch.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_ling3vl_media: tests/test_ling3vl_media.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-ling3vl-media
test-ling3vl-media: tests/test_ling3vl_media
	./tests/test_ling3vl_media

tests/test_ling3vl_vision.o: tests/test_ling3vl_vision.c ds4.c ds4_ling3vl_graph.inc ds4_ling3vl_vision.inc ds4_ling3vl_rope.h ds4_ling3vl_batch.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_ling3vl_vision: tests/test_ling3vl_vision.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_vision.o: tests/test_step37_vision.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_vision: tests/test_step37_vision.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_forward.o: tests/test_step37_forward.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_forward: tests/test_step37_forward.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_norm.o: tests/test_step37_norm.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_norm: tests/test_step37_norm.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_session.o: tests/test_step37_session.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_session: tests/test_step37_session.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_mtp.o: tests/test_step37_mtp.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_mtp: tests/test_step37_mtp.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_spec.o: tests/test_step37_spec.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_spec: tests/test_step37_spec.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_cont.o: tests/test_step37_cont.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_step37_cont: tests/test_step37_cont.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_step37_checkpoint.o: tests/test_step37_checkpoint.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_step37_checkpoint: tests/test_step37_checkpoint.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_span_lease.o: tests/test_cuda_span_lease.c ds4.c ds4_step37_graph.inc ds4_step37_vision.inc ds4.h ds4_gpu.h ds4_mem_gov.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_cuda_span_lease: tests/test_cuda_span_lease.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_artifact_scope.o: tests/test_cuda_artifact_scope.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_cuda_artifact_scope: tests/test_cuda_artifact_scope.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-artifact-scope
test-cuda-artifact-scope: tests/test_cuda_artifact_scope
	python3 tests/test_cuda_artifact_scope.py
endif

tests/test_solar_loader: tests/test_solar_loader.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-solar-loader: tests/test_solar_loader
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_loader "$(DS4_SOLAR_MODEL)"

tests/test_glm53_loader: tests/test_glm53_loader.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-glm53-loader: tests/test_glm53_loader
	@test -n "$(DS4_GLM53_MODEL)" || \
		{ echo "set DS4_GLM53_MODEL to GLM-5.3-Flash-Q2.gguf" >&2; exit 2; }
	./tests/test_glm53_loader "$(DS4_GLM53_MODEL)"

tests/test_glm53_quant: tests/test_glm53_quant.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-glm53-quant: tests/test_glm53_quant
	./tests/test_glm53_quant

tests/test_glm53_mixed.o: tests/test_glm53_mixed.c ds4_gpu.h cuda/mmq/ggml-common.h
	$(CC) $(CFLAGS) -std=c11 -I. -c -o $@ $<

tests/test_glm53_mixed: tests/test_glm53_mixed.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-glm53-mixed: tests/test_glm53_mixed
	./tests/test_glm53_mixed

tests/test_glm53_pair: tests/test_glm53_pair.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $^ $(CUDA_LDLIBS)

test-glm53-pair: tests/test_glm53_pair
	./tests/test_glm53_pair

tests/test_glm53_dense_gemm: tests/test_glm53_dense_gemm.cu cuda/glm53_dense_attn.cuh ds4_glm53_attn.h
	$(NVCC) $(NVCCFLAGS) -DTEST_DENSE_GEMM -I. -o $@ $< $(CUDA_LDLIBS)

test-glm53-dense-gemm: tests/test_glm53_dense_gemm
	./tests/test_glm53_dense_gemm

tests/test_glm53_upload_async: tests/test_glm53_upload_async.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $^ $(CUDA_LDLIBS)

test-glm53-upload-async: tests/test_glm53_upload_async
	./tests/test_glm53_upload_async

tests/glm53_state_gate: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example glm53_state_gate --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/glm53_state_gate" $@

tests/glm53_mtp_gate: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --example glm53_mtp_gate --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/examples/glm53_mtp_gate" $@

GLM53_NATIVE_DEPS = ds4_glm53_cache.h ds4_glm53_stream.inc ds4_glm53_compact.h \
	ds4_glm53_graph.inc ds4_glm53_payload.inc ds4_glm53_mtp.inc ds4_glm53_batch.inc \
	ds4_glm53_image.inc
ds4.o: $(GLM53_NATIVE_DEPS)
tests/test_glm53_loader tests/test_glm53_vision_loader tests/test_glm53_image \
	tests/test_glm53_vision.o tests/test_glm53_bounds.o tests/test_glm53_stream \
	tests/test_glm53_payload: $(GLM53_NATIVE_DEPS)
ds4_cuda.o: ds4_glm53_compact.h ds4_glm53_compact_gpu.cuh ds4_glm53_map.inc cuda/glm53_dense_attn.cuh \
	cuda/glm53_vision_norm.cuh ds4_glm53_attn.h cuda/glm53_low_attn.cuh cuda/glm53_pool_score.cuh

tests/test_glm53_cache: tests/test_glm53_cache.c ds4_glm53_cache.h
	$(CC) $(CFLAGS) -Werror -o $@ $<

tests/test_glm53_compact.o: tests/test_glm53_compact.c ds4_glm53_compact.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_glm53_compact: tests/test_glm53_compact.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-glm53-cache test-glm53-compact
test-glm53-cache: tests/test_glm53_cache
	./tests/test_glm53_cache

tests/test_glm53_stream: tests/test_glm53_stream.c ds4.c ds4.h ds4_gpu.h ds4_glm53_cache.h ds4_glm53_stream.inc
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -o $@ $< -Wl,--gc-sections -lm -pthread

.PHONY: test-glm53-stream
test-glm53-stream: tests/test_glm53_stream
	./tests/test_glm53_stream

.PHONY: test-glm53-lifetime
test-glm53-lifetime:
	@glm_fixture_dir=$$(mktemp -d); \
	trap 'rm -rf "$$glm_fixture_dir"' EXIT; \
	python3 tests/test_glm53_close_fixture.py "$$glm_fixture_dir/close" \
		ctor close finish finish_retry finish_cancel && \
	python3 tests/test_glm53_prefill_fixture.py "$$glm_fixture_dir/prefill"

.PHONY: test-glm53-fit
test-glm53-fit:
	@glm_fixture_dir=$$(mktemp -d); \
	trap 'rm -rf "$$glm_fixture_dir"' EXIT; \
	python3 tests/test_glm53_fit_fixture.py "$$glm_fixture_dir/fit" --sanitize

tests/test_glm53_width: tests/test_glm53_width.cu $(DS4_CUDA_CORE_OBJS)
	@test -f "$(DS4_GLM53_WIDTH_FIXTURE)/weights.h" || \
		{ echo "set DS4_GLM53_WIDTH_FIXTURE to the extracted fixture directory" >&2; exit 2; }
	$(NVCC) $(NVCCFLAGS) -I. -I"$(DS4_GLM53_WIDTH_FIXTURE)" -o $@ $^ $(CUDA_LDLIBS)

test-glm53-compact: tests/test_glm53_compact
	./tests/test_glm53_compact

tests/test_glm53_weight.o: tests/test_glm53_weight.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_glm53_weight: tests/test_glm53_weight.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-glm53-weight: tests/test_glm53_weight
	./tests/test_glm53_weight
	./tests/test_glm53_weight freeze

tests/test_glm53_payload: tests/test_glm53_payload.c ds4.c ds4.h ds4_glm53_payload.inc ds4_glm53_compact.h
	$(CC) $(CFLAGS) -O0 -fno-fast-math -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-glm53-payload: tests/test_glm53_payload
	./tests/test_glm53_payload

tests/test_glm53_stop: tests/test_glm53_stop.c ds4.c ds4.h $(GLM53_NATIVE_DEPS)
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

.PHONY: test-glm53-stop
test-glm53-stop: tests/test_glm53_stop
	./tests/test_glm53_stop

tests/test_glm53_tokens: tests/test_glm53_tokens.c ds4.c ds4.h $(GLM53_NATIVE_DEPS)
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

tests/test_glm53_mtp: tests/test_glm53_mtp.c ds4_glm53_mtp.inc ds4_gpu.h
	$(CC) $(CFLAGS) -Werror -Wno-unused-function -I. -o $@ $< -lm

test-glm53-mtp: tests/test_glm53_mtp
	./tests/test_glm53_mtp

tests/test_glm53_banks: tests/test_glm53_banks.c ds4_glm53_batch.inc ds4_glm53_compact.h
	$(CC) $(CFLAGS) -Werror -I. -o $@ $< -lm

test-glm53-banks: tests/test_glm53_banks
	./tests/test_glm53_banks

tests/test_glm53_map: tests/test_glm53_map.c ds4_glm53_map.inc
	$(CC) $(CFLAGS) -Werror -Wno-unused-function -I. -o $@ $<

test-glm53-map: tests/test_glm53_map
	./tests/test_glm53_map

tests/test_glm53_vision_norm: tests/test_glm53_vision_norm.cu cuda/glm53_vision_norm.cuh
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< -lcudart

test-glm53-vision-norm: tests/test_glm53_vision_norm
	./tests/test_glm53_vision_norm

tests/test_glm53_attention: tests/test_glm53_attention.c ds4_glm53_attn.h
	$(CC) $(CFLAGS) -Werror -I. -o $@ $<

tests/test_glm53_attention_cuda: tests/test_glm53_attention.cu ds4_glm53_attn.h cuda/glm53_low_attn.cuh
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< -lcudart

tests/bench_glm53_attention: tests/bench_glm53_attention.cu ds4_glm53_attn.h ds4_glm53_compact.h cuda/glm53_low_attn.cuh
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< -lcudart

tests/bench_glm53_pool: tests/bench_glm53_pool.cu ds4_glm53_compact.h cuda/glm53_pool_score.cuh
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< -lcudart

tests/bench_glm53_shared: tests/bench_glm53_shared.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< $(DS4_CUDA_CORE_OBJS) $(CUDA_LDLIBS)

.PHONY: test-glm53-attention test-glm53-attention-cuda
test-glm53-attention: tests/test_glm53_attention
	./tests/test_glm53_attention

test-glm53-attention-cuda: tests/test_glm53_attention_cuda
	./tests/test_glm53_attention_cuda

.PHONY: test-glm53-quant test-glm53-mixed test-glm53-weight \
	test-glm53-payload test-glm53-mtp test-glm53-banks test-glm53-map test-glm53-vision-norm

tests/test_glm53_vision_loader: tests/test_glm53_vision_loader.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-glm53-vision-loader: tests/test_glm53_vision_loader
	@test -n "$(DS4_GLM53_VISION_MODEL)" || \
		{ echo "set DS4_GLM53_VISION_MODEL to GLM-5.3-Flash-Vision-Encoder.gguf" >&2; exit 2; }
	./tests/test_glm53_vision_loader "$(DS4_GLM53_VISION_MODEL)"

tests/test_glm53_image: tests/test_glm53_image.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-glm53-image: tests/test_glm53_image
	./tests/test_glm53_image

tests/test_glm53_vision.o: tests/test_glm53_vision.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -Wno-unused-function -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_glm53_vision: tests/test_glm53_vision.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-glm53-vision: tests/test_glm53_vision
	@test -n "$(DS4_GLM53_VISION_MODEL)" || \
		{ echo "set DS4_GLM53_VISION_MODEL to GLM-5.3-Flash-Vision-Encoder.gguf" >&2; exit 2; }
	./tests/test_glm53_vision "$(DS4_GLM53_VISION_MODEL)"

tests/test_glm53_dsa.o: tests/test_glm53_dsa.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_glm53_dsa: tests/test_glm53_dsa.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-glm53-dsa: tests/test_glm53_dsa
	./tests/test_glm53_dsa

tests/test_k2_rewind: tests/test_k2_rewind.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -Wl,--gc-sections -o $@ $< -lm

tests/test_deepseek_budget: tests/test_deepseek_budget.c ds4_server.c ds4.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -Wl,--gc-sections -o $@ $< -lm -pthread

tests/test_k2_lifecycle.o: tests/test_k2_lifecycle.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_k2_lifecycle: tests/test_k2_lifecycle.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_glm53_bounds.o: tests/test_glm53_bounds.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. -c -o $@ $<

tests/test_glm53_bounds: tests/test_glm53_bounds.o $(DS4_CUDA_SUPPORT_OBJS)
	$(NVCC) $(NVCCFLAGS) -Xlinker --gc-sections -o $@ $^ $(CUDA_LDLIBS)

test-glm53-bounds: tests/test_glm53_bounds
	./tests/test_glm53_bounds

tests/test_glm53_session.o: tests/test_glm53_session.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_glm53_session: tests/test_glm53_session.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_glm53_long.o: tests/test_glm53_long.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_glm53_long: tests/test_glm53_long.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_glm53_mtp_actual.o: tests/test_glm53_mtp_actual.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_glm53_mtp_actual: tests/test_glm53_mtp_actual.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_glm53_compare: tests/test_glm53_mtp_long.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -DGLM53_COMPARE_ONLY -I. -o $@ $<

tests/test_glm53_mtp_long.o: tests/test_glm53_mtp_long.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_glm53_mtp_long: tests/test_glm53_mtp_long.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bench_glm53_stream.o: tests/bench_glm53_stream.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ $<

tests/bench_glm53_stream: tests/bench_glm53_stream.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

# Build only; the operator runs each arm in a separate process.
.PHONY: bench-glm53-stream glm53-stream-plan
bench-glm53-stream: tests/bench_glm53_stream

glm53-stream-plan:
	@test -n "$(DS4_GLM53_MODEL)" -a -n "$(DS4_GLM53_HASH_RECEIPT)" -a -n "$(DS4_GLM53_IO_PLAN_DIR)" || \
		{ echo "set DS4_GLM53_MODEL, DS4_GLM53_HASH_RECEIPT and DS4_GLM53_IO_PLAN_DIR" >&2; exit 2; }
	python3 tests/glm53_stream_fixture.py --model "$(DS4_GLM53_MODEL)" \
		--receipt "$(DS4_GLM53_HASH_RECEIPT)" --output "$(DS4_GLM53_IO_PLAN_DIR)"

test-glm53-session: tests/test_glm53_session
	@test -n "$(DS4_GLM53_MODEL)" || \
		{ echo "set DS4_GLM53_MODEL to GLM-5.3-Flash-Q2.gguf" >&2; exit 2; }
	./tests/test_glm53_session "$(DS4_GLM53_MODEL)"

test-glm53-multimodal-session: tests/test_glm53_session
	@test -n "$(DS4_GLM53_MODEL)" || \
		{ echo "set DS4_GLM53_MODEL to GLM-5.3-Flash-Q2.gguf" >&2; exit 2; }
	@test -n "$(DS4_GLM53_VISION_MODEL)" || \
		{ echo "set DS4_GLM53_VISION_MODEL to GLM-5.3-Flash-Vision-Encoder.gguf" >&2; exit 2; }
	./tests/test_glm53_session "$(DS4_GLM53_MODEL)" "$(DS4_GLM53_VISION_MODEL)"

tests/test_solar_tokenizer: tests/test_solar_tokenizer.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-solar-tokenizer: tests/test_solar_tokenizer
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_tokenizer "$(DS4_SOLAR_MODEL)"

test: ds4_test ds4-eval tests/test_split_gguf
	./ds4-eval --self-test-extractors
	./ds4_test
	./tests/test_split_gguf

# PQ2_0 block format (Prism/Bonsai ternary).  The block bytes and the expected
# f32 checksums come from the reference dequantizer in the PrismML llama.cpp
# fork, so a mismatch means this tree no longer reads the file the way the
# exporter's runtime does.  tests/pq2_0/reference_checksums.txt is the same
# oracle over all 851 tensors of the Ternary-Bonsai-2-27B-PQ2_0 artifact.
pq2-0-test: tests/test_pq2_0.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_pq2_0 tests/test_pq2_0.c -lm
	./tests/test_pq2_0

# A tiny CPU projection needs no model; sanitizer instrumentation keeps null
# state accesses visible even when the optimizer would discard the load.
tests/test_qwen35_ref: tests/test_qwen35_ref.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O1 -DDS4_NO_GPU -Wno-unused-function \
	-fsanitize=undefined -fno-sanitize-recover=undefined \
	-ffunction-sections -fdata-sections -I. -o $@ $< \
	-Wl,--gc-sections $(LDLIBS)

.PHONY: test-qwen35-ref
test-qwen35-ref: tests/test_qwen35_ref
	./tests/test_qwen35_ref

# Bonsai (qwen35) reference checks.  They need the Prism Bonsai GGUF and the
# CPU host binary (make cpu); no llama.cpp is involved.  DS4_BONSAI_MODEL and
# DS4_BONSAI_STEPS override the artifact path and the greedy step count.
DS4_BONSAI_MODEL ?= /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
DS4_BONSAI_STEPS ?= 12

# Every tensor of the artifact read through this tree's own row reader, against
# the checksums the exporter's ggml dequantizer produced (tests/pq2_0).
tests/test_qwen35_rows: tests/test_qwen35_rows.c ds4.c ds4.h tests/pq2_0/reference_checksums.txt
	$(CC) $(CFLAGS) -O2 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
	-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-qwen35-rows: tests/test_qwen35_rows
	@test -n "$(DS4_BONSAI_MODEL)" || \
	{ echo "set DS4_BONSAI_MODEL to the Prism Bonsai GGUF" >&2; exit 2; }
	./tests/test_qwen35_rows "$(DS4_BONSAI_MODEL)"

.PHONY: bonsai-fold-selftest bonsai-ref-check
bonsai-fold-selftest:
	DS4_QWEN35_FOLD_SELFTEST=1 ./ds4-c -m "$(DS4_BONSAI_MODEL)" --cpu --first-token-test -p "x" | grep "fold selftest"

bonsai-ref-check:
	DS4_QWEN35_STEPS="$${DS4_QWEN35_STEPS:-$(DS4_BONSAI_STEPS)}" ./ds4-c -m "$(DS4_BONSAI_MODEL)" --cpu --first-token-test -p "The capital of France is" | grep -E "^token|next-token"

# The same greedy check on the CUDA graph.  On this box the whole-map host
# registration fails (RLIMIT_MEMLOCK is 8 MiB), so the artifact is copied to the
# device instead: DS4_CUDA_COPY_MODEL=1.  Needs the CUDA build of ds4-c.
DS4_BONSAI_PARITY_TOKENS ?= 760,6511,314,9338,369
DS4_BONSAI_PARITY_STEPS ?= 8

.PHONY: bonsai-cuda-check bonsai-cuda-parity
bonsai-cuda-check:
	DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_STEPS="$${DS4_QWEN35_STEPS:-$(DS4_BONSAI_STEPS)}" ./ds4-c -m "$(DS4_BONSAI_MODEL)" --cuda --first-token-test -p "The capital of France is" | grep -E "^token|next-token"

# CPU reference and CUDA graph on the same prompt ids: the two streams must
# print the same ids.  Slow, the CPU reference is about 3 s per token.
bonsai-cuda-parity:
	@mkdir -p misc/scratch
	DS4_QWEN35_TOKENS=$(DS4_BONSAI_PARITY_TOKENS) DS4_QWEN35_STEPS=$(DS4_BONSAI_PARITY_STEPS) ./ds4-c -m "$(DS4_BONSAI_MODEL)" --cpu --first-token-test -p x 2>/dev/null | grep -E "^token " > misc/scratch/bonsai-cpu.tokens
	DS4_CUDA_COPY_MODEL=1 DS4_QWEN35_TOKENS=$(DS4_BONSAI_PARITY_TOKENS) DS4_QWEN35_STEPS=$(DS4_BONSAI_PARITY_STEPS) ./ds4-c -m "$(DS4_BONSAI_MODEL)" --cuda --first-token-test -p x 2>/dev/null | grep -E "^token " > misc/scratch/bonsai-cuda.tokens
	@diff misc/scratch/bonsai-cpu.tokens misc/scratch/bonsai-cuda.tokens && echo "bonsai cuda parity: PASS"

# Metadata and full tensor-layout smoke. The structural GGUF is sparse, so
# this validates all descriptors without materializing an 88 GiB copy.
tests/test_motif3_loader: tests/test_motif3_loader.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-motif3-loader: tests/test_motif3_loader
	@test -n "$(DS4_MOTIF3_MODEL)" || \
		{ echo "set DS4_MOTIF3_MODEL to the structural or completed GGUF" >&2; exit 2; }
	./tests/test_motif3_loader "$(DS4_MOTIF3_MODEL)"

tests/test_dots3_loader: tests/test_dots3_loader.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-dots3-loader: tests/test_dots3_loader
	@test -n "$(DS4_DOTS3_MODEL)" || \
		{ echo "set DS4_DOTS3_MODEL to the first dots3 GGUF shard" >&2; exit 2; }
	./tests/test_dots3_loader "$(DS4_DOTS3_MODEL)"

tests/test_qwen4exp_loader: tests/test_qwen4exp_loader.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-qwen4exp-loader: tests/test_qwen4exp_loader
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the first Qwen4Exp GGUF shard" >&2; exit 2; }
	./tests/test_qwen4exp_loader "$(DS4_QWEN4EXP_MODEL)"

tests/test_qwen4exp_tokenizer: tests/test_qwen4exp_tokenizer.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-qwen4exp-tokenizer: tests/test_qwen4exp_tokenizer
	@test -n "$(DS4_QWEN4EXP_MODEL)" || \
		{ echo "set DS4_QWEN4EXP_MODEL to the structural or completed Qwen4Exp GGUF" >&2; exit 2; }
	./tests/test_qwen4exp_tokenizer "$(DS4_QWEN4EXP_MODEL)"

tests/test_qwen4exp_ple: tests/test_qwen4exp_ple.c ds4_ple.c ds4_ple.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_qwen4exp_ple.c ds4_ple.c $(LDLIBS)

test-qwen4exp-ple: tests/test_qwen4exp_ple
	@test -n "$(DS4_QWEN4EXP_ROOT)" || \
		{ echo "set DS4_QWEN4EXP_ROOT to the SSD-PLE artifact root" >&2; exit 2; }
	./tests/test_qwen4exp_ple "$(DS4_QWEN4EXP_ROOT)"

tests/libds4ple_test.so: ds4_ple.c ds4_ple.h
	$(CC) $(CFLAGS) -fPIC -shared -I. -o $@ ds4_ple.c $(LDLIBS)

.PHONY: test-ple-formats
test-ple-formats: tests/libds4ple_test.so
	python3 tests/test_ple_fp8.py

test-qwen4exp-ple-reference: tests/libds4ple_test.so
	@test -n "$(DS4_QWEN4EXP_ROOT)" || \
		{ echo "set DS4_QWEN4EXP_ROOT to the SSD-PLE artifact root" >&2; exit 2; }
	@test -n "$(DS4_QWEN4EXP_SOURCE)" || \
		{ echo "set DS4_QWEN4EXP_SOURCE to the pinned safetensors root" >&2; exit 2; }
	python3 tests/test_qwen4exp_ple_reference.py \
		--library tests/libds4ple_test.so \
		--artifact-root "$(DS4_QWEN4EXP_ROOT)" \
		--source-root "$(DS4_QWEN4EXP_SOURCE)"

ifeq ($(UNAME_S),Darwin)
test-qwen4exp-ple-cuda:
	@echo "test-qwen4exp-ple-cuda requires a CUDA build"
else
cuda/qwen38_ple.o: cuda/qwen38_ple.cu cuda/qwen38_ple.h ds4_ple.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ $<

tests/test_qwen4exp_ple_cuda.o: tests/test_qwen4exp_ple_cuda.cu cuda/qwen38_ple.h ds4_ple.h
	$(NVCC) $(NVCCFLAGS) -I. -c -o $@ $<

tests/test_qwen4exp_ple_cuda: tests/test_qwen4exp_ple_cuda.o cuda/qwen38_ple.o ds4_ple.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-qwen4exp-ple-cuda: tests/test_qwen4exp_ple_cuda
	@test -n "$(DS4_QWEN4EXP_ROOT)" || \
		{ echo "set DS4_QWEN4EXP_ROOT to the SSD-PLE artifact root" >&2; exit 2; }
	./tests/test_qwen4exp_ple_cuda "$(DS4_QWEN4EXP_ROOT)"
endif

tests/test_motif3_reference: tests/test_motif3_reference.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-motif3-reference: tests/test_motif3_reference
	./tests/test_motif3_reference "$(DS4_MOTIF3_FIXTURES)"

tests/test_dots3_tokenizer: tests/test_dots3_tokenizer.c tests/dots3_tokenizer_goldens.inc ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-dots3-tokenizer: tests/test_dots3_tokenizer
	@test -n "$(DS4_DOTS3_MODEL)" || \
		{ echo "set DS4_DOTS3_MODEL to the first dots3 GGUF shard" >&2; exit 2; }
	./tests/test_dots3_tokenizer "$(DS4_DOTS3_MODEL)"

tests/test_motif3_tokenizer: tests/test_motif3_tokenizer.c ds4.c ds4.h
	$(CC) $(CFLAGS) -O0 -DDS4_NO_GPU -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-motif3-tokenizer: tests/test_motif3_tokenizer
	@test -n "$(DS4_MOTIF3_MODEL)" || \
		{ echo "set DS4_MOTIF3_MODEL to the structural or completed GGUF" >&2; exit 2; }
	./tests/test_motif3_tokenizer "$(DS4_MOTIF3_MODEL)" \
		"$(DS4_MOTIF3_FIXTURES)/tokenizer-chat.ds4tok"

ifneq ($(UNAME_S),Darwin)
tests/test_motif3_cuda: tests/test_motif3_cuda.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< $(DS4_CUDA_CORE_OBJS) $(CUDA_LDLIBS)

test-motif3-cuda: tests/test_motif3_cuda
	./tests/test_motif3_cuda "$(DS4_MOTIF3_FIXTURES)"

tests/test_dots3_cuda: tests/test_dots3_cuda.cu $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -I. -o $@ $< $(DS4_CUDA_CORE_OBJS) $(CUDA_LDLIBS)

test-dots3-cuda: tests/test_dots3_cuda
	./tests/test_dots3_cuda

tests/test_dots3_resident.o: tests/test_dots3_resident.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_dots3_resident: tests/test_dots3_resident.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-dots3-resident: tests/test_dots3_resident
	@test -n "$(DS4_DOTS3_MODEL)" || \
		{ echo "set DS4_DOTS3_MODEL to the first dots3 GGUF shard" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES=0 ./tests/test_dots3_resident "$(DS4_DOTS3_MODEL)"

tests/test_motif3_resident.o: tests/test_motif3_resident.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_motif3_resident: tests/test_motif3_resident.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-motif3-resident: tests/test_motif3_resident
	CUDA_VISIBLE_DEVICES=0 ./tests/test_motif3_resident "$(DS4_MOTIF3_MODEL)"

tests/test_motif3_batch.o: tests/test_motif3_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_motif3_batch: tests/test_motif3_batch.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-motif3-batch: tests/test_motif3_batch
	@test -n "$(DS4_MOTIF3_MODEL)" || \
		{ echo "set DS4_MOTIF3_MODEL to the completed GGUF" >&2; exit 2; }
	CUDA_VISIBLE_DEVICES=0 ./tests/test_motif3_batch "$(DS4_MOTIF3_MODEL)"

tests/test_motif3_long.o: tests/test_motif3_long.c ds4.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_motif3_long: tests/test_motif3_long.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

clean:
	rm -f tests/bench_glm53_attention tests/bench_glm53_pool tests/bench_glm53_shared
	rm -f tests/bench_glm53_stream tests/bench_glm53_stream.o
	rm -f tests/test_glm53_tokens tests/test_glm53_long tests/test_glm53_long.o tests/test_glm53_mtp_actual tests/test_glm53_mtp_actual.o
	rm -f tests/test_glm53_compare tests/test_glm53_mtp_long tests/test_glm53_mtp_long.o
	rm -f tests/test_glm53_quant tests/test_glm53_mixed tests/test_glm53_mixed.o tests/test_glm53_compact tests/test_glm53_compact.o tests/test_glm53_cache tests/test_glm53_stream tests/test_glm53_weight tests/test_glm53_weight.o tests/test_glm53_payload tests/test_glm53_stop tests/test_glm53_mtp tests/test_glm53_banks tests/test_glm53_map tests/test_glm53_vision_norm tests/test_glm53_attention tests/test_glm53_attention_cuda
	rm -f tests/test_qwen35_ref
	rm -f tests/test_solar_fattn tests/test_solar_fattn.o
	rm -f tests/test_step37_media tests/test_step37_media.o
	rm -f tests/test_ling3vl_media tests/test_ling3vl_media.o
	rm -f tests/test_ling3vl_vision tests/test_ling3vl_vision.o
	rm -f tests/test_ling3vl_primitives
	rm -f tests/test_ling3vl_matmul tests/test_ling3vl_matmul.o
	rm -f tests/test_ling3vl_q5pair tests/test_ling3vl_q5pair.o
	rm -f tests/test_ling3vl_moefuse tests/test_ling3vl_moefuse.o
	rm -f tests/test_ling3vl_mla tests/test_ling3vl_mla_expand
	rm -f tests/test_step37_vision_ops tests/test_step37_vision tests/test_step37_vision.o
	rm -f tests/test_step37_primitives tests/test_step37_loader tests/test_step37_forward tests/test_step37_forward.o
	rm -f tests/test_step37_session tests/test_step37_session.o tests/test_step37_state
	rm -f tests/test_step37_mtp tests/test_step37_mtp.o
	rm -f tests/test_step37_spec tests/test_step37_spec.o
	rm -f tests/test_step37_cont tests/test_step37_cont.o
	rm -f tests/test_dots3_mtp tests/test_dots3_mtp.o tests/test_dots3_mtp_guards tests/test_eos_sampling
	rm -f tests/test_dots3_checkpoint tests/test_dots3_checkpoint.o tests/test_dots3_batch tests/test_dots3_batch.o
	rm -f tests/test_exaone_partial tests/test_exaone_partial.o
	rm -f tests/test_exaone_checkpoint tests/test_exaone_checkpoint.o
	rm -f tests/test_step37_checkpoint tests/test_step37_checkpoint.o
	rm -f tests/test_step37_norm tests/test_step37_norm.o
	rm -f tests/test_cuda_span_lease tests/test_cuda_span_lease.o
	rm -f tests/test_cuda_artifact_scope tests/test_cuda_artifact_scope.o
	rm -f tests/test_inkling_kernels tests/test_inkling_kernels.o
	rm -f tests/test_inkling_moe tests/test_inkling_moe.o
	rm -f tests/test_inkling_attn_prep tests/test_inkling_attn_prep.o
	rm -f tests/test_inkling_attention tests/test_inkling_attention.o
	rm -f tests/test_inkling_norm tests/test_inkling_norm.o
	rm -f tests/test_inkling_linear tests/test_inkling_linear.o
	rm -f tests/test_inkling_batch cuda/mmq/test/test_inkling_batch.o
	rm -f tests/test_inkling_q8_batch tests/test_inkling_q8_batch.o
	rm -f tests/test_inkling_media tests/test_inkling_media.o
	rm -f tests/test_inkling_loader
	rm -f tests/test_inkling_forward tests/test_inkling_forward.o
	rm -f tests/test_inkling_session tests/test_inkling_session.o
	rm -f tests/test_inkling_mtp tests/test_inkling_mtp.o
	rm -f tests/test_inkling_mtp_shared tests/test_inkling_mtp_shared.o
	rm -f tests/test_inkling_encoders tests/test_inkling_encoders.o
	rm -f tests/test_qwen_vision_norm tests/test_qwen_vision_norm.o
	rm -f tests/test_qwen_vision_rope tests/test_qwen_vision_rope.o
	rm -f tests/test_qwen_vision_attention tests/test_qwen_vision_attention.o tests/test_qwen_vision_model tests/test_qwen_vision_model.o
	rm -f tests/test_qwen_vision_host tests/test_qwen_vision_host.o
	rm -f ds4-agent-rs tests/parity/agent_c_oracle tests/parity/agent_c_oracle.o
	rm -f tests/test_glm53_loader tests/test_glm53_vision_loader tests/test_glm53_image tests/test_glm53_vision tests/test_glm53_vision.o tests/test_glm53_dsa tests/test_glm53_dsa.o tests/test_glm53_session tests/test_glm53_session.o tests/test_glm53_bounds tests/test_glm53_bounds.o tests/test_k2_lifecycle tests/test_k2_lifecycle.o
	rm -f tests/test_k2_rewind tests/test_deepseek_budget
	rm -f ds4 ds4-server ds4-bench ds4-bench-perf ds4-eval ds4-agent ds4-c ds4-server-c ds4-bench-c ds4-agent-c ds4-rs ds4-bench-rs ds4-server-rs ds4-agent-rs ds4_weight_server tests/parity/shape_c_oracle tests/parity/shape_c_oracle.o tests/parity/catalog_c_oracle tests/parity/catalog_c_oracle.o tests/parity/tensor_c_oracle tests/parity/tensor_c_oracle.o tests/parity/bind_c_oracle tests/parity/bind_c_oracle.o tests/parity/bind_lookup_c_oracle tests/parity/bind_lookup_c_oracle.o tests/parity/load_c_oracle tests/parity/load_c_oracle.o tests/parity/validate_c_oracle tests/parity/validate_c_oracle.o tests/parity/layout_c_oracle tests/parity/layout_c_oracle.o tests/parity/vocab_c_oracle tests/parity/vocab_c_oracle.o tests/parity/tokenizer_c_oracle tests/parity/tokenizer_c_oracle.o tests/parity/session_c_oracle tests/parity/session_c_oracle.o tests/parity/payload_c_oracle tests/parity/payload_c_oracle.o tests/parity/kv_c_oracle tests/parity/kv_c_oracle.o tests/parity/kv_c_stubs.o tests/parity/web_c_oracle tests/parity/web_c_oracle.o tests/parity/dist_c_oracle tests/parity/dist_c_oracle.o tests/parity/route_c_oracle tests/parity/route_c_oracle.o tests/parity/server_c_oracle tests/parity/server_c_oracle.o tests/parity/parse_c_oracle tests/parity/parse_c_oracle.o tests/parity/stream_c_oracle tests/parity/stream_c_oracle.o tests/parity/tool_stream_c_oracle tests/parity/tool_stream_c_oracle.o tests/parity/dsml_c_oracle tests/parity/dsml_c_oracle.o tests/parity/retry_c_oracle tests/parity/retry_c_oracle.o tests/parity/admit_c_oracle tests/parity/admit_c_oracle.o tests/parity/render_c_oracle tests/parity/render_c_oracle.o tests/parity/bridge_null_oracle tests/parity/bridge_null_oracle.o tests/parity/bridge_null_stubs.o tests/parity/cont_c_oracle tests/parity/cont_c_oracle.o tests/parity/memgov_c_oracle tests/parity/memgov_c_oracle.o ds4_cpu ds4_native ds4_server_test ds4_test tests/test_motif3_loader tests/test_motif3_reference tests/test_motif3_tokenizer tests/test_motif3_cuda tests/test_motif3_resident tests/test_motif3_batch tests/test_motif3_long tests/test_motif3_resident.o tests/test_motif3_batch.o tests/test_motif3_long.o tests/test_exaone_ref tests/test_exaone_kernels tests/test_exaone_batch tests/test_exaone_ref.o tests/test_exaone_kernels.o tests/test_exaone_batch.o *.o cuda/mmq/test/test_mmq_parity.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/cuda_tokentile_ldmatrix tests/cuda_tokentile_ldmatrix.o tests/test_split_gguf tests/test_solar_loader tests/test_solar_tokenizer tests/test_repack_premapped tests/test_mmq_parity tests/test_mmid_fast tests/test_mmid_fast.o tests/test_model_family_kernels tests/test_model_family_kernels.o tests/test_solar_forward tests/test_solar_forward.o tests/test_solar_session tests/test_solar_session.o tests/test_solar_kda tests/test_solar_kda_prefill tests/test_solar_kda_chunk tests/test_glm53_dsa tests/test_glm53_dsa.o tests/test_solar_gates tests/test_solar_kv tests/test_solar_kda.o tests/test_solar_kda_prefill.o tests/test_solar_kda_chunk.o tests/test_solar_gates.o tests/test_solar_kv.o native/bridge/ds4_bridge.o
	rm -f tests/test_qwen4exp_loader tests/test_qwen4exp_tokenizer tests/test_qwen4exp_ple tests/test_qwen4exp_ple_cuda tests/test_qwen4exp_ple_cuda.o tests/test_qwen4exp_primitives tests/test_qwen4exp_primitives.o tests/test_qwen4exp_hc_forward tests/test_qwen4exp_hc_forward.o tests/test_qwen4exp_ple_compute tests/test_qwen4exp_ple_compute.o tests/test_qwen4exp_ple_forward tests/test_qwen4exp_ple_forward.o tests/test_qwen4exp_moe tests/test_qwen4exp_moe_forward tests/test_qwen4exp_gdn tests/test_qwen4exp_gdn_forward tests/test_qwen4exp_qsa tests/test_qwen4exp_qsa_forward tests/test_qwen4exp_batch tests/test_qwen4exp_batch.o tests/libds4ple_test.so cuda/qwen38_ple.o
