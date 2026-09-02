CC ?= cc
UNAME_S := $(shell uname -s)

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
        proof-cuda-smoke proof-cuda-long proof-cuda-opp-c \
        proof-rust-cuda-opp-c print-version \
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
        test-mmid-fast \
        test-mmq-parity test-model-family-kernels \
        test-solar-loader test-solar-kda test-solar-kda-prefill \
        test-solar-kda-chunk \
        test-solar-gates test-solar-kv test-solar-tokenizer \
        test-solar-forward test-solar-session \
        test-exaone-ref test-exaone-kernels test-exaone-batch \
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

proof-cuda-smoke proof-cuda-long proof-cuda-opp-c proof-rust-cuda-opp-c:
	@echo "$@ requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10 (sm_121a arch alias)"
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

ds4.o: ds4.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_gpu.h vendor/stb_image.h
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
DS4_RS_SOURCES := Cargo.toml Cargo.lock $(shell find crates -type f -print)
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

ds4-bench: $(DS4_RS_SOURCES) native/bridge/ds4_bridge.o $(CORE_OBJS)
	cargo rustc -p ds4-cli --bin ds4-bench-rs --release --features native -- \
		-C metadata=$(DS4_RS_LINK_FINGERPRINT) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) \
		$(DS4_RS_LIBS)
	cp -f "$(DS4_RS_TARGET_DIR)/release/ds4-bench-rs" $@

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

tests/test_solar_kda.o: tests/test_solar_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_kda_prefill.o: tests/test_solar_kda_prefill.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_gates.o: tests/test_solar_gates.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_solar_kv.o: tests/test_solar_kv.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_model_family_kernels.o: tests/test_model_family_kernels.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

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

ds4_cpu.o: ds4.c ds4.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_distributed.h ds4_gpu.h
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

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_mem_census.h ds4_model_catalog.h ds4_mem_gov.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_repack.h cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces. ds4_mmq.cu transitively pulls in mmq.cuh which has
# heavy template instantiation - compile in its own TU and link in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

# Shared aligned-artifact layout library: compiled into the engine (via
# MMQ_OBJS) and linked into ds4_weight_server so both producers build
# bit-identical repack artifacts.
cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

cuda/mmq/ds4_fattn.o: cuda/mmq/ds4_fattn.cu cuda/mmq/common.cuh cuda/mmq/mma.cuh cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

tests/test_repack_premapped: tests/test_repack_premapped.cu cuda/mmq/ds4_repack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-repack-premapped: tests/test_repack_premapped
	./tests/test_repack_premapped

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

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

cuda/mmq/test/test_mmq_parity.o: cuda/mmq/test/test_mmq_parity.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) $(MMQ_INCLUDES) -c -o $@ $<

tests/test_mmq_parity: cuda/mmq/test/test_mmq_parity.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-mmq-parity: tests/test_mmq_parity
	./tests/test_mmq_parity

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

tests/test_qwen4exp_primitives: tests/test_qwen4exp_primitives.o $(DS4_CUDA_CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

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

tests/test_solar_loader: tests/test_solar_loader.c ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections \
		-Wno-unused-function -I. -o $@ $< -Wl,--gc-sections $(LDLIBS)

test-solar-loader: tests/test_solar_loader
	@test -n "$(DS4_SOLAR_MODEL)" || \
		{ echo "set DS4_SOLAR_MODEL to the first Solar GGUF shard" >&2; exit 2; }
	./tests/test_solar_loader "$(DS4_SOLAR_MODEL)"

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
	rm -f ds4-agent-rs tests/parity/agent_c_oracle tests/parity/agent_c_oracle.o
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4-c ds4-server-c ds4-bench-c ds4-agent-c ds4-rs ds4-bench-rs ds4-server-rs ds4-agent-rs ds4_weight_server tests/parity/shape_c_oracle tests/parity/shape_c_oracle.o tests/parity/catalog_c_oracle tests/parity/catalog_c_oracle.o tests/parity/tensor_c_oracle tests/parity/tensor_c_oracle.o tests/parity/bind_c_oracle tests/parity/bind_c_oracle.o tests/parity/bind_lookup_c_oracle tests/parity/bind_lookup_c_oracle.o tests/parity/load_c_oracle tests/parity/load_c_oracle.o tests/parity/validate_c_oracle tests/parity/validate_c_oracle.o tests/parity/layout_c_oracle tests/parity/layout_c_oracle.o tests/parity/vocab_c_oracle tests/parity/vocab_c_oracle.o tests/parity/tokenizer_c_oracle tests/parity/tokenizer_c_oracle.o tests/parity/session_c_oracle tests/parity/session_c_oracle.o tests/parity/payload_c_oracle tests/parity/payload_c_oracle.o tests/parity/kv_c_oracle tests/parity/kv_c_oracle.o tests/parity/kv_c_stubs.o tests/parity/web_c_oracle tests/parity/web_c_oracle.o tests/parity/dist_c_oracle tests/parity/dist_c_oracle.o tests/parity/route_c_oracle tests/parity/route_c_oracle.o tests/parity/server_c_oracle tests/parity/server_c_oracle.o tests/parity/parse_c_oracle tests/parity/parse_c_oracle.o tests/parity/stream_c_oracle tests/parity/stream_c_oracle.o tests/parity/tool_stream_c_oracle tests/parity/tool_stream_c_oracle.o tests/parity/dsml_c_oracle tests/parity/dsml_c_oracle.o tests/parity/retry_c_oracle tests/parity/retry_c_oracle.o tests/parity/admit_c_oracle tests/parity/admit_c_oracle.o tests/parity/render_c_oracle tests/parity/render_c_oracle.o tests/parity/bridge_null_oracle tests/parity/bridge_null_oracle.o tests/parity/bridge_null_stubs.o tests/parity/cont_c_oracle tests/parity/cont_c_oracle.o tests/parity/memgov_c_oracle tests/parity/memgov_c_oracle.o ds4_cpu ds4_native ds4_server_test ds4_test tests/test_motif3_loader tests/test_motif3_reference tests/test_motif3_tokenizer tests/test_motif3_cuda tests/test_motif3_resident tests/test_motif3_batch tests/test_motif3_long tests/test_motif3_resident.o tests/test_motif3_batch.o tests/test_motif3_long.o tests/test_exaone_ref tests/test_exaone_kernels tests/test_exaone_batch tests/test_exaone_ref.o tests/test_exaone_kernels.o tests/test_exaone_batch.o *.o cuda/mmq/test/test_mmq_parity.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/test_split_gguf tests/test_solar_loader tests/test_solar_tokenizer tests/test_repack_premapped tests/test_mmq_parity tests/test_mmid_fast tests/test_mmid_fast.o tests/test_model_family_kernels tests/test_model_family_kernels.o tests/test_solar_forward tests/test_solar_forward.o tests/test_solar_session tests/test_solar_session.o tests/test_solar_kda tests/test_solar_kda_prefill tests/test_solar_kda_chunk tests/test_solar_gates tests/test_solar_kv tests/test_solar_kda.o tests/test_solar_kda_prefill.o tests/test_solar_kda_chunk.o tests/test_solar_gates.o tests/test_solar_kv.o native/bridge/ds4_bridge.o
	rm -f tests/test_qwen4exp_loader tests/test_qwen4exp_tokenizer tests/test_qwen4exp_ple tests/test_qwen4exp_ple_cuda tests/test_qwen4exp_ple_cuda.o tests/test_qwen4exp_primitives tests/test_qwen4exp_primitives.o tests/test_qwen4exp_hc_forward tests/test_qwen4exp_hc_forward.o tests/test_qwen4exp_ple_compute tests/test_qwen4exp_ple_compute.o tests/test_qwen4exp_ple_forward tests/test_qwen4exp_ple_forward.o tests/test_qwen4exp_moe tests/test_qwen4exp_moe_forward tests/test_qwen4exp_gdn tests/test_qwen4exp_gdn_forward tests/test_qwen4exp_qsa tests/test_qwen4exp_qsa_forward tests/test_qwen4exp_batch tests/test_qwen4exp_batch.o tests/libds4ple_test.so cuda/qwen38_ple.o
