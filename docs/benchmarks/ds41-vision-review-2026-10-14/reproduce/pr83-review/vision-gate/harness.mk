VISION_DIR := $(abspath scratch/pr83-review/vision-gate)
VISION_FREEZE := $(abspath scratch/vq-pr80-81/port-long-batch/freeze.py)
VISION_CORE_RLIB ?=
VISION_DEPS = $(dir $(abspath $(VISION_CORE_RLIB)))
VISION_INPUTS = $(VISION_DIR)/vision_gate.rs $(VISION_DIR)/harness.mk $(VISION_CORE_RLIB) $(DS4_RS_LINK_OBJS)

# Deliberately no prerequisites: this links already-built, frozen inputs only.
.PHONY: vision-gate
vision-gate:
	@test -n "$(VISION_CORE_RLIB)" || { echo 'Pass VISION_CORE_RLIB from the final native build' >&2; exit 1; }
	python3 $(VISION_FREEZE) before $(VISION_DIR)/frozen-inputs.json $(VISION_INPUTS)
	rustc --edition=2021 -C opt-level=3 --crate-name vision_gate $(VISION_DIR)/vision_gate.rs \
		-L dependency=$(VISION_DEPS) --extern ds4_core=$(VISION_CORE_RLIB) \
		$(patsubst %,-C link-arg=$(DS4_RS_ROOT)/%,$(DS4_RS_LINK_OBJS)) $(DS4_RS_LIBS) \
		-o $(VISION_DIR)/vision_gate
	python3 $(VISION_FREEZE) after $(VISION_DIR)/frozen-inputs.json $(VISION_INPUTS)
	sha256sum $(VISION_DIR)/vision_gate.rs $(VISION_DIR)/harness.mk $(VISION_DIR)/vision_gate $(VISION_DIR)/frozen-inputs.json > $(VISION_DIR)/build.sha256
