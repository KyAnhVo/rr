# Sets up everything a full five-stage run needs, from a fresh clone. Only uv is assumed.
#
#   make setup          both environments, Mask2Former weights, Replica scenes + semantic labels
#   make check          confirm both environments can reach the GPU
#   make test           unit tests + every downloaded scene (hours; sleep is blocked meanwhile)
#   make test-unit      unit tests only (seconds)
#   make benchmark      the same scenes without pytest
#   make run VIDEO=room.mp4 OUTPUT=out/
#                       all five stages on your own video, then meshes:
#                       $(OUTPUT)/pictures/ (per-phase pictures) and $(OUTPUT)/mesh/ (.ply meshes)
#
# Variables: SCENES (default "room0 office0"), SEG_MODEL, MODES (default "estimated known"),
#            MESH_DEPTH (Poisson detail for `run`, default 10; 11-12 is finer but 4-16x larger).
# Example:   make setup SCENES=room0 && make benchmark SCENES=room0 MODES=estimated

SCENES     ?= room0 office0
MODES      ?= estimated known
SEG_MODEL  ?= facebook/mask2former-swin-large-ade-semantic
MESH_DEPTH ?= 10
DATA      := room-data/replica

# Two environments, because pycolmap needs CUDA 12 and torch pins CUDA 13 (see README).
VENV     := .venv
SEG_VENV := .venv-seg
WEIGHTS  := $(SEG_VENV)/.weights-$(subst /,_,$(SEG_MODEL))

SCENE_FILES    := $(foreach s,$(SCENES),$(DATA)/$(s)/traj.txt $(DATA)/$(s)_mesh.ply)
SEMANTIC_FILES := $(foreach s,$(SCENES),$(DATA)/$(s)_semantic/mesh_semantic.ply)

# Keeps the machine awake through long runs; released when the command exits.
AWAKE := systemd-inhibit --what=sleep:idle --who=rr --why="rr scene run"

.PHONY: setup envs data semantics weights check test test-unit benchmark run clean-envs

setup: envs weights data semantics check

envs: $(VENV)/.synced $(SEG_VENV)/.synced

# Stages 1-3, 5 and the tests: numpy, OpenCV, pycolmap-cuda12, pytest.
$(VENV)/.synced: pyproject.toml uv.lock .python-version
	uv python install
	uv sync
	@touch $@

# Stage 4: torch + transformers. The pipeline calls this environment as a subprocess.
# After .venv (order only): two parallel syncs (make -j) could both rewrite uv.lock.
$(SEG_VENV)/.synced: pyproject.toml uv.lock .python-version | $(VENV)/.synced
	uv python install
	UV_PROJECT_ENVIRONMENT=$(SEG_VENV) uv sync --no-group colmap --group segmentation
	@touch $@

# About 1 GB into the Hugging Face cache, so the first stage 4 run does not stall on it.
weights: $(WEIGHTS)
$(WEIGHTS): $(SEG_VENV)/.synced
	$(SEG_VENV)/bin/python -c 'from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation as M; \
		AutoImageProcessor.from_pretrained("$(SEG_MODEL)"); M.from_pretrained("$(SEG_MODEL)")'
	@touch $@

# RGB-D sequences and ground-truth meshes (range requests into a 12.4 GB zip; quick).
# Non-commercial research/education terms.
data: $(SCENE_FILES)
$(SCENE_FILES) &:
	uv run --no-project scripts/download_replica.py $(SCENES)

# Semantic labels for scoring stage 5. Streams tens of GB, so this is the slow step.
semantics: $(SEMANTIC_FILES)
$(SEMANTIC_FILES) &:
	uv run --no-project scripts/download_replica_semantics.py $(SCENES)

check: envs
	$(VENV)/bin/python -c 'import pycolmap; assert pycolmap.has_cuda, "pycolmap was built without CUDA"; \
		print("pycolmap", pycolmap.__version__, "with CUDA")'
	$(SEG_VENV)/bin/python -c 'import torch; assert torch.cuda.is_available(), "torch sees no GPU (driver updated without a reboot?)"; \
		print("torch", torch.__version__, "on", torch.cuda.get_device_name())'

test: setup
	$(AWAKE) uv run pytest

test-unit: $(VENV)/.synced
	uv run pytest -m "not scenes"

benchmark: setup
	$(AWAKE) uv run rr-benchmark --scenes $(SCENES) --modes $(MODES) --seg-model $(SEG_MODEL)

# Your own video (or a Replica scene directory). Needs no dataset downloads. OUTPUT is created
# if missing; it may already exist, but must not hold an earlier run (stages refuse to overwrite).
ifneq ($(filter run,$(MAKECMDGOALS)),)
  ifeq ($(strip $(VIDEO)),)
    $(error usage: make run VIDEO=room.mp4 OUTPUT=out/)
  endif
  ifeq ($(strip $(OUTPUT)),)
    $(error usage: make run VIDEO=room.mp4 OUTPUT=out/)
  endif
  ifeq ($(wildcard $(VIDEO)),)
    $(error VIDEO not found: $(VIDEO))
  endif
endif

run: envs weights check
	mkdir -p "$(OUTPUT)"
	$(AWAKE) uv run rr "$(VIDEO)" "$(OUTPUT)" --seg-model "$(SEG_MODEL)"
	uv run python -m rr.mesh "$(OUTPUT)" --depth $(MESH_DEPTH)
	@echo "pictures: $(OUTPUT)/pictures/   meshes: $(OUTPUT)/mesh/"

# Environments only; downloaded data and results are kept.
clean-envs:
	rm -rf $(VENV) $(SEG_VENV)
