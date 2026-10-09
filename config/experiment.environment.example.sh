# Copy to experiment.environment.sh (ignored), edit, then source before Jupyter.
# Use absolute paths. Put outputs and model caches on your allocated project disk.
export EXPERIMENT4_CODE_ROOT="/path/to/ECG_CXR_alignment/experiment"
export ECG_CXR_BRIDGE_ARTIFACT_ROOT="/path/to/ecg_cxr_bridge_artifacts"
export EXPERIMENT4_ROOT="$ECG_CXR_BRIDGE_ARTIFACT_ROOT/experiment4"
export SYMILE_ROOT="/path/to/symile-mimic/1.0.0"
export HF_HOME="/path/to/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export MPLCONFIGDIR="$EXPERIMENT4_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="/path/to/pip_cache"

# Optional: token file outside the repository, containing one read token.
# Otherwise use `hf auth login` or a secret-managed HF_TOKEN environment variable.
# export HF_TOKEN_FILE="/path/to/private/hf_token.txt"
# export PULSE_REPO_ROOT="/path/to/PULSE"
# export E4_IDENTITY_CROSSWALK="/path/to/private/verified_identity_crosswalk.csv"

# Pilot first. Set each limit to 0 only when ready to run all pending cases.
export E4_MAX_GENERATION_ROWS=50
export E4_MAX_SPECIALIST_ROWS=50
export E4_GENERATION_RUN="generation_v3"
# Notebook 13C uses pulse_compact_v1; see analysis_model_runs.json.
export E4_ANALYSIS_ID="jbhi_five_models_v1"
export E4_ALLOW_INCOMPLETE=0

# Extension notebooks 17–21: keep this root distinct from the base experiment.
export E4_REVIEW_ROOT="$EXPERIMENT4_ROOT/review_extension_v1"
export E4_REVIEW_METHODS=clip
export E4_REVIEW_QUERY_N=250
export E4_REVIEW_RENDER_N=100
export E4_REVIEW_MAX_ROWS=50
export E4_REVIEW_IMAGES_APPROVED=0
export E4_REVIEW_ALLOW_PARTIAL=0
export E4_REVIEW_BOOTSTRAPS=1000
