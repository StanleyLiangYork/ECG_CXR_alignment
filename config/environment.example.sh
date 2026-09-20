#!/usr/bin/env bash

# Copy this file to config/environment.sh, edit the paths, and source it.
# Never commit the copied file, credentials, or protected-data locations.

export SYMILE_ROOT="/path/to/symile-mimic/1.0.0"
export MIMIC_ECG_ROOT="/path/to/mimic-iv-ecg/1.0"
export MIMIC_CXR_TRAIN_ROOT="/path/to/multi_kg/train"
export MIMIC_CXR_TEST_ROOT="/path/to/multi_kg/test"
export MIMIC_CXR_EXPECTED_TRAIN_IMAGES="43700"

export ECG_CXR_BRIDGE_ARTIFACT_ROOT="/path/to/ecg_cxr_bridge_artifacts"
export HF_HOME="/path/to/huggingface_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"

# Encoding and training defaults; tune for the allocated GPU and CPU resources.
export HUBERT_BATCH_SIZE="8"
export RADDINO_BATCH_SIZE="16"
export ECG_CXR_NUM_WORKERS="4"
export ABLATION_NUM_WORKERS="4"
export EXTERNAL_ECG_BATCH_SIZE="32"
export EXTERNAL_CXR_BATCH_SIZE="64"
export EXTERNAL_TEXT_BATCH_SIZE="256"

# Reproducible full experiment defaults.
export ABLATION_VARIANTS="clip_continuous,shared_vq,shared_specific_vq,shared_specific_no_code_match,shuffled_pairs_control"
export ABLATION_SEEDS="43,44,45"
export SEMANTIC_ADAPTER_SEEDS="43,44,45"
export RAG_MODELS="medgemma_4b,qwen35_4b"
export RAG_MAX_TEST_ECG="250"
export RAG_CONTEXT_CASES="4"
export RAG_BOTH_ECG_CASES="2"
export RAG_BOTH_CXR_CASES="2"
export RAG_GENERATION_BATCH_SIZE="4"
export RAG_EVAL_BOOTSTRAPS="500"
