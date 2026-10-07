"""Dependency-free, direction-aware generator registry for Experiment 4."""

GENERAL_MODELS = {
    "qwen35_4b": {"model_id": "Qwen/Qwen3.5-4B", "mode": "multimodal",
                    "directions": ("ecg", "cxr"), "role": "general_multimodal"},
    "medgemma15_4b": {"model_id": "google/medgemma-1.5-4b-it", "mode": "multimodal",
                       "directions": ("cxr",), "role": "cxr_specialist",
                       "training_overlap": "model card discloses MIMIC-CXR training; exact evaluation-patient overlap unverified"},
    "gpt_oss_20b": {"model_id": "openai/gpt-oss-20b", "mode": "text_only",
                     "directions": ("ecg", "cxr"), "role": "retrieval_text_only"},
}

SPECIALIST_MODELS = {
    "pulse7b_ecg": {
        "model_id": "PULSE-ECG/PULSE-7B", "loader": "pulse_llava",
        "mode": "multimodal", "directions": ("ecg",),
        "role": "ecg_image_specialist", "license": "apache-2.0",
        "training_overlap": (
            "ECGInstruct construction used MIMIC-IV-ECG reports; exact patient-level "
            "overlap with this evaluation is not verifiable"
        ),
    },
    "ecg_instruct_llama32_11b": {
        "model_id": "convaiinnovations/ECG-Instruct-Llama-3.2-11B-Vision",
        "loader": "mllama_transformers", "mode": "multimodal", "directions": ("ecg",),
        "role": "ecg_image_specialist", "license": "llama3",
        "training_overlap": (
            "model card discloses ECGInstruct derived partly from MIMIC-IV-ECG; "
            "exact patient-level overlap with this evaluation is not verifiable"
        ),
    },
}

ALL_MODELS = {**GENERAL_MODELS, **SPECIALIST_MODELS}
