import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
EXPECTED = [
    "00_environment_and_configuration.ipynb",
    "01_verify_symile_ecg_cxr_pairs.ipynb",
    "02_build_ecg_cxr_pair_manifests.ipynb",
    "03_encode_ecg_hubert.ipynb",
    "04_encode_cxr_raddino.ipynb",
    "05_define_ecg_cxr_codebind.ipynb",
    "06_train_ecg_cxr_bridge.ipynb",
    "07_evaluate_ecg_cxr_bridge.ipynb",
    "08_external_mimic_adapter_contract.ipynb",
    "09_external_ablation_design_and_configuration.ipynb",
    "10_prepare_external_ecg_mimic.ipynb",
    "11_prepare_external_cxr_mimic.ipynb",
    "12_train_symile_bridge_ablations.ipynb",
    "13_encode_external_modalities_with_shared_codebook.ipynb",
    "14_encode_external_text_and_concepts.ipynb",
    "15_train_external_semantic_adapters.ipynb",
    "16_evaluate_external_retrieval_ablations.ipynb",
    "17_configure_rag_report_generation.ipynb",
    "18_build_rag_retrieval_contexts.ipynb",
    "19_generate_rag_reports.ipynb",
    "20_evaluate_rag_report_generation.ipynb",
]


def load_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    assert isinstance(notebook.get("cells"), list) and notebook["cells"]
    return notebook


def test_ordered_notebook_release_is_complete():
    observed = sorted(path.name for path in NOTEBOOK_ROOT.glob("*.ipynb"))
    assert observed == EXPECTED


def test_release_notebooks_have_no_saved_outputs_or_execution_counts():
    for name in EXPECTED:
        notebook = load_notebook(NOTEBOOK_ROOT / name)
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("outputs", []) == [], name
            assert cell.get("execution_count") is None, name


def test_required_configuration_contract_is_present():
    combined = "\n".join(
        (NOTEBOOK_ROOT / name).read_text(encoding="utf-8") for name in EXPECTED
    )
    for variable in [
        "SYMILE_ROOT",
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "MIMIC_ECG_ROOT",
        "MIMIC_CXR_TRAIN_ROOT",
        "MIMIC_CXR_TEST_ROOT",
        "MIMIC_CXR_EXPECTED_TRAIN_IMAGES",
        "RAG_MODELS",
    ]:
        assert variable in combined
