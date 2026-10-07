"""Three-modality CLIP, CodeBind-inspired, and hybrid bridge experiments."""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from e4_common import (ARTIFACT_ROOT, EXPERIMENT4_ROOT, MODALITIES, SPLITS,
                       SYMILE_ROOT, atomic_numpy, read_jsonl, seed_all, sha256,
                       signature, write_json, write_jsonl)


def _observed_labs(percentiles, mask):
    p, m = np.asarray(percentiles, dtype=np.float32), np.asarray(mask)
    if p.ndim != 2 or p.shape[1] != 50 or m.shape != p.shape:
        raise ValueError(f"Expected aligned (n,50) lab arrays, got {p.shape}/{m.shape}")
    if not np.isin(m, [0, 1]).all():
        raise ValueError("Symile labs_missingness must be binary; 1 means observed")
    observed = m.astype(bool)
    if not np.isfinite(p[observed]).all() or np.any((p[observed] < 0) | (p[observed] > 1)):
        raise ValueError("Observed Symile lab percentiles must be finite in [0,1]")
    return np.concatenate([np.where(observed, p, .5), m.astype(np.float32)], axis=1)


def prepare_symile(artifact=ARTIFACT_ROOT, symile=SYMILE_ROOT, out=EXPERIMENT4_ROOT):
    import pandas as pd
    from e4_evidence import LAB_IDS
    report = {"status": "preparing", "splits": {}}
    write_json(out / "bridge" / "symile_preparation.json", report)
    patient_sets = {}
    for split in SPLITS:
        linked = read_jsonl(out / "partition" / "symile" / f"{split}.jsonl")
        if not linked:
            raise FileNotFoundError("Run notebook 01 before preparing Symile")
        pdir = symile / "data_npy" / split
        frame = pd.read_csv(symile / f'{split}.csv')
        hadm = np.load(pdir / f'hadm_id_{split}.npy', mmap_mode='r').reshape(-1)
        if len(frame) != len(linked) or not np.array_equal(hadm, frame.hadm_id.to_numpy()):
            raise ValueError(f'{split}: CSV/admission array identity mismatch')
        if any(int(r['npy_index']) != i or int(r['hadm_id']) != int(frame.iloc[i].hadm_id)
               or int(r['subject_id']) != int(frame.iloc[i].subject_id) for i, r in enumerate(linked)):
            raise ValueError(f'{split}: linked manifest order/identity mismatch')
        percentiles = np.load(pdir / f"labs_percentiles_{split}.npy", mmap_mode="r")
        mask = np.load(pdir / f"labs_missingness_{split}.npy", mmap_mode="r")
        labs_all = _observed_labs(percentiles, mask)
        if np.any(mask.sum(axis=1) == 0): raise ValueError('Symile row has no observed laboratory tests.')
        means = symile / 'labs_means.json'
        expected = [k + '_percentile' for k in LAB_IDS]
        if means.exists() and sorted(json.loads(means.read_text())) != expected:
            raise ValueError('Unexpected lab feature order in labs_means.json')
        checked = 0
        for j, lab in enumerate(LAB_IDS):
            seen = np.asarray(mask[:, j]).astype(bool)
            if lab in frame and not np.array_equal(frame[lab].notna().to_numpy(), seen):
                raise ValueError(f'{split}: reversed observation mask or wrong lab order for {lab}')
            if lab + '_percentile' in frame:
                checked += 1
                if not np.allclose(frame.loc[seen, lab+'_percentile'], percentiles[seen,j], atol=1e-6):
                    raise ValueError(f'{split}: lab percentile array/CSV mismatch for {lab}')
        if not means.exists() and checked != 50:
            raise ValueError('Cannot verify encoded lab order without labs_means.json or 50 percentile CSV columns.')
        ecg_all = np.load(artifact / "features" / split / "ecg_hubert.npy", mmap_mode="r")
        cxr_all = np.load(artifact / "features" / split / "cxr_raddino.npy", mmap_mode="r")
        if not (len(linked) == len(labs_all) == len(ecg_all) == len(cxr_all)):
            raise ValueError(f"{split}: manifest/features/labs row mismatch")
        if any(a.ndim != 2 or not np.isfinite(a).all() for a in [ecg_all,cxr_all]):
            raise ValueError('Invalid frozen sensor features.')
        groups = {}
        for i, row in enumerate(linked):
            groups.setdefault(int(row["hadm_id"]), []).append(i)
        chosen = np.asarray([indexes[0] for indexes in groups.values()], dtype=np.int64)
        for indexes in groups.values():
            if len(indexes) > 1:
                for array, name in [(labs_all, "labs"), (ecg_all, "ecg"), (cxr_all, "cxr")]:
                    if not np.allclose(array[indexes], array[indexes[0]], atol=1e-5):
                        raise ValueError(f"{split}: repeated admission has inconsistent {name}")
        records = [{**linked[i], "source_index": int(i)} for i in chosen]
        dest = out / "bridge" / "prepared" / split
        write_jsonl(dest / "manifest.jsonl", records)
        atomic_numpy(dest / "ecg.npy", np.asarray(ecg_all[chosen], dtype=np.float32))
        atomic_numpy(dest / "cxr.npy", np.asarray(cxr_all[chosen], dtype=np.float32))
        atomic_numpy(dest / "labs.npy", np.asarray(labs_all[chosen], dtype=np.float32))
        patients = {int(r["subject_id"]) for r in records}
        patient_sets[split] = patients
        report["splits"][split] = {
            "source_rows": len(linked), "unique_admissions": len(records),
            "patients": len(patients), "observed_lab_fraction": float(labs_all[chosen, 50:].mean()),
            "input_hashes": {str(p): sha256(p) for p in [symile / f'{split}.csv',
                pdir / f'labs_percentiles_{split}.npy', pdir / f'labs_missingness_{split}.npy',
                artifact / 'features' / split / 'ecg_hubert.npy', artifact / 'features' / split / 'cxr_raddino.npy',
                out / 'partition' / 'symile' / f'{split}.jsonl']},
        }
    for a, b in itertools.combinations(SPLITS, 2):
        if patient_sets[a] & patient_sets[b]:
            raise AssertionError(f"Symile patient leakage: {a}/{b}")
    report["status"] = "complete"
    report["patient_disjoint"] = True
    report['partition_signature'] = json.loads((out/'partition'/'global_partition_audit.json').read_text())['partition_signature']
    report['code_sha256'] = sha256(__file__)
    report["signature"] = signature(report)
    write_json(out / "bridge" / "symile_preparation.json", report)
    return report


def load_split(out, split):
    root = Path(out) / "bridge" / "prepared" / split
    records = read_jsonl(root / "manifest.jsonl")
    arrays = {m: np.load(root / f"{m}.npy", mmap_mode="r") for m in MODALITIES}
    return records, arrays


class CompositionalCodebook(nn.Module):
    def __init__(self, size, code_dim, temperature=.1):
        super().__init__()
        self.code_dim, self.temperature = int(code_dim), float(temperature)
        self.codes = nn.Parameter(torch.randn(size, code_dim) * code_dim ** -.5)

    def forward(self, features):
        chunks = features.reshape(features.shape[0], -1, self.code_dim)
        logits = torch.einsum("bnd,kd->bnk", F.normalize(chunks, dim=-1),
                              F.normalize(self.codes, dim=-1))
        indexes = logits.argmax(-1)
        hard = F.embedding(indexes, self.codes)
        commitment = F.mse_loss(chunks, hard.detach()) + F.mse_loss(hard, chunks.detach())
        quantized = chunks + (hard - chunks).detach()
        return quantized.flatten(1), indexes, F.softmax(logits / self.temperature, -1), commitment


@dataclass
class Encoding:
    representation: torch.Tensor
    continuous: torch.Tensor
    quantized: torch.Tensor | None
    probabilities: torch.Tensor | None
    commitment: torch.Tensor
    specific_continuous: torch.Tensor | None
    reconstruction: torch.Tensor | None


class TriModalBridge(nn.Module):
    def __init__(self, dims, method="clip", common_dim=256, specific_dim=256,
                 code_dim=8, common_codes=512, specific_codes=256, dropout=.1):
        super().__init__()
        if method not in {"clip", "codebind", "combined"}:
            raise ValueError(method)
        if common_dim % code_dim or specific_dim % code_dim:
            raise ValueError('Projection dimensions must be divisible by code dimension.')
        self.method, self.common_dim, self.specific_dim = method, common_dim, specific_dim
        uses_vq = method != "clip"
        total = common_dim + (specific_dim if uses_vq else 0)
        self.projectors = nn.ModuleDict({m: nn.Sequential(
            nn.LayerNorm(dims[m]), nn.Dropout(dropout), nn.Linear(dims[m], total)
        ) for m in MODALITIES})
        self.common_codebook = CompositionalCodebook(common_codes, code_dim) if uses_vq else None
        self.specific_codebooks = nn.ModuleDict({m: CompositionalCodebook(specific_codes, code_dim)
                                                 for m in MODALITIES}) if uses_vq else None
        self.decoders = nn.ModuleDict({m: nn.Sequential(
            nn.LayerNorm(common_dim + specific_dim), nn.Linear(common_dim + specific_dim, dims[m])
        ) for m in MODALITIES}) if uses_vq else None

    @property
    def uses_vq(self):
        return self.common_codebook is not None

    def encode_modality(self, modality, features):
        projected = self.projectors[modality](features)
        continuous_raw = projected[:, :self.common_dim]
        continuous = F.normalize(continuous_raw, dim=-1)
        if not self.uses_vq:
            zero = continuous.sum() * 0
            return Encoding(continuous, continuous, None, None, zero, None, None)
        quantized_raw, _, probs, c1 = self.common_codebook(continuous_raw)
        specific_raw = projected[:, self.common_dim:]
        specific_quantized, _, _, c2 = self.specific_codebooks[modality](specific_raw)
        quantized = F.normalize(quantized_raw, dim=-1)
        representation = quantized if self.method == "codebind" else F.normalize(
            continuous + quantized, dim=-1)
        reconstruction = self.decoders[modality](torch.cat([quantized_raw, specific_quantized], -1))
        return Encoding(representation, continuous, quantized, probs, (c1 + c2) / 2,
                        specific_raw, reconstruction)

    def forward(self, batch):
        return {m: self.encode_modality(m, batch[m]) for m in MODALITIES if m in batch}


def _pair_loss(a, b, patients, temperature=.07):
    logits = a @ b.T / temperature
    same = patients[:, None].eq(patients[None, :])
    offdiag = ~torch.eye(len(patients), dtype=torch.bool, device=patients.device)
    logits = logits.masked_fill(same & offdiag, -1e4)
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def _js(a, b):
    eps = torch.finfo(a.dtype).eps
    mid = .5 * (a + b)
    return .5 * ((a * ((a + eps).log() - (mid + eps).log())).sum(-1) +
                 (b * ((b + eps).log() - (mid + eps).log())).sum(-1)).mean()


def _orthogonal(a, b):
    a = (a - a.mean(0)) / a.std(0, unbiased=False).clamp_min(1e-4)
    b = (b - b.mean(0)) / b.std(0, unbiased=False).clamp_min(1e-4)
    return ((a.T @ b / len(a)) ** 2).mean()


def bridge_loss(model, outputs, batch, no_code_match=False):
    pairs = list(itertools.combinations(MODALITIES, 2))
    representation_loss = torch.stack([
        _pair_loss(outputs[a].representation, outputs[b].representation, batch["subject_id"])
        for a, b in pairs]).mean()
    zero = representation_loss * 0
    continuous_loss = zero
    if model.method == "combined":
        continuous_loss = torch.stack([
            _pair_loss(outputs[a].continuous, outputs[b].continuous, batch["subject_id"])
            for a, b in pairs]).mean()
    if model.uses_vq:
        relation = torch.stack([_js(outputs[a].probabilities, outputs[b].probabilities)
                                for a, b in pairs]).mean()
        commitment = torch.stack([outputs[m].commitment for m in MODALITIES]).mean()
        orth = torch.stack([_orthogonal(outputs[m].continuous, outputs[m].specific_continuous)
                            for m in MODALITIES]).mean()
        reconstruction = torch.stack([F.mse_loss(
            outputs[m].reconstruction, F.layer_norm(batch[m], (batch[m].shape[-1],)))
            for m in MODALITIES]).mean()
        usage = torch.stack([outputs[m].probabilities.mean(0) for m in MODALITIES])
        usage = (usage * (usage + 1e-8).log()).sum(-1).mean()
    else:
        relation = commitment = orth = reconstruction = usage = zero
    total = representation_loss + .5 * continuous_loss + (0 if no_code_match else .1) * relation
    total = total + .25 * commitment + .05 * orth + .10 * reconstruction + .01 * usage
    return total


def _derangement(patients, seed):
    rng = np.random.default_rng(seed)
    patients = np.asarray(patients)
    for _ in range(10000):
        order = rng.permutation(len(patients))
        if np.all(patients != patients[order]):
            return order
    raise RuntimeError("Unable to form cross-patient derangement")


@torch.inference_mode()
def encode(model, arrays, device="cpu", batch_size=1024):
    result = {m: [] for m in arrays if m in MODALITIES}
    model.eval()
    for start in range(0, len(next(iter(arrays.values()))), batch_size):
        batch = {m: torch.as_tensor(np.array(arrays[m][start:start + batch_size], copy=True), device=device)
                 for m in result}
        for m, value in model(batch).items():
            result[m].append(value.representation.cpu().numpy())
    return {m: np.concatenate(parts) if parts else np.empty((0, model.common_dim), dtype=np.float32)
            for m, parts in result.items()}


def retrieval(query, bank):
    if len(query) != len(bank) or len(bank) < 2:
        raise ValueError('Paired retrieval requires at least two aligned candidate admissions.')
    scores = np.asarray(query) @ np.asarray(bank).T
    truth = np.arange(len(query))
    ranks, margins = [], []
    for i, row in enumerate(scores):
        rank = 1 + np.sum(row > row[truth[i]]) + (np.sum(row == row[truth[i]]) - 1) / 2
        order = np.argsort(-row, kind="stable")
        ranks.append(float(rank))
        margins.append(float(row[order[0]] - row[order[1]]))
    ranks = np.asarray(ranks)
    return {"queries": len(ranks), "hit1": float(np.mean(ranks <= 1)),
            "hit5": float(np.mean(ranks <= 5)), "mrr": float(np.mean(1 / ranks))}, ranks, margins


def validation_score(model, arrays, device):
    z = encode(model, arrays, device)
    values = []
    for a, b in itertools.permutations(MODALITIES, 2):
        values.append(retrieval(z[a], z[b])[0]["mrr"])
    return float(np.mean(values))


def train(out=EXPERIMENT4_ROOT, method="clip", pairing="genuine", labs_mode="values",
          no_code_match=False, seed=43, epochs=50, batch_size=128, lr=3e-4,
          patience=8, device=None):
    if pairing not in {"genuine", "shuffled"} or labs_mode not in {"values", "mask_only"}:
        raise ValueError((pairing, labs_mode))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)
    records, base = load_split(out, "train")
    _, val_base = load_split(out, "val")
    arrays = {m: np.array(base[m], copy=True) for m in MODALITIES}
    val = {m: np.array(val_base[m], copy=True) for m in MODALITIES}
    patients = np.asarray([r["subject_id"] for r in records], dtype=np.int64)
    if pairing == "shuffled":
        rng = np.random.default_rng(seed)
        arrays["cxr"] = arrays["cxr"][_derangement(patients, seed)]
        arrays["labs"] = arrays["labs"][_derangement(patients, seed + 10000)]
    if labs_mode == "mask_only":
        arrays["labs"][:, :50] = .5
        val["labs"][:, :50] = .5
    dims = {m: int(arrays[m].shape[1]) for m in MODALITIES}
    protocol = {"method": method, "pairing": pairing, "labs_mode": labs_mode,
                "no_code_match": bool(no_code_match), "seed": seed, "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "patience": patience, "dims": dims,
                "selection": "mean validation MRR over six directed modality pairs",
                "code_sha256": sha256(__file__),
                "symile_signature": json.loads((out / "bridge" / "symile_preparation.json").read_text())["signature"]}
    run_name = f"{method}__{pairing}__{labs_mode}" + ("__no_code_match" if no_code_match else "")
    run = out / "bridge" / "models" / run_name / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    protocol_path = run / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f"Run protocol changed; choose a fresh EXPERIMENT4_ROOT: {run}")
    write_json(protocol_path, protocol)
    if (run / "complete.json").exists():
        return json.loads((run / "complete.json").read_text())
    model = TriModalBridge(dims, method=method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history, best, stale, first = [], -math.inf, 0, 0
    if (run / "last.pt").exists():
        state = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        history, best, stale, first = state["history"], state["best"], state["stale"], state["epoch"] + 1
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    for epoch in range(first, epochs):
        model.train(); losses = []
        order = rng.permutation(len(records))
        for start in range(0, len(order), batch_size):
            indexes = order[start:start + batch_size]
            if len(indexes) < 2: continue
            batch = {m: torch.as_tensor(arrays[m][indexes], device=device) for m in MODALITIES}
            batch["subject_id"] = torch.as_tensor(patients[indexes], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bridge_loss(model, model(batch), batch, no_code_match)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        score = validation_score(model, val, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_mean_mrr": score})
        if score > best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "protocol": protocol}, run / "best.pt")
        else:
            stale += 1
        temporary = run / 'last.tmp.pt'
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "history": history, "best": best, "stale": stale,
                    "numpy_rng": rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temporary)
        temporary.replace(run / 'last.pt')
        write_json(run / "history.json", history)
        if stale >= patience: break
    result = {"run_name": run_name, "method": method, "pairing": pairing,
              "labs_mode": labs_mode, "no_code_match": bool(no_code_match), "seed": seed,
              "validation_mean_mrr": best, "checkpoint": str(run / "best.pt"),
              "checkpoint_sha256": sha256(run / 'best.pt')}
    write_json(run / "complete.json", result)
    return result


def load_model(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    p = state["protocol"]
    model = TriModalBridge(p["dims"], method=p["method"]).to(device)
    model.load_state_dict(state["model"]); model.eval()
    return model, p


def evaluate_runs(out=EXPERIMENT4_ROOT, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs = [json.loads(p.read_text()) for p in (out / "bridge" / "models").rglob("complete.json")]
    rows = []
    for run in runs:
        model, protocol = load_model(run["checkpoint"], device=device)
        for split in ("val", "test"):
            _, arrays0 = load_split(out, split)
            arrays = {m: np.array(arrays0[m], copy=True) for m in MODALITIES}
            if protocol["labs_mode"] == "mask_only": arrays["labs"][:, :50] = .5
            z = encode(model, arrays, device=device)
            for a, b in itertools.permutations(MODALITIES, 2):
                metrics, _, _ = retrieval(z[a], z[b])
                rows.append({**run, "split": split, "direction": f"{a}_to_{b}", **metrics})
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "bridge" / "retrieval_evaluation.csv", index=False)
    selected = {}
    for method in ["clip", "codebind", "combined"]:
        eligible = [r for r in runs if r["method"] == method and r["pairing"] == "genuine"
                    and r["labs_mode"] == "values" and not r["no_code_match"]]
        if not eligible: continue
        selected[method] = max(eligible, key=lambda r: r["validation_mean_mrr"])
        controls = [r for r in runs if r["method"] == method and r not in eligible]
        selected[method]["controls"] = {}
        for control_name, predicate in {
            "shuffled": lambda r: r["pairing"] == "shuffled" and r["labs_mode"] == "values",
            "mask_only": lambda r: r["pairing"] == "genuine" and r["labs_mode"] == "mask_only",
            "no_code_match": lambda r: r["no_code_match"],
        }.items():
            matches = [r for r in controls if predicate(r)]
            if matches:
                selected[method]["controls"][control_name] = max(matches, key=lambda r: r["validation_mean_mrr"])
    write_json(out / "bridge" / "selected_bridges.json", selected)
    return frame, selected


def calibrate_confidence_gates(out=EXPERIMENT4_ROOT, device=None):
    from e4_evidence import calibrate_gates
    return calibrate_gates(out)
