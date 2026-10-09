"""Post-review controls, isolated from all completed Experiment 4 runs.

No existing context, generation, image or clinical-review output is modified.
Frozen ECG cosine retrieval is genuinely ECG-only (no trimodal training).
The bridge ECG-only arm isolates scoring, not representation training.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

VERSION = "review_controls_v1"
DEFAULT_ROOT = Path(str(Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", str(Path(__file__).resolve().parent.parent / "artifacts"))) / "experiment4"))
ARMS = ("zero_shot", "raw_ecg_same", "raw_ecg_full", "bridge_ecg_same",
        "trimodal_same", "trimodal_full", "trimodal_broken")


def roots():
    source = Path(os.environ.get("EXPERIMENT4_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    dest = Path(os.environ.get("E4_REVIEW_ROOT", source / "review_extension_v1")).expanduser().resolve()
    if dest == source or dest in source.parents:
        raise ValueError("E4_REVIEW_ROOT must be a dedicated output directory, never the original root")
    return source, dest


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024**2), b""): h.update(b)
    return h.hexdigest()


def sig(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        for row in rows: f.write(json.dumps(row, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def identity(row):
    # Identifier matching must be exact; no basename or row-position guesses.
    return tuple(str(row.get(k)) for k in ("subject_id", "hadm_id", "ecg_study_id", "cxr_study_id"))


def unique_index(rows, field):
    keys = [str(r[field]) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError(f"Duplicate {field}")
    return {key: i for i, key in enumerate(keys)}


def normalized(values):
    x = np.asarray(values, dtype=np.float32)
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if x.ndim != 2 or not np.isfinite(x).all() or (lengths <= 1e-12).any():
        raise ValueError("Features must be finite nonzero row vectors")
    return x / lengths


def rank(query, bank, records, subject, k=2):
    scores = np.asarray(bank @ query)
    allowed = np.array([str(r["subject_id"]) != str(subject) for r in records])
    order = [int(i) for i in np.argsort(-scores, kind="stable") if allowed[i]]
    if len(order) < k: raise ValueError("Insufficient patient-disjoint candidates")
    return order[:k], [float(scores[i]) for i in order[:k]]


def make_anchor(r, score, order):
    return {"anchor_id": str(r["sample_id"]), "subject_id": r["subject_id"],
            "hadm_id": r["hadm_id"], "rank": order, "score": score,
            "linkage": "same Symile admission; exact ECG/CXR study IDs",
            "ecg": {"machine_diagnosis": r["ecg_machine_diagnosis"], "study_id": r["ecg_study_id"]},
            "cxr": {"report": r["cxr_report"], "labels": r.get("cxr_labels", []),
                    "study_id": r["cxr_study_id"]}, "labs": r["lab_summary"]}


def only_ecg(anchors):
    return [{k: copy.deepcopy(v) for k, v in a.items() if k not in {"cxr", "labs"}}
            for a in anchors]


def break_linkage(anchors, seed=4410):
    """A constrained permutation preserves the exact global CXR/lab text multiset.

    ECG anchors/rank are fixed. CXR+lab bundles move together across patients.
    Matching uses component lengths only, never diagnostic labels or outcomes.
    True donor provenance is retained in audit fields but hidden from generation
    to avoid changing the prompt alongside the manipulated linkage.
    """
    n = len(anchors)
    if n < 2: raise ValueError("Broken linkage needs at least two anchor slots")
    lengths = np.array([[len(str(a["cxr"].get("report", ""))[:1800]),
                         len(str(a["labs"].get("text", ""))[:1800])] for a in anchors])
    rng = np.random.default_rng(seed)
    cost = np.abs(lengths[:, None] - lengths[None, :]).sum(2).astype(float)
    cost += rng.uniform(0, .01, (n, n))
    forbidden = np.array([[str(a["subject_id"]) == str(b["subject_id"]) for b in anchors]
                          for a in anchors])
    cost[forbidden] = 1e12
    # Patient-grouped cyclic bijections avoid an extra SciPy dependency. Choose
    # the lowest-length-cost valid rotation, not an outcome-optimized assignment.
    ii = np.array(sorted(range(n), key=lambda i: (str(anchors[i]['subject_id']), i)))
    feasible = [(float(cost[ii, np.roll(ii, shift)].sum()), shift) for shift in range(1,n)
                if not forbidden[ii, np.roll(ii, shift)].any()]
    if not feasible:
        raise ValueError("No cross-patient derangement exists; increase the outcome-blind cohort")
    _, shift = min(feasible)
    donor_by_slot = dict(zip(ii.tolist(), np.roll(ii, shift).tolist()))
    result, audit = [], []
    for i in range(n):
        j = donor_by_slot[i]
        a, donor = copy.deepcopy(anchors[i]), anchors[j]
        a["cxr"], a["labs"] = copy.deepcopy(donor["cxr"]), copy.deepcopy(donor["labs"])
        a["component_donor"] = {k: donor[k] for k in ("anchor_id", "subject_id", "hadm_id")}
        a["linkage_is_experimentally_broken"] = True
        result.append(a)
        audit.append({"slot": int(i), "donor_slot": int(j), "ecg_anchor": a["anchor_id"],
                      "donor_anchor": donor["anchor_id"], "length_difference": int(cost[i,j]),
                      "ecg_subject": a["subject_id"], "donor_subject": donor["subject_id"]})
    return result, audit


def prepare_controls(source=None, dest=None, methods=("clip",), max_queries=250,
                     render_subset=100, seed=4410):
    source0, dest0 = roots(); source, dest = Path(source or source0), Path(dest or dest0)
    if not methods or len(set(methods)) != len(methods) or max_queries < 2 or render_subset < 2:
        raise ValueError("Invalid method/cohort settings")
    inputs = []
    def rows(p): inputs.append(p); return read_rows(p)
    def array(p): inputs.append(p); return np.load(p, mmap_mode="r")
    audit_path = source / "partition/global_partition_audit.json"; inputs.append(audit_path)
    audit = json.loads(audit_path.read_text())
    if not audit.get("patient_disjoint_verified") or audit.get("status") != "complete":
        raise ValueError("Completed global patient-disjoint audit required")
    original = rows(source / "contexts/bidirectional_contexts.jsonl")
    contexts = [r for r in original if r["direction"] == "ecg" and
                r["method"] == methods[0] and r["condition"] == "full_trimodal"]
    unique_index(contexts, "sample_id")
    contexts.sort(key=lambda r: sig([seed, str(r["sample_id"])]))
    cohort = contexts[:max_queries]
    if len(cohort) < 2: raise ValueError("No original ECG cohort found")
    calibration_ids = {str(r["sample_id"]) for r in cohort[:render_subset]}
    prepared = rows(source / "bridge/prepared/train/manifest.jsonl")
    prepared_idx = unique_index(prepared, "sample_id")
    raw_train = array(source / "bridge/prepared/train/ecg.npy")
    external = rows(source / "external/ecg/test/manifest.jsonl")
    external_idx = unique_index(external, "sample_id")
    raw_external = array(source / "external/ecg/test/foundation.npy")
    if len(prepared) != len(raw_train) or len(external) != len(raw_external):
        raise ValueError("Feature/manifest size mismatch")
    all_rows, retrieval, permutations = [], [], []
    bank_ids0 = None
    for method in methods:
        bank_root = source / "evidence/banks" / method / "genuine/symile"
        records = rows(bank_root / "manifest.jsonl")
        unique_index(records, "sample_id")
        if bank_ids0 is None: bank_ids0 = [identity(r) for r in records]
        elif bank_ids0 != [identity(r) for r in records]: raise ValueError("Candidate banks differ")
        if {str(r["subject_id"]) for r in records} & {str(r["subject_id"]) for r in external}:
            raise ValueError("Evidence/query patient leakage")
        indexes = []
        for r in records:
            j = prepared_idx[str(r["sample_id"])]
            if identity(r) != identity(prepared[j]): raise ValueError("Foundation bank identity mismatch")
            indexes.append(j)
        raw_bank = normalized(raw_train[indexes])
        ecg_bank = np.asarray(array(bank_root / "ecg.npy"))
        tri_bank = (ecg_bank + array(bank_root / "cxr.npy") + array(bank_root / "labs.npy")) / 3
        if len(tri_bank) != len(records): raise ValueError("Bridge bank row mismatch")
        qroot = source / "evidence/queries" / method / "genuine/ecg/test"
        qrecords = rows(qroot / "manifest.jsonl"); qidx = unique_index(qrecords, "sample_id")
        qvectors = array(qroot / "ecg.npy")
        if len(qrecords) != len(qvectors): raise ValueError("Bridge query row mismatch")
        full_rows = []
        for old in cohort:
            sid = str(old["sample_id"]); ei = external_idx[sid]; qi = qidx[sid]
            query = external[ei]
            if str(query["subject_id"]) != str(old["subject_id"]) or query != qrecords[qi]:
                raise ValueError("Query identity/metadata changed; recreate frozen study inputs")
            raw_q = normalized(raw_external[ei:ei+1])[0]
            if raw_q.shape != raw_bank.shape[1:]: raise ValueError("Raw ECG encoder dimensions differ")
            selections = {}
            for name, qv, bank in [("raw", raw_q, raw_bank), ("bridge_ecg", qvectors[qi], ecg_bank),
                                    ("trimodal", qvectors[qi], tri_bank)]:
                top, scores = rank(qv, bank, records, query["subject_id"])
                selections[name] = [make_anchor(records[i], sc, rank0+1)
                                    for rank0, (i,sc) in enumerate(zip(top,scores))]
                retrieval.append({"method": method, "sample_id": sid, "retriever": name,
                                  "top1_anchor": records[top[0]]["sample_id"],
                                  "top2_anchor": records[top[1]]["sample_id"], "top1_score": scores[0]})
            definitions = {"zero_shot": [], "raw_ecg_same": only_ecg(selections["raw"]),
                           "raw_ecg_full": selections["raw"],
                           "bridge_ecg_same": only_ecg(selections["bridge_ecg"]),
                           "trimodal_same": only_ecg(selections["trimodal"]),
                           "trimodal_full": selections["trimodal"]}
            for arm, evidence in definitions.items():
                # Zero-shot and raw retrieval are independent of bridge architecture.
                if method != methods[0] and arm in {"zero_shot", "raw_ecg_same", "raw_ecg_full"}: continue
                row = {"direction": "ecg", "sample_id": sid, "subject_id": query["subject_id"],
                       "query": query, "method": method, "condition": arm, "evidence": evidence,
                       "calibration_subset": sid in calibration_ids, "rendering": "legacy",
                       "global_patient_disjoint": True, "k": 0 if arm == "zero_shot" else 2}
                all_rows.append(row)
                if arm == "trimodal_full": full_rows.append(row)
        broken, pa = break_linkage([a for r in full_rows for a in r["evidence"]], seed)
        permutations.extend({"method": method, **p} for p in pa)
        for i, row in enumerate(full_rows):
            r = copy.deepcopy(row); r["condition"] = "trimodal_broken"; r["evidence"] = broken[2*i:2*i+2]
            all_rows.append(r)
    for row in all_rows: row["context_signature"] = sig(row)
    protocol = {"version": VERSION, "methods": list(methods), "seed": seed,
                "cohort_n": len(cohort), "render_subset_n": len(calibration_ids),
                "requested_max_queries": max_queries, "requested_render_subset": render_subset,
                "rows": len(all_rows), "arms": list(ARMS), "code_sha256": digest(__file__),
                "cohort_selection": "hash ordering of existing test cohort, without outcomes",
                "candidate_policy": "identical eligible linked Symile training admissions for all retrievers",
                "raw_ecg": "L2-normalized frozen foundation vectors; no trimodal training or text",
                "bridge_ecg": "ECG-to-ECG scoring after trimodal training; NOT an ECG-only learned encoder",
                "broken": "cross-patient permutation of CXR/lab bundles across selected slots; fixed ECG anchors",
                "input_sha256": {str(p.relative_to(source)): digest(p) for p in sorted(set(inputs))},
                "context_signature": sig(all_rows)}
    p = dest / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != protocol:
        raise ValueError("Frozen protocol/input changed; choose a new E4_REVIEW_ROOT")
    write_json(p, protocol)
    write_rows(dest / "contexts.jsonl", all_rows)
    write_csv(dest / "retrieval.csv", retrieval)
    write_csv(dest / "broken_linkage_audit.csv", permutations)
    write_rows(dest / "cohort.jsonl", [{"sample_id": r["sample_id"], "subject_id": r["subject_id"],
               "calibration_subset": str(r["sample_id"]) in calibration_ids} for r in cohort])
    return protocol


def verify_contexts(dest):
    dest = Path(dest); p = json.loads((dest / "protocol.json").read_text())
    rows = read_rows(dest / "contexts.jsonl")
    if sig(rows) != p["context_signature"]: raise ValueError("Frozen contexts changed")
    for r in rows:
        if sig({k:v for k,v in r.items() if k != "context_signature"}) != r["context_signature"]:
            raise ValueError("Invalid context signature")
    return rows


def calibrated_ecg(signal, fields, output, dpi=150):
    """Twelve full-length strips, 25 mm/s and 10 mm/mV in source pixel geometry.

    No per-lead amplitude normalization or clipping; grid and calibration pulse.
    Physical millimetres apply at the recorded DPI, not arbitrary screen zoom.
    The generator may subsequently resize/tile the image, which is audited separately.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.asarray(signal, dtype=float)
    fs = float(fields["fs"])
    if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all() or fs <= 0:
        raise ValueError("Finite 12-lead data and valid sample rate required")
    if len(x) < round(10*fs): raise ValueError("Ten seconds required; no padding or extrapolation")
    x = x[:round(10*fs)].copy()
    names = fields.get("sig_name", []); units = fields.get("units", [])
    expected = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if len(names) != 12 or {n.lower() for n in names} != {n.lower() for n in expected}:
        raise ValueError("Verified standard lead names required")
    if len(units) != 12: raise ValueError("WFDB physical units are required")
    factors = {"mv": 1., "uv": .001, "µv": .001, "μv": .001, "v": 1000.}
    for j, u in enumerate(units):
        if str(u).strip().lower() not in factors: raise ValueError(f"Unknown physical unit: {u}")
        x[:, j] *= factors[str(u).strip().lower()]
    order = [next(i for i,n in enumerate(names) if n.lower() == target.lower()) for target in expected]
    x = x[:, order]
    half_mv = max(1.5, np.ceil(np.max(np.abs(x)) * 2) / 2 + .25)
    row_mm = 2 * half_mv * 10
    width_mm, height_mm = 282., 12*row_mm + 22
    if height_mm > 1200: raise ValueError("Extreme amplitude needs manual waveform quality review")
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4), dpi=dpi)
    for j, name in enumerate(expected):
        left, bottom = 20/width_mm, (12+(11-j)*row_mm)/height_mm
        ax = fig.add_axes([left,bottom,250/width_mm,row_mm/height_mm])
        ax.set_xlim(0, 10); ax.set_ylim(-half_mv, half_mv)
        ax.set_xticks(np.arange(0,10.001,.2)); ax.set_xticks(np.arange(0,10.001,.04),minor=True)
        ax.set_yticks(np.arange(np.ceil(-half_mv/.5)*.5,half_mv,.5))
        ax.set_yticks(np.arange(np.ceil(-half_mv/.1)*.1,half_mv,.1),minor=True)
        ax.grid(which="minor",color="#f5dede",linewidth=.25)
        ax.grid(which="major",color="#debaba",linewidth=.5)
        ax.plot(np.arange(len(x))/fs,x[:,j],color="black",linewidth=.45)
        ax.tick_params(which="both",length=0,labelbottom=False,labelleft=False)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(-.65,0,name,fontsize=7,ha="right",va="center")
        # 1 mV pulse, 0.2 second plateau; outside the waveform region.
        ax.plot([-.5,-.45,-.45,-.25,-.25,-.2],[0,0,1,1,0,0],color="black",lw=.6,clip_on=False)
    fig.text(.07,1-6/height_mm,"25 mm/s   10 mm/mV   12 simultaneous leads   10 seconds",fontsize=8)
    fig.text(.07,4/height_mm,"1 mm grid; 1 mV calibration pulse. Scale applies at original print size.",fontsize=7)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    # A caller's matplotlib defaults must not silently crop the physical scale.
    with plt.rc_context({'savefig.bbox':None}):
        fig.savefig(output,dpi=dpi,facecolor="white")
    plt.close(fig)
    return {"dpi": dpi, "mm_per_second":25, "mm_per_mv":10, "duration_seconds":10,
            "pixels_per_second":25*dpi/25.4, "pixels_per_mv":10*dpi/25.4,
            "lead_names":expected,"lead_half_range_mv":float(half_mv),"source_units":units,
            "source_fs":fs,"source_pixel_geometry_only":True}


def prepare_images(source=None,dest=None):
    from e4_generation import _render_ecg
    import wfdb
    source0,dest0=roots();source,dest=Path(source or source0),Path(dest or dest0)
    rows=verify_contexts(dest); queries={r["sample_id"]:r for r in rows}
    audit=[]
    for sid,row in queries.items():
        stem=Path(row["query"]["waveform_path"])
        if stem.suffix in {".hea",".dat"}: stem=stem.with_suffix("")
        # Header plus all referenced WFDB signal files are bound to the image audit.
        header=wfdb.rdheader(str(stem))
        files=[stem.with_suffix('.hea')]+[stem.parent/n for n in sorted(set(header.file_name))]
        hashes={str(p):digest(p) for p in files}
        key=sig([sid,str(stem)])[:24]
        for rendering in (["legacy","calibrated"] if row["calibration_subset"] else ["legacy"]):
            path=dest/"query_images"/rendering/(key+".png")
            meta_path=path.with_suffix(".json")
            meta={"sample_id":sid,"rendering":rendering,"waveform_sha256":hashes,
                  "renderer_sha256":digest(__file__),
                  "legacy_renderer_sha256":digest(Path(__file__).with_name('e4_generation.py'))}
            if meta_path.exists():
                old=json.loads(meta_path.read_text())
                if any(old.get(k)!=v for k,v in meta.items()) or not path.is_file() or digest(path)!=old['image_sha256']:
                    raise ValueError("Image provenance changed; use a new review root")
                audit.append(old); continue
            if path.exists(): raise ValueError(f"Unverified image already exists: {path}")
            if rendering=="legacy": _render_ecg(stem,path)
            else:
                signal,fields=wfdb.rdsamp(str(stem));meta.update(calibrated_ecg(signal,fields,path))
            meta.update(image_path=str(path),image_sha256=digest(path))
            write_json(meta_path,meta);audit.append(meta)
    write_rows(dest/"image_audit.jsonl",audit)
    return {"images":len(audit),"quality_gate":"Inspect paired PNGs, then set E4_REVIEW_IMAGES_APPROVED=1"}


def generation_tasks(dest,model_key):
    rows=verify_contexts(dest)
    if model_key=="qwen35_4b": return rows
    if model_key!="pulse7b_ecg": raise ValueError("Prespecified generators are Qwen and PULSE")
    # Same fixed examinations and evidence under both renderings. No success-based selection.
    chosen=[r for r in rows if r["calibration_subset"] and r["condition"] in
            {"zero_shot","trimodal_same","trimodal_full"}]
    result=[]
    for row in chosen:
        for rendering in ("legacy","calibrated"):
            item=copy.deepcopy(row);item["rendering"]=rendering
            item["context_signature"]=sig({k:v for k,v in item.items() if k!='context_signature'})
            result.append(item)
    return result


def task_key(r):
    return sig([r['sample_id'],r['method'],r['condition'],r['rendering']])


def balanced_tasks(rows):
    """Interleave conditions/renderings so a small operational pilot covers each arm."""
    groups={}
    for row in rows:groups.setdefault((row['method'],row['condition'],row['rendering']),[]).append(row)
    for group in groups.values():group.sort(key=task_key)
    return [groups[k][i] for i in range(max((len(g) for g in groups.values()),default=0))
            for k in sorted(groups) if i<len(groups[k])]


def read_log(path,repair=False):
    """Recover only a torn last append, retaining its original bytes in a sidecar."""
    path=Path(path)
    if not path.exists(): return []
    raw=path.read_bytes(); lines=raw.splitlines(keepends=True); rows=[]; valid_bytes=0
    for i,line in enumerate(lines):
        try: row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if i!=len(lines)-1 or line.endswith(b'\n') or not repair: raise ValueError(f"Malformed JSONL: {path}:{i+1}")
            backup=path.with_name(path.name+'.torn-'+hashlib.sha256(raw).hexdigest()[:12])
            if not backup.exists(): backup.write_bytes(raw)
            with path.open('r+b') as f: f.truncate(valid_bytes)
            break
        rows.append(row);valid_bytes+=len(line)
    if repair and rows and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n')
    return rows


def run_generation(model_key,dest=None,max_rows=50,max_new_tokens=512,
                   max_input_tokens=None,component_cap=None):
    """Single-attempt, fsync-per-row inference. Restart skips successes AND errors.

    No old success is mixed into this experiment. Both sides of every comparison
    use the same newly pinned model and generation protocol.
    """
    import fcntl
    import torch
    from importlib.metadata import version
    from e4_model_registry import ALL_MODELS
    from e4_generation import build_prompt,_extract_clinical_output,_load_model
    from e4_generation_runtime import token_diagnostics,check_input_budget
    from PIL import Image
    dest=Path(dest or roots()[1]); tasks=generation_tasks(dest,model_key)
    if max_rows<0: raise ValueError("max_rows must be nonnegative; 0 means all remaining")
    if os.environ.get('E4_REVIEW_IMAGES_APPROVED')!='1':
        raise ValueError('Inspect legacy/calibrated images before setting E4_REVIEW_IMAGES_APPROVED=1')
    is_pulse=model_key=='pulse7b_ecg'
    max_input_tokens=max_input_tokens or (3584 if is_pulse else 12288)
    component_cap=component_cap or (120 if is_pulse else 1800)
    from e4_pulse_generation import build_pulse_prompt,parse_pulse_report,pulse_generate
    images={(r['sample_id'],r['rendering']):r for r in read_rows(dest/'image_audit.jsonl')}
    image_inputs={}
    for task in tasks:
        item=images[(task['sample_id'],task['rendering'])]
        if digest(item['image_path'])!=item['image_sha256']:raise ValueError('Image changed')
        image_inputs[task_key(task)]=item
    spec=ALL_MODELS[model_key]
    out=dest/'generations'/model_key;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_path=out/'protocol.json'
        settings={'version':VERSION,'model_key':model_key,'model_id':spec['model_id'],
                  'contexts':sig(tasks),'images':sig(image_inputs),'max_new_tokens':max_new_tokens,
                  'max_input_tokens':max_input_tokens,'component_cap':component_cap,
                  'code_hashes':{p.name:digest(p) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in
                        ('e4_generation.py','e4_ecg_specialists.py','e4_pulse_generation.py','e4_model_registry.py')]]},
                  'torch':torch.__version__,'transformers':version('transformers'),
                  'attempts_per_cell':1,'hidden_reasoning':'not requested or stored'}
        old=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
        if old and old['settings']!=settings: raise ValueError('Run settings changed; use a fresh review root')
        path=out/'reports.jsonl';prior=read_log(path,repair=True)
        if prior and old is None:raise ValueError('Reports without protocol')
        keys=[r['task_key'] for r in prior]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate completed tasks')
        if not set(keys).issubset({task_key(r) for r in tasks}):raise ValueError('Unexpected task in generation log')
        if any(r['protocol_signature']!=sig(old) for r in prior):raise ValueError('Mixed generation protocols')
        done=set(keys);pending=[r for r in tasks if task_key(r) not in done]
        print({'model':model_key,'planned':len(tasks),'attempted':len(prior),'pending':len(pending)},flush=True)
        if not pending:return {'status':'complete','attempted':len(prior)}
        from huggingface_hub import HfApi
        revision=old['resolved_revision'] if old else HfApi().model_info(spec['model_id']).sha
        protocol=old or {'settings':settings,'resolved_revision':revision}
        write_json(protocol_path,protocol)
        if is_pulse:
            from e4_ecg_specialists import _load_pulse,validate_specialist_environment
            validate_specialist_environment(model_key)
            runtime=_load_pulse({**spec,'revision':revision})
        else:
            model,processor,tokenizer,dtype=_load_model({**spec,'revision':revision})
            device=model.get_input_embeddings().weight.device
        for row in balanced_tasks(pending)[:max_rows or None]:
            im=image_inputs[task_key(row)];image_path=im['image_path']
            prompt=(build_pulse_prompt(row,component_cap) if is_pulse else
                    build_prompt(row,'raw_sensor_plus_retrieval',component_cap))
            result={k:row[k] for k in ('sample_id','subject_id','method','condition','rendering','context_signature')}
            result.update(task_key=task_key(row),model_key=model_key,protocol_signature=sig(protocol),
                          query_image=image_path,query_image_sha256=im['image_sha256'],prompt=prompt,
                          generated_report='',evidence_trace=[],status='error',error=None,
                          structured_schema_compliant=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                for dev in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(dev)
            start=time.perf_counter()
            try:
                if is_pulse:
                    text,diagnostics=pulse_generate(runtime,image_path,prompt,max_new_tokens,max_input_tokens)
                    parsed,stage,compliant,visible=parse_pulse_report(text)
                else:
                    with Image.open(image_path) as opened:image=opened.convert('RGB')
                    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
                    inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                         return_dict=True,return_tensors='pt',enable_thinking=False)
                    count=int(inp['input_ids'].shape[-1])
                    check_input_budget(count,max_input_tokens,max_new_tokens,model.config)
                    inp={k:(v.to(device=device,dtype=dtype) if v.is_floating_point() else v.to(device)) for k,v in inp.items()}
                    with torch.inference_mode():output=model.generate(**inp,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id)
                    generated=output[0,count:]
                    diagnostics=token_diagnostics(generated,max_new_tokens,getattr(model.generation_config,'eos_token_id',tokenizer.eos_token_id))
                    diagnostics['processor_input_tokens']=count
                    text=tokenizer.decode(generated,skip_special_tokens=False)
                    parsed,stage,compliant,visible=_extract_clinical_output(text,{a['anchor_id'] for a in row['evidence']})
                result.update(diagnostics,output_parse_stage=stage,visible_response=visible)
                if diagnostics.get('finish_reason')=='length':raise ValueError('Truncated output excluded')
                if parsed is None:raise ValueError('No usable visible report: '+stage)
                result.update(status='ok',generated_report=parsed['report'],evidence_trace=parsed.get('evidence_trace',[]),
                              uncertainties=parsed.get('uncertainties',[]),structured_schema_compliant=compliant)
            except Exception as exc:
                result['error']=repr(exc)
            if torch.cuda.is_available():torch.cuda.synchronize()
            result['wall_seconds_including_processing']=time.perf_counter()-start
            result['peak_allocated_bytes_by_device']=([int(torch.cuda.max_memory_allocated(d)) for d in
                range(torch.cuda.device_count())] if torch.cuda.is_available() else [])
            with path.open('a') as f:
                f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            print(model_key,row['sample_id'],row['condition'],row['rendering'],result['status'],flush=True)
            if result['error'] and ('out of memory' in result['error'].lower()):raise RuntimeError(result['error'])
        return {'status':'pilot_or_batch_finished','attempted':len(read_log(path)), 'planned':len(tasks)}


def label_metrics(truth,pred):
    truth,pred=np.asarray(truth,dtype=bool),np.asarray(pred,dtype=bool)
    tp=np.sum(truth & pred,axis=0);fp=np.sum(~truth & pred,axis=0);fn=np.sum(truth & ~pred,axis=0)
    row_fp=np.sum(~truth & pred,axis=1);row_fn=np.sum(truth & ~pred,axis=1)
    return {'micro_f1':float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1)),
            'macro_f1':float(np.mean(2*tp/np.maximum(2*tp+fp+fn,1))),
            'omission':float(np.mean(row_fn/np.maximum(truth.sum(1),1))),
            'reference_discordance':float(np.mean(row_fp/np.maximum(pred.sum(1),1))),
            'hamming_accuracy':float(np.mean(truth==pred)),
            'all_negative_hamming':float(np.mean(~truth))}


def paired_bootstrap(left,right,bootstraps=1000,seed=4411):
    """Report right minus left on the identical successful patient-cluster sample."""
    a={r['sample_id']:r for r in left};b={r['sample_id']:r for r in right}
    ids=sorted(set(a)&set(b))
    if not ids:return []
    for sid in ids:
        if a[sid]['subject_id']!=b[sid]['subject_id'] or a[sid]['y_true']!=b[sid]['y_true']:
            raise ValueError('Paired identity/reference mismatch')
    groups={}
    for i,sid in enumerate(ids):groups.setdefault(str(a[sid]['subject_id']),[]).append(i)
    patients=sorted(groups);truth=np.array([a[s]['y_true'] for s in ids]);pa=np.array([a[s]['y_pred'] for s in ids]);pb=np.array([b[s]['y_pred'] for s in ids])
    ma,mb=label_metrics(truth,pa),label_metrics(truth,pb);rng=np.random.default_rng(seed)
    draws={k:[] for k in ('micro_f1','omission','reference_discordance')}
    if len(patients)>1:
        for _ in range(bootstraps):
            idx=np.concatenate([groups[p] for p in rng.choice(patients,len(patients),replace=True)])
            aa,bb=label_metrics(truth[idx],pa[idx]),label_metrics(truth[idx],pb[idx])
            for k in draws:draws[k].append(bb[k]-aa[k])
    return [{'metric':k,'paired_n':len(ids),'patients':len(patients),'left':ma[k],'right':mb[k],
             'delta_right_minus_left':mb[k]-ma[k],
             'ci_low':float(np.quantile(v,.025)) if v else None,
             'ci_high':float(np.quantile(v,.975)) if v else None,
             'exploratory_unadjusted':True} for k,v in draws.items()]


def evaluate(dest=None,bootstraps=1000,allow_partial=False):
    """No GPU needed. Copy baselines are available immediately after notebook 17."""
    from e4_evaluation import present,ECG_RULES,ECG_LABELS
    dest=Path(dest or roots()[1]);contexts=verify_contexts(dest)
    def score(row,report,model_key):
        truth=present(row['query'].get('machine_diagnosis',''),ECG_RULES);pred=present(report,ECG_RULES)
        return {**{k:row[k] for k in ('sample_id','subject_id','method','condition','rendering')},
                'model_key':model_key,'y_true':[int(x in truth) for x in ECG_LABELS],
                'y_pred':[int(x in pred) for x in ECG_LABELS]}
    details=[]
    for row in contexts:
        if row['condition'] in {'raw_ecg_same','bridge_ecg_same','trimodal_same'}:
            details.append(score(row,row['evidence'][0]['ecg']['machine_diagnosis'],'nearest_neighbor_copy'))
    coverage=[]; resources=[]; missing_models=[]; abstention_sensitivity=[]
    for model in ('qwen35_4b','pulse7b_ecg'):
        tasks=generation_tasks(dest,model);lookup={task_key(r):r for r in tasks}
        path=dest/'generations'/model/'reports.jsonl'
        existing=read_log(path) if path.exists() else []
        if len({r['task_key'] for r in existing})!=len(existing):raise ValueError('Duplicate result keys')
        protocol_path=path.with_name('protocol.json')
        if existing:
            protocol=json.loads(protocol_path.read_text())
            if protocol['settings']['contexts']!=sig(tasks):raise ValueError('Result protocol/context mismatch')
        for r in existing:
            if r['task_key'] not in lookup or r['protocol_signature']!=sig(protocol):raise ValueError('Unexpected/stale result')
            row=lookup[r['task_key']]
            if r['context_signature']!=row['context_signature']:raise ValueError('Stale context signature')
            if r['status']=='ok':details.append(score(row,r['generated_report'],model))
            abstention_sensitivity.append(score(row,r['generated_report'] if r['status']=='ok' else '',model))
            resources.append({k:r.get(k) for k in ('model_key','sample_id','method','condition','rendering','status',
                             'wall_seconds_including_processing','processor_input_tokens','generated_token_count')})
            resources[-1]['peak_allocated_bytes_by_device_json']=json.dumps(r.get('peak_allocated_bytes_by_device',[]))
        counts={}
        for row in tasks:counts.setdefault((row['method'],row['condition'],row['rendering']),[0,0,0])[0]+=1
        for r in existing:
            c=counts[(r['method'],r['condition'],r['rendering'])];c[1]+=1;c[2]+=r['status']=='ok'
        for key,(n,attempted,ok) in counts.items():
            coverage.append(dict(model_key=model,method=key[0],condition=key[1],rendering=key[2],
                                 planned=n,attempted=attempted,success=ok,failed=attempted-ok,pending=n-attempted))
        if len(existing)<len(tasks):missing_models.append(model)
    groups={}
    for r in details:groups.setdefault((r['model_key'],r['method'],r['condition'],r['rendering']),[]).append(r)
    summary=[]
    for key,group in groups.items():
        summary.append(dict(model_key=key[0],method=key[1],condition=key[2],rendering=key[3],n=len(group),
                       **label_metrics([r['y_true'] for r in group],[r['y_pred'] for r in group])))
    comparisons=[];methods=json.loads((dest/'protocol.json').read_text())['methods'];first=methods[0]
    contrasts=[('raw_ecg_same','trimodal_same'),('bridge_ecg_same','trimodal_same'),
               ('trimodal_same','trimodal_full'),('trimodal_broken','trimodal_full'),
               ('raw_ecg_full','trimodal_full'),('zero_shot','trimodal_full')]
    for model in ('qwen35_4b','pulse7b_ecg'):
        for method in methods:
            for rendering in ('legacy','calibrated'):
                for left,right in contrasts:
                    lm=first if left in {'raw_ecg_same','raw_ecg_full','zero_shot'} else method
                    a=groups.get((model,lm,left,rendering),[]);b=groups.get((model,method,right,rendering),[])
                    for item in paired_bootstrap(a,b,bootstraps):
                        comparisons.append(dict(model_key=model,method=method,rendering=rendering,
                                          contrast=right+' minus '+left,**item))
                # Generation must beat direct copying on the SAME cases, not a different denominator.
                for arm in ('raw_ecg_same','bridge_ecg_same','trimodal_same'):
                    for item in paired_bootstrap(groups.get(('nearest_neighbor_copy',method,arm,'legacy'),[]),
                                                 groups.get((model,method,arm,rendering),[]),bootstraps):
                        comparisons.append(dict(model_key=model,method=method,rendering=rendering,
                                             contrast=arm+' generation minus top1_copy',**item))
            for arm in ('zero_shot','trimodal_same','trimodal_full'):
                for item in paired_bootstrap(groups.get((model,method,arm,'legacy'),[]),
                                             groups.get((model,method,arm,'calibrated'),[]),bootstraps):
                    comparisons.append(dict(model_key=model,method=method,rendering='paired_renderings',
                                             contrast=arm+' calibrated minus legacy',**item))
    sensitivity_groups={}
    for r in abstention_sensitivity:
        sensitivity_groups.setdefault((r['model_key'],r['method'],r['condition'],r['rendering']),[]).append(r)
    sensitivity=[]
    for key,group in sensitivity_groups.items():
        sensitivity.append(dict(model_key=key[0],method=key[1],condition=key[2],rendering=key[3],
                                attempted_n=len(group),policy='failed attempts treated as empty-label abstention; not a quality bound',
                                **label_metrics([r['y_true'] for r in group],[r['y_pred'] for r in group])))
    out=dest/'evaluation';write_rows(out/'detailed.jsonl',details)
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_comparisons.csv',comparisons)
    write_csv(out/'coverage.csv',coverage);write_csv(out/'resources.csv',resources)
    write_csv(out/'failure_as_abstention_sensitivity.csv',sensitivity)
    status={'status':'partial_do_not_use_as_final' if missing_models else 'complete',
            'missing_models':missing_models,'reference':'machine interpretations, not expert ground truth',
            'bootstrap_replicates':bootstraps,'seed':4411,'conditioning':'common successful cases',
            'warnings':['Copying tests reference transfer, not clinically valid reporting.',
                        'Do not interpret unadjusted exploratory intervals as confirmatory.',
                        'Timing excludes loading and is not an energy or end-to-end cost measurement.',
                        'Failure coverage must be reported beside complete-case estimates.']}
    write_json(out/'status.json',status)
    if missing_models and not allow_partial:
        raise RuntimeError('Generation incomplete; coverage and clearly marked exploratory files saved. Use allow_partial=True only for pilots/copy baselines.')
    return status
