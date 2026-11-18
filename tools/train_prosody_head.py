"""Train the tone-of-voice -> mood head (`models/prosody-head/mood_head.npz`) on CREMA-D.

The head reads the 12 prosody numbers `duet.prosody.utterance_features` computes at runtime (pitch and level relative to
the caller's own `Baseline`) and outputs a softmax over emotion classes; `MoodReader` turns that into expected arousal and
valence with the per-class values stored next to the weights. Network: 12 -> 16 (ReLU) -> k, numpy-only at runtime.

Data: CREMA-D (Cao et al. 2014), 7,442 acted clips, 91 actors, 12 sentences, 6 emotions, ODbL 1.0 (contents DbCL 1.0).
Only CREMA-D is used. Raw audio lives on D: (`--data`), not in the project.

Steps (run from the project root so `import duet` works):
    python tools/train_prosody_head.py download   # manifest from a no-checkout clone + sha256-verified LFS downloads
    python tools/train_prosody_head.py features   # per-clip features, runtime-style baselines, noisy sessions -> cache
    python tools/train_prosody_head.py train      # actor-grouped 5-fold CV: 6 vs 4 classes, export, MoodReader check
    python tools/train_prosody_head.py train --publish   # same, and write models/prosody-head/mood_head[_vN].npz
    python tools/train_prosody_head.py all

`train` writes results.json and fold/final heads under `--data`; only `--publish` writes into the project, and never
over an existing file. The published head (2026-11-03) is `mood_head_v2.npz`: 6 classes, duration_s ignored.

Download by hand instead (same result): `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1
https://github.com/CheyneyComputerScience/CREMA-D` then `git lfs pull --include "AudioWAV/*"`, and point `--data` at the
folder that holds `AudioWAV/`. Checking out ~30k pointer files is very slow on Windows, hence `--no-checkout` + the
media URLs below.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from duet.config import RATE  # noqa: E402
from duet.prosody import FEATURES, Baseline, MoodReader, _prior, utterance_features  # noqa: E402

DATA = Path("D:/duet-data/crema-d")
REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git"
MEDIA_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/"
OUT_DIR = ROOT / "models" / "prosody-head"

EMO = ("NEU", "HAP", "SAD", "ANG", "FEA", "DIS")
NAME = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful", "DIS": "disgust"}
AROUSAL = {"ANG": 0.8, "FEA": 0.6, "DIS": 0.2, "HAP": 0.6, "NEU": 0.0, "SAD": -0.6}
VALENCE = {"ANG": -0.6, "FEA": -0.7, "DIS": -0.6, "HAP": 0.7, "NEU": 0.0, "SAD": -0.6}
# candidate class sets: emotion code -> class name (emotions left out are not trained on)
SETS = {
    "6": {e: NAME[e] for e in ("NEU", "HAP", "SAD", "ANG", "FEA", "DIS")},
    "4": {"NEU": "neutral", "HAP": "happy", "ANG": "angry", "SAD": "sad"},
    "4m": {"NEU": "neutral", "HAP": "happy", "ANG": "angry", "DIS": "angry", "SAD": "sad"},
}
GATE = 0.05  # MoodReader.read answers "no reading" below this voiced share; the head never sees those turns
AROUSAL_T = 0.2  # expected arousal above this = "high" for the binary arousal score
NOISY_COPIES = 2
MASKS = {"none": (), "dur": ("duration_s",), "dur+f0abs": ("duration_s", "f0_st_abs")}
ALPHAS = (1e-3, 1e-2, 1e-1)
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- download

def _git(repo, *args, stdin=None):
    cmd = ["git", "-c", "safe.directory=*", "-C", str(repo), *args]
    return subprocess.run(cmd, input=stdin, capture_output=True, check=True).stdout


def manifest(data):
    """[(relpath, sha256, size)] for AudioWAV, read from the LFS pointers of a shallow no-checkout clone."""
    repo = data / "repo"
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", REPO_URL, str(repo)], check=True)
    tree = _git(repo, "ls-tree", "HEAD", "AudioWAV/").decode().splitlines()
    names = [line.split("\t")[1] for line in tree]
    shas = "\n".join(line.split()[2] for line in tree).encode() + b"\n"
    blobs = _git(repo, "cat-file", "--batch", stdin=shas).decode()
    oids = [ln.split("sha256:")[1] for ln in blobs.splitlines() if ln.startswith("oid sha256:")]
    sizes = [int(ln.split()[1]) for ln in blobs.splitlines() if ln.startswith("size ")]
    assert len(names) == len(oids) == len(sizes), (len(names), len(oids), len(sizes))
    (data / "LICENSE.txt").write_bytes(_git(repo, "show", "HEAD:LICENSE.txt"))
    (data / "VideoDemographics.csv").write_bytes(_git(repo, "show", "HEAD:VideoDemographics.csv"))
    return list(zip(names, oids, sizes))


def _fetch(item, dest):
    rel, oid, size = item
    path = dest / Path(rel).name
    if path.exists() and path.stat().st_size == size:
        return "kept"
    for _ in range(4):
        try:
            with urllib.request.urlopen(MEDIA_URL + rel, timeout=60) as r:
                blob = r.read()
            if len(blob) == size and hashlib.sha256(blob).hexdigest() == oid:
                path.write_bytes(blob)
                return "ok"
        except OSError:
            pass
    return f"FAILED {rel}"


def download(data):
    items = manifest(data)
    dest = data / "AudioWAV"
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        res = list(pool.map(lambda it: _fetch(it, dest), items))
    bad = [r for r in res if r.startswith("FAILED")]
    print(f"{len(items)} files, {res.count('ok')} downloaded, {res.count('kept')} already there, {len(bad)} failed")
    for b in bad[:20]:
        print(" ", b)


# ---------------------------------------------------------------- features

def clips(data):
    """[(path, actor, sentence, emotion, level)] from `<actor>_<sentence>_<emotion>_<level>.wav`."""
    rows = []
    for f in sorted((data / "AudioWAV").glob("*.wav")):
        a, s, e, lv = f.stem.split("_")
        assert a.isdigit() and len(s) == 3 and e in EMO and lv in ("LO", "MD", "HI", "XX", "X"), f.name
        rows.append((f, int(a), s, e, lv))
    return rows


def load(path):
    """16 kHz mono float32 in [-1, 1]."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != RATE:
        from scipy.signal import resample_poly
        g = gcd(sr, RATE)
        x = resample_poly(x, RATE // g, sr // g).astype(np.float32)
    return x


def _pink(n, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    y = np.fft.irfft(spec / np.sqrt(f), n)
    return y / (y.std() + 1e-12)


def channel(x, ch, rng):
    """Add white/pink noise at an absolute RMS (dBFS), then a gain; clip like a real ADC would."""
    if ch is None:
        return x
    n = len(x)
    if ch.get("noise_db") is not None:
        noise = rng.standard_normal(n) if ch["color"] == "white" else _pink(n, rng)
        x = x + noise * 10 ** (ch["noise_db"] / 20)
    return np.clip(x * 10 ** (ch.get("gain_db", 0.0) / 20), -1, 1).astype(np.float32)


def _filled(f):
    """Did utterance_features get past its voicing checks (so f0/level carry a value)?"""
    return f["hf_ratio"] != 0.0 or f["level_range_db"] != 0.0 or f["f0_st_abs"] != 0.0


def rebase(raw, b):
    """Features of a clip relative to baseline `b`, from its baseline-free features: only f0_st and level_db change.
    Equals utterance_features(pcm, b) (checked in `features`), at no extraction cost."""
    f = dict(raw)
    if _filled(raw):
        f["f0_st"] = raw["f0_st_abs"] - (b.f0_st if b.n else _prior(raw["f0_st_abs"]))
        f["level_db"] = (raw["level_db"] - 26.0) - (b.level_db if b.n else -26.0)
    return f


def _raw(path):
    with np.errstate(divide="ignore", invalid="ignore"):  # prosody.track's np.where divides in the unused branch
        return utterance_features(load(path))


def _session(task):
    """One simulated call: the caller's earlier turns build the Baseline, exactly as at runtime, then the target turn
    is measured against it. `hist_ch` applies to the earlier turns, `tgt_ch` to the target turn."""
    paths, hist_ch, tgt_ch, seed = task
    rng = np.random.default_rng(seed)
    b = Baseline()
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in paths[:-1]:
            b.update(utterance_features(channel(load(p), hist_ch, rng), b))
        f = utterance_features(channel(load(paths[-1]), tgt_ch, rng), b)
    return _vec(f) + [b.n]


def _vec(f):
    return [float(f[k]) for k in FEATURES]


def features(data, workers):
    rows = clips(data)
    n = len(rows)
    paths = [str(r[0]) for r in rows]
    actor = np.array([r[1] for r in rows])
    print(f"{n} clips, {len(set(actor))} actors, {len({r[2] for r in rows})} sentences, "
          + ", ".join(f"{e} {sum(r[3] == e for r in rows)}" for e in EMO))
    with Pool(workers) as pool:
        raw = pool.map(_raw, paths, chunksize=32)
        rng = np.random.default_rng(7)
        by_actor = {a: np.nonzero(actor == a)[0] for a in np.unique(actor)}

        def history(i):
            others = by_actor[actor[i]][by_actor[actor[i]] != i]
            return rng.choice(others, int(rng.integers(3, 7)), replace=False)  # 3-6 other clips, random order

        # clean: baseline from 3-6 other clips of the same actor (any emotion), never the clip itself
        hist = [history(i) for i in range(n)]
        clean, clean_n = np.zeros((n, len(FEATURES))), np.zeros(n, int)
        for i in range(n):
            b = Baseline()
            for j in hist[i]:
                b.update(rebase(raw[j], b))
            clean[i], clean_n[i] = _vec(rebase(raw[i], b)), b.n
        # proof that rebase() == the runtime path (re-extract every turn against the running baseline)
        check = rng.choice(n, 200, replace=False)
        ref = np.array(pool.map(_session, [([paths[j] for j in hist[i]] + [paths[i]], None, None, 0)
                                           for i in check]))
        err = np.abs(ref[:, :-1] - clean[check]).max()
        print(f"rebase vs runtime path on 200 clips: max abs diff {err:.2e}")
        assert err < 1e-6 and (ref[:, -1] == clean_n[check]).all()

        # the caller's typical speech level (mean voiced-frame dB), to set a fixed noise floor per simulated call
        lvl = np.array([r["level_db"] - 26.0 if _filled(r) else np.nan for r in raw])
        ref_db = {a: float(np.nanmedian(lvl[idx])) for a, idx in by_actor.items()}

        # noisy calls (training copies; copy 0 of the test actors is also the noisy test condition): one noise floor
        # for the whole call (white or pink, 20-30 dB below this caller's typical speech level, so quiet turns get a
        # worse SNR, as on a real line), plus a -6..+6 dB gain on the target turn alone (mic distance, phone AGC)
        tasks, meta = [], []
        for k in range(NOISY_COPIES):
            for i in range(n):
                h = history(i)
                color, snr, g = str(rng.choice(["white", "pink"])), float(rng.uniform(20, 30)), float(rng.uniform(-6, 6))
                noise = {"color": color, "noise_db": ref_db[actor[i]] - snr}
                tasks.append(([paths[j] for j in h] + [paths[i]], noise, {**noise, "gain_db": g},
                              int(rng.integers(1 << 31))))
                meta.append((k, i, color, snr, g))
        # per-turn gain only (test of level drift): clean baseline, target alone shifted by -6..+6 dB
        for i in range(n):
            g = float(rng.uniform(-6, 6))
            tasks.append(([paths[j] for j in hist[i]] + [paths[i]], None, {"gain_db": g}, 0))
            meta.append((-1, i, "gain", 0.0, g))
        out = np.array(pool.map(_session, tasks, chunksize=16))
    noisy = out[:NOISY_COPIES * n, :-1].reshape(NOISY_COPIES, n, -1)
    noisy_n = out[:NOISY_COPIES * n, -1].reshape(NOISY_COPIES, n).astype(int)
    gain = out[NOISY_COPIES * n:, :-1]
    np.savez(data / "features.npz", clean=clean, clean_n=clean_n, noisy=noisy, noisy_n=noisy_n, turn_gain=gain,
             raw=np.array([_vec(r) for r in raw]), actor=actor, emotion=np.array([r[3] for r in rows]),
             sentence=np.array([r[2] for r in rows]), level=np.array([r[4] for r in rows]),
             name=np.array([Path(p).name for p in paths]), meta=np.array(meta, dtype=object).astype(str))
    print(f"wrote {data / 'features.npz'}")


# ---------------------------------------------------------------- train

VS = FEATURES.index("voiced_share")
HIGH, LOW = ("ANG", "FEA", "HAP"), ("NEU", "SAD")  # binary arousal truth; DIS reported separately


def classes_of(set_name):
    return list(dict.fromkeys(SETS[set_name].values()))


def class_values(set_name, emo):
    """Per-class arousal/valence: the emotion values, clip-count weighted when a class merges emotions."""
    ar, va = [], []
    for c in classes_of(set_name):
        es = [e for e, cc in SETS[set_name].items() if cc == c]
        w = np.array([(emo == e).sum() for e in es], float)
        ar.append(w @ [AROUSAL[e] for e in es] / w.sum())
        va.append(w @ [VALENCE[e] for e in es] / w.sum())
    return np.array(ar), np.array(va)


def _std(X, mean, std, mask):
    Z = (X - mean) / std
    Z[:, mask] = 0.0  # masked inputs: zero rows in w1 at export, so the runtime value is ignored
    return Z


def _fit(Z, y, alpha, seed):
    import warnings
    from sklearn.neural_network import MLPClassifier
    k = int(y.max()) + 1
    w = (len(y) / (k * np.bincount(y, minlength=k)))[y]  # class-balanced
    m = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", alpha=alpha, batch_size=256,
                      learning_rate_init=2e-3, max_iter=300, tol=1e-5, n_iter_no_change=15, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Z, y, sample_weight=w)
    return m


def _train_rows(F, idx, y):
    """Clean + noisy copies of the training clips that the head would see (voiced share >= GATE)."""
    X = np.concatenate([F["clean"][idx]] + [F["noisy"][k][idx] for k in range(NOISY_COPIES)])
    Y = np.tile(y[idx], 1 + NOISY_COPIES)
    keep = (Y >= 0) & (X[:, VS] >= GATE)
    return X[keep], Y[keep]


def fit_head(F, idx, set_name, mask_name, alpha, workers=1):
    classes = classes_of(set_name)
    y = np.array([classes.index(SETS[set_name][e]) if e in SETS[set_name] else -1 for e in F["emotion"]])
    X, Y = _train_rows(F, idx, y)
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    mask = [FEATURES.index(f) for f in MASKS[mask_name]]
    Z = _std(X, mean, std, mask)
    from joblib import Parallel, delayed
    models = Parallel(n_jobs=min(workers, len(SEEDS)))(delayed(_fit)(Z, Y, alpha, s) for s in SEEDS)
    m = min(models, key=lambda mm: mm.loss_)  # best of 3 restarts by training loss
    return {"set": set_name, "mask": mask, "mean": mean, "std": std, "model": m, "classes": classes,
            "arousal": class_values(set_name, F["emotion"])[0], "valence": class_values(set_name, F["emotion"])[1]}


def predict(head, X):
    return head["model"].predict_proba(_std(X, head["mean"], head["std"], head["mask"]))


def _bal(y, p, k):
    rec = [np.mean(p[y == c] == c) for c in range(k) if (y == c).any()]
    return float(np.mean(rec))


def collapse(head, P, target):
    """Probabilities of `head` regrouped onto the classes of set `target` (classes it lacks get no mass)."""
    tcls = classes_of(target)
    Q = np.zeros((len(P), len(tcls)))
    for ci, c in enumerate(head["classes"]):
        members = [e for e, cc in SETS[head["set"]].items() if cc == c and e in SETS[target]]
        if members:
            Q[:, tcls.index(SETS[target][members[0]])] += P[:, ci]
    return Q


def evaluate(head, F, idx, cond="clean"):
    X = F["clean"][idx] if cond == "clean" else F["noisy"][0][idx] if cond == "noisy" else F["turn_gain"][idx]
    emo = F["emotion"][idx]
    seen = X[:, VS] >= GATE
    P = predict(head, X[seen])
    e = emo[seen]
    out = {"gated": float(1 - seen.mean()), "gated_by_emotion": {k: float(1 - seen[emo == k].mean()) for k in EMO}}
    # class accuracy on its own set and on the 4 / 4m subsets (same clips for every model)
    for target in ("own", "4", "4m"):  # "4": NEU/HAP/ANG/SAD clips; "4m": the same plus DIS clips scored as angry
        t = head["set"] if target == "own" else target
        tcls = classes_of(t)
        rows = np.array([x in SETS[t] for x in e])
        yt = np.array([tcls.index(SETS[t][x]) for x in e[rows]])
        Q = P[rows] if target == "own" else collapse(head, P[rows], t)
        pred = Q.argmax(1)
        out[f"bal_acc_{target}"] = _bal(yt, pred, Q.shape[1])
        if target == "own":
            k = len(tcls)
            out["confusion"] = np.array([[int(np.sum((yt == a) & (pred == b))) for b in range(k)]
                                         for a in range(k)]).tolist()
    # arousal from the expected value the runtime uses
    A = P @ head["arousal"]
    from sklearn.metrics import roc_auc_score
    hi, lo = np.isin(e, HIGH), np.isin(e, LOW)
    out["arousal_bal_acc"] = float((np.mean(A[hi] > AROUSAL_T) + np.mean(A[lo] <= AROUSAL_T)) / 2)
    out["arousal_auc"] = float(roc_auc_score(hi[hi | lo], A[hi | lo]))
    dis = e == "DIS"
    out["arousal_bal_acc_dis_low"] = float((np.mean(A[hi] > AROUSAL_T) + np.mean(A[lo | dis] <= AROUSAL_T)) / 2)
    out["dis_called_high"] = float(np.mean(A[dis] > AROUSAL_T))
    out["mean_arousal_by_emotion"] = {k: float(A[e == k].mean()) for k in EMO}
    out["mean_valence_by_emotion"] = {k: float((P @ head["valence"])[e == k].mean()) for k in EMO}
    # end to end: gated turns read as arousal 0 (= low)
    Aall = np.zeros(len(idx))
    Aall[seen] = A
    hi_all, lo_all = np.isin(emo, HIGH), np.isin(emo, LOW)
    out["arousal_bal_acc_end_to_end"] = float((np.mean(Aall[hi_all] > AROUSAL_T)
                                               + np.mean(Aall[lo_all] <= AROUSAL_T)) / 2)
    return out


def export(head, path):
    m = head["model"]
    w1 = m.coefs_[0].copy()
    w1[head["mask"]] = 0.0
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    np.savez(path, mean=f32(head["mean"]), std=f32(head["std"]), w1=f32(w1), b1=f32(m.intercepts_[0]),
             w2=f32(m.coefs_[1]), b2=f32(m.intercepts_[1]), labels=np.array(head["classes"], dtype="<U8"),
             arousal=f32(head["arousal"]), valence=f32(head["valence"]))


def agreement(head, path, X):
    """Load `path` with the runtime MoodReader and compare with the trained model on rows X."""
    reader = MoodReader(Path(path))
    seen = X[:, VS] >= GATE
    P = predict(head, X[seen])
    ours = [head["classes"][i] for i in P.argmax(1)]
    readings = [reader.read(dict(zip(FEATURES, row))) for row in X]
    theirs = [r.label for r, s in zip(readings, seen) if s]
    a_ours = np.clip(P @ head["arousal"], -1, 1)
    a_theirs = np.array([r.arousal for r, s in zip(readings, seen) if s])
    gated_ok = all(r.label == "" for r, s in zip(readings, seen) if not s)
    return {"label_agree": float(np.mean([a == b for a, b in zip(ours, theirs)])), "n": int(seen.sum()),
            "max_arousal_diff": float(np.abs(a_ours - a_theirs).max()), "gated_rows_empty": gated_ok}


def _summ(folds, key):
    v = np.array([f[key] for f in folds])
    return float(v.mean()), float(v.std())


def fresh(path):
    """Never overwrite an earlier head: mood_head.npz, then mood_head_v2.npz, ..."""
    k = 1
    p = path
    while p.exists():
        k += 1
        p = path.with_name(f"{path.stem}_v{k}{path.suffix}")
    return p


def train(data, workers, final_set, publish):
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold
    F = dict(np.load(data / "features.npz"))
    n = len(F["emotion"])
    folds = list(GroupKFold(5).split(np.zeros(n), groups=F["actor"]))
    res = {"n_clips": n, "n_actors": int(len(np.unique(F["actor"]))),
           "baseline_empty_clean": float(np.mean(F["clean_n"] == 0)),
           "baseline_n_hist": np.bincount(F["clean_n"]).tolist()}
    seen = F["clean"][:, VS] >= GATE
    res["feature_means_by_emotion"] = {e: dict(zip(FEATURES, F["clean"][seen & (F["emotion"] == e)].mean(0).round(3)
                                                   .tolist())) for e in EMO}
    res["voiced_share_pct"] = np.percentile(F["clean"][:, VS], [10, 25, 50, 75, 90]).round(3).tolist()

    # 1) model selection on the 6-class problem: input masks x L2 strength
    grid = [(mk, a) for mk in MASKS for a in ALPHAS]
    jobs = Parallel(n_jobs=workers)(delayed(fit_head)(F, tr, "6", mk, a) for mk, a in grid for tr, _ in folds)
    sel = {}
    for gi, (mk, a) in enumerate(grid):
        ev = [evaluate(jobs[gi * 5 + f], F, folds[f][1]) for f in range(5)]
        sel[f"{mk}|{a}"] = {"bal_acc_own": _summ(ev, "bal_acc_own"), "arousal_bal_acc": _summ(ev, "arousal_bal_acc")}
        print(f"6-class mask={mk:9s} alpha={a:<6} bal_acc {sel[f'{mk}|{a}']['bal_acc_own'][0]:.3f} "
              f"± {sel[f'{mk}|{a}']['bal_acc_own'][1]:.3f}  arousal {sel[f'{mk}|{a}']['arousal_bal_acc'][0]:.3f}")
    res["selection_6class"] = sel
    # duration_s is dropped by design: CREMA-D clips are 12 fixed sentences of 1.6-4 s, so the head can only learn
    # "longer than this sentence usually is"; at runtime the length is mostly what was said. Probe: give the
    # full-input heads runtime-like lengths and count how many labels flip.
    a_none = max(ALPHAS, key=lambda a: sel[f"none|{a}"]["bal_acc_own"][0])
    gi = grid.index(("none", a_none))
    probe = {}
    for d in (0.6, 1.0, 4.0, 6.0, 10.0):
        flips = []
        for f in range(5):
            X = F["clean"][folds[f][1]]
            X = X[X[:, VS] >= GATE]
            Y = X.copy()
            Y[:, FEATURES.index("duration_s")] = d
            flips.append(np.mean(predict(jobs[gi * 5 + f], X).argmax(1) != predict(jobs[gi * 5 + f], Y).argmax(1)))
        probe[str(d)] = float(np.mean(flips))
    res["duration_probe_label_flips"] = probe
    print("full-input head, share of labels that flip when duration_s is set to:", probe)
    mask_name = "dur"
    alpha = max(ALPHAS, key=lambda a: sel[f"{mask_name}|{a}"]["bal_acc_own"][0])
    res["chosen"] = {"mask": mask_name, "alpha": alpha}
    print(f"chosen inputs mask={mask_name} alpha={alpha}")

    # 2) class sets with the chosen config, every condition
    heads = {s: Parallel(n_jobs=workers)(delayed(fit_head)(F, tr, s, mask_name, alpha) for tr, _ in folds)
             for s in SETS}
    res["sets"] = {}
    (data / "heads").mkdir(exist_ok=True)
    for s in SETS:
        per = {c: [evaluate(heads[s][f], F, folds[f][1], c) for f in range(5)] for c in ("clean", "noisy", "gain")}
        keys = ["bal_acc_own", "bal_acc_4", "bal_acc_4m", "arousal_bal_acc", "arousal_auc", "arousal_bal_acc_dis_low",
                "dis_called_high", "arousal_bal_acc_end_to_end", "gated"]
        r = {c: {k: _summ(per[c], k) for k in keys} for c in per}
        r["confusion_clean"] = np.sum([np.array(p["confusion"]) for p in per["clean"]], axis=0).tolist()
        r["classes"] = classes_of(s)
        r["mean_arousal_by_emotion"] = {e: float(np.mean([p["mean_arousal_by_emotion"][e] for p in per["clean"]]))
                                        for e in EMO}
        r["mean_valence_by_emotion"] = {e: float(np.mean([p["mean_valence_by_emotion"][e] for p in per["clean"]]))
                                        for e in EMO}
        r["gated_by_emotion"] = {e: float(np.mean([p["gated_by_emotion"][e] for p in per["clean"]])) for e in EMO}
        # the exported weights through MoodReader must give the same answers on every test fold
        agree = []
        for f in range(5):
            p = data / "heads" / f"fold{f}_{s}.npz"
            export(heads[s][f], p)
            agree.append(agreement(heads[s][f], p, F["clean"][folds[f][1]]))
        r["moodreader_agreement"] = agree
        # do the data argue for other per-class values? least-squares readout on the out-of-fold probabilities
        P, E = [], []
        for f in range(5):
            X = F["clean"][folds[f][1]]
            keep = X[:, VS] >= GATE
            P.append(predict(heads[s][f], X[keep]))
            E.append(F["emotion"][folds[f][1]][keep])
        P, E = np.concatenate(P), np.concatenate(E)
        for name, table in (("arousal", AROUSAL), ("valence", VALENCE)):
            t = np.array([table[e] for e in E])
            given = heads[s][0][name]
            fit = np.linalg.lstsq(P, t, rcond=None)[0]
            r[f"readout_{name}"] = {"given": given.tolist(), "lstsq": fit.round(3).tolist(),
                                    "mae_given": float(np.abs(np.clip(P @ given, -1, 1) - t).mean()),
                                    "mae_lstsq": float(np.abs(np.clip(P @ fit, -1, 1) - t).mean()),
                                    "corr_given": float(np.corrcoef(P @ given, t)[0, 1])}
        res["sets"][s] = r
        c = r["clean"]
        print(f"set {s:3s} own {c['bal_acc_own'][0]:.3f}±{c['bal_acc_own'][1]:.3f}  on4 {c['bal_acc_4'][0]:.3f}  "
              f"on4m {c['bal_acc_4m'][0]:.3f}  arousal {c['arousal_bal_acc'][0]:.3f}±{c['arousal_bal_acc'][1]:.3f} "
              f"auc {c['arousal_auc'][0]:.3f}  noisy own {r['noisy']['bal_acc_own'][0]:.3f}  "
              f"agree {min(a['label_agree'] for a in agree):.4f}")

    # 3) final head on every actor (to D: always; into models/ only with --publish, never over an earlier file)
    head = fit_head(F, np.arange(n), final_set, mask_name, alpha, workers)
    path = data / "heads" / f"final_{final_set}.npz"
    if publish:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = fresh(OUT_DIR / "mood_head.npz")
    export(head, path)
    res["final"] = {"path": str(path), "set": final_set, "classes": head["classes"],
                    "arousal": head["arousal"].tolist(), "valence": head["valence"].tolist(),
                    "train_loss": float(head["model"].loss_), "epochs": int(head["model"].n_iter_),
                    "moodreader_agreement_all_clean": agreement(head, path, F["clean"])}
    print(f"wrote {path}: {res['final']['moodreader_agreement_all_clean']}")
    (data / "results.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {data / 'results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "features", "train", "all"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--classes", choices=list(SETS), default="6", help="class set of the exported head")
    ap.add_argument("--publish", action="store_true",
                    help="also write the head to models/prosody-head/ (mood_head.npz, or _v2, _v3 if taken)")
    args = ap.parse_args()
    if args.step in ("download", "all"):
        download(args.data)
    if args.step in ("features", "all"):
        features(args.data, args.workers)
    if args.step in ("train", "all"):
        train(args.data, args.workers, args.classes, args.publish)


if __name__ == "__main__":
    main()
