# -*- coding: utf-8 -*-
"""Dense video captioning metrics (E.T. Bench F1 / METEOR / CIDEr / Sim, and SODA_c) for a test run.

SODA_c here: preds and GT are aligned by the SODA dynamic program on the IoU matrix, then METEOR
(GT as reference) is summed over the aligned pairs without IoU weighting; P = sum / #pred,
R = sum / #gt, F averaged over videos.

Predictions are parsed as one "start - end caption" record per line (continuation lines join the
previous record). IoU, P/R, caption pairing and the failed-sample handling follow the official
E.T. Bench evaluation code (github.com/PolyU-ChenLab/ETBench): videos whose prediction cannot be
parsed are left out and penalised only through the scale factor (#parsed / #total).
Reads <run_dir>/test_results_rank*.json (written by scripts_run/test.sh) and scores it against the
ChronusAV eval json (only dvc samples are scored; other tasks in the file are ignored); the result is
printed and saved to <run_dir>/etbench_metrics.json.
usage: python3 my_tools/etbench_dvc_metrics.py --run_dir OUTPUT_ROOT/RUN_NAME --gt /path/to/eval.json [--gpu 0]
"""
import argparse, ast, glob, json, random, re, string
import numpy as np, torch
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

TIOUS = [0.1, 0.3, 0.5, 0.7]
random.seed(0)

RECORD_HEAD = re.compile(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(.*)$")

def parse_pred_records(text):
    """'start - end caption' lines -> [(interval, caption)]; continuation lines join the previous record."""
    records = []
    for line in text.split("\n"):
        m = RECORD_HEAD.match(line)
        if m:
            records.append([[float(m.group(1)), float(m.group(2))], m.group(3)])
        elif records:
            records[-1][1] += " " + line.strip()
    return records

def parse_gt_events(entry):
    """Eval-json dvc entry -> ([intervals], [captions]), or None if unparseable."""
    t = entry.get("tgt")
    t = ast.literal_eval(t) if isinstance(t, str) else t
    if not t:
        return None
    h = t[: len(t) // 2]
    intervals = [(h[i], h[i + 1]) for i in range(0, len(h) - 1, 2)]
    ref = entry["conversations"][1]["value"]
    caps = ["Visual:" + c for c in ref.split("Visual:") if c.strip()]
    if len(caps) != len(intervals):
        return None
    return intervals, [c.strip() for c in caps]

def iou_et(a, b):   # E.T. Bench DVCEval.iou
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = min(max(a[1], b[1]) - min(a[0], b[0]), b[1] - b[0] + a[1] - a[0])
    return float(inter) / (union + 1e-8)

def remove_nonascii(t):
    return "".join(c if ord(c) < 128 else " " for c in t)

def random_string(n):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))

class SentSim:
    def __init__(self, device):
        from transformers import AutoTokenizer, AutoModel
        name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tok = AutoTokenizer.from_pretrained(name); self.m = AutoModel.from_pretrained(name).to(device).eval(); self.dev = device
    @torch.no_grad()
    def encode(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            b = self.tok(texts[i:i+256], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.dev)
            h = self.m(**b).last_hidden_state; mask = b["attention_mask"].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1).clamp_min(1e-9)          # mean pooling
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())    # Normalize module
        return torch.cat(out)
    def compute_score(self, gts, res):   # pycocoevalcap scorer interface
        keys = list(gts.keys()); a = self.encode([res[k][0] for k in keys]); b = self.encode([gts[k][0] for k in keys])
        s = (a * b).sum(-1).tolist(); return float(np.mean(s)), s

def load_run(run_dir):
    res = []
    for f in sorted(glob.glob(f"{run_dir}/test_results_rank*.json")):
        res += json.load(open(f))
    return res

def soda_align(scores):
    """SODA DP alignment (Fujita et al. 2020): ordered one-to-one matching maximising the summed score.
    scores: [num_gt, num_pred]. Returns [(gt_idx, pred_idx), ...]."""
    M, N = scores.shape; dp = np.full((M, N), -1.0); path = np.zeros((M, N), dtype=int)
    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0: st = [-1, -1, scores[i, j]]
            elif i == 0: st = [-1, dp[i, j-1], scores[i, j]]
            elif j == 0: st = [dp[i-1, j], -1, scores[i, j]]
            else: st = [dp[i-1, j], dp[i, j-1], dp[i-1, j-1] + scores[i, j]]
            dp[i, j] = max(st); path[i, j] = int(np.argmax(st))
    pairs, i, j = [], M - 1, N - 1          # backtrack (same rule as the official get_pairs)
    while True:
        p = np.where(path[i][:j + 1] == 2)[0]
        if i != 0 and len(p) == 0:
            i -= 1; continue
        pairs.append((i, int(p[-1])))
        if i == 0 or p[-1] == 0:
            break
        i, j = i - 1, int(p[-1]) - 1
    return pairs[::-1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True, help="directory holding test_results_rank*.json"); ap.add_argument("--gt", required=True, help="eval annotation json"); ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    gtd = {str(e["video_id"]): e for e in json.load(open(a.gt)) if e.get("task", "dvc") == "dvc"}
    preds = {}; n_total = 0; n_failed = 0
    for r in load_run(a.run_dir):
        if r.get("task", "dvc") != "dvc": continue          # the eval file may hold all seven tasks
        v = str(r["video_id"]); n_total += 1
        recs = parse_pred_records(r["pred"] if isinstance(r["pred"], str) else r["pred"][0])
        if not recs: n_failed += 1; continue
        preds[v] = [(tuple(t), remove_nonascii(c)) for t, c in recs]
    gts = {}
    for v in preds:
        g = parse_gt_events(gtd[v]) if v in gtd else None
        if g: gts[v] = (list(g[0]), [remove_nonascii(c) for c in g[1]])
    vids = list(gts)                        # only successfully parsed samples are scored
    scale = len(preds) / max(n_total, 1)    # failed samples count through this scale factor
    tok = PTBTokenizer(); meteor = Meteor(); cider = Cider(); sent = SentSim(dev)
    print(f"[metrics] run_dir={a.run_dir} gt={a.gt} videos={len(vids)} parsed={len(preds)} failed={n_failed} scale={scale:.4f}")

    # ---------------- E.T. Bench: detection P/R per threshold ----------------
    f1s = {}; 
    for thr in TIOUS:
        P, R = [], []
        for v in vids:
            gi = gts[v][0]; pr = preds.get(v, [])
            ref_cov, pred_cov = set(), set()
            for pi, (pt, _) in enumerate(pr):
                for ri, rt in enumerate(gi):
                    if iou_et(pt, rt) > thr: ref_cov.add(ri); pred_cov.add(pi)
            P.append(len(pred_cov) / len(pr)); R.append(len(ref_cov) / len(gi))
        p, r = np.mean(P) * scale, np.mean(R) * scale
        f1s[thr] = 0 if p + r == 0 else 2 * p * r / (p + r)
    F1 = float(np.mean(list(f1s.values())))

    # ---------------- E.T. Bench: caption scores per threshold ----------------
    cap = {"METEOR": [], "CIDEr": [], "Sim": []}
    for thr in TIOUS:
        cur_res, cur_gts, vid2 = {}, {}, {v: [] for v in vids}; k = 0
        for v in vids:
            for pt, pc in preds.get(v, []):
                added = False
                for ri, rt in enumerate(gts[v][0]):
                    if iou_et(pt, rt) >= thr:
                        cur_res[k] = [{"caption": pc}]; cur_gts[k] = [{"caption": gts[v][1][ri]}]; vid2[v].append(k); k += 1; added = True
                if not added:
                    cur_res[k] = [{"caption": pc}]; cur_gts[k] = [{"caption": random_string(random.randint(10, 20))}]; vid2[v].append(k); k += 1
        tr, tg = tok.tokenize(cur_res), tok.tokenize(cur_gts)
        for scorer, name in ((meteor, "METEOR"), (cider, "CIDEr"), (sent, "Sim")):
            per = []
            for v in vids:
                ids = vid2[v]
                if not ids: per.append(0.0); continue
                sc, _ = scorer.compute_score({i: tg[i] for i in ids}, {i: tr[i] for i in ids}); per.append(float(sc))
            cap[name].append(float(np.mean(per)))
    METEOR, CIDER, SIM = [float(np.mean(cap[n])) for n in ("METEOR", "CIDEr", "Sim")]

    # ---------------- SODA_c: IoU-only alignment, unweighted METEOR sum ----------------
    ps, rs, fs = [], [], []
    for v in vids:
        pr = sorted(preds.get(v, []), key=lambda x: x[0][0])                     # preds and GT sorted by start
        gi, gc = (list(x) for x in zip(*sorted(zip(gts[v][0], gts[v][1]), key=lambda x: x[0][0])))
        if not pr: ps.append(0.0); rs.append(0.0); fs.append(0.0); continue
        io = np.array([[iou_et(pt, g) for pt, _ in pr] for g in gi])
        pairs = soda_align(io)
        res = {k: [{"caption": pr[pi][1].replace("\x00", " ")}] for k, (ri, pi) in enumerate(pairs)}
        ref = {k: [{"caption": gc[ri]}] for k, (ri, pi) in enumerate(pairs)}
        tr, tg = tok.tokenize(res), tok.tokenize(ref)
        _, sc = meteor.compute_score(tg, tr)                       # per-pair METEOR, GT as reference
        best = float(np.sum(sc))
        p, r = best / len(pr), best / len(gi); ps.append(p); rs.append(r); fs.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    SODA = float(np.mean(fs))

    out = dict(run_dir=a.run_dir, gt=a.gt, videos=len(vids), failed=n_failed, F1=round(F1 * 100, 2), Sim=round(SIM * 100, 2), SODA_c=round(SODA * 100, 2),
               CIDEr=round(CIDER * 100, 2), METEOR=round(METEOR * 100, 2), F1_at=({str(t): round(f * 100, 2) for t, f in f1s.items()}),
               SODA_P=round(float(np.mean(ps)) * 100, 2), SODA_R=round(float(np.mean(rs)) * 100, 2))
    print(json.dumps(out, ensure_ascii=False))
    with open(f"{a.run_dir}/etbench_metrics.json", "w") as f: json.dump(out, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
