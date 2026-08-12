"""
benchmark.py - Latency Benchmark
Smart Content Recommender

OLD search : O(n) iterrows + SequenceMatcher
NEW search : 3-stage index (bisect + token inverted + trigram inverted)
OLD cosine : sklearn cosine_similarity on sparse TF-IDF (20 000 dims)
NEW cosine : FAISS IndexFlatIP on LSA-reduced dense vectors (256 dims)

Run from backend/:  python benchmark.py
No server needed -- loads pickles directly.
"""

import bisect
import os
import pickle
import re
import time
from difflib import SequenceMatcher
from statistics import median, quantiles

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ── helpers ───────────────────────────────────────────────────

def _trigrams(s):
    p = "  " + s + "  "
    return {p[i:i+3] for i in range(len(p) - 2)}

def _sim(q, t):
    if not q or not t: return 0.0
    if q == t: return 1.0
    if q in t or t in q: return 0.9
    a = set(re.findall(r"[a-z0-9]+", q))
    b = set(re.findall(r"[a-z0-9]+", t))
    ov = len(a & b) / len(a | b) if a and b else 0.0
    return max(ov, SequenceMatcher(None, q, t).ratio())


# ═══════════════════════════════════════════════════════════════
# SECTION 1 -- SEARCH BENCHMARKS
# ═══════════════════════════════════════════════════════════════

# ── OLD: O(n) iterrows + SequenceMatcher ─────────────────────

def search_old(df, query, limit=10):
    q = query.lower().strip()
    if not q: return []
    pre, sub, fuz = [], [], []
    for _, row in df.iterrows():
        t = row["title"].lower()
        if t.startswith(q): pre.append(row["title"])
        elif q in t:        sub.append(row["title"])
        else:
            s = _sim(q, t)
            if s >= 0.35:   fuz.append((s, row["title"]))
    fuz.sort(key=lambda x: x[0], reverse=True)
    return (pre + sub + [t for _, t in fuz])[:limit]


# ── NEW: 3-stage index ────────────────────────────────────────

def build_search_indexes(df):
    it = list(enumerate(df["title"]))
    pairs = sorted([(t.lower(), i) for i, t in it])
    sk    = [p[0] for p in pairs]
    tok = {}
    for i, title in it:
        for w in re.findall(r"[a-z0-9]+", title.lower()):
            if len(w) >= 2:
                tok.setdefault(w, set()).add(i)
    tok  = {k: frozenset(v) for k, v in tok.items()}
    stok = sorted(tok.keys())
    tri = {}
    for i, title in it:
        for tg in _trigrams(title.lower()):
            tri.setdefault(tg, set()).add(i)
    tri = {k: frozenset(v) for k, v in tri.items()}
    return pairs, sk, tok, stok, tri


def search_new(df, pairs, sk, tok, stok, tri, query, limit=10):
    q = query.lower().strip()
    if not q: return []
    col = {}
    hi_ch = q[:-1] + chr(ord(q[-1]) + 1)
    lo = bisect.bisect_left(sk, q)
    hi = bisect.bisect_left(sk, hi_ch)
    for i in range(lo, hi):
        col[pairs[i][1]] = 1.0
    if len(col) >= limit:
        return [df.iloc[r]["title"] for r, _ in sorted(col.items(), key=lambda x: -x[1])[:limit]]
    for w in re.findall(r"[a-z0-9]+", q):
        if len(w) < 2: continue
        if w in tok:
            for r in tok[w]:
                if r not in col: col[r] = 0.75
        else:
            we = w[:-1] + chr(ord(w[-1]) + 1)
            tlo = bisect.bisect_left(stok, w)
            thi = bisect.bisect_left(stok, we)
            for ti in range(tlo, thi):
                for r in tok[stok[ti]]:
                    if r not in col: col[r] = 0.75
    if len(col) >= limit:
        return [df.iloc[r]["title"] for r, _ in sorted(col.items(), key=lambda x: -x[1])[:limit]]
    qtri = _trigrams(q)
    ntri = max(len(qtri), 1)
    cand = {}
    for tg in qtri:
        for r in tri.get(tg, frozenset()):
            if r not in col:
                cand[r] = cand.get(r, 0) + 1
    for r, cnt in cand.items():
        s = cnt / ntri
        if s >= 0.20:
            col[r] = s * 0.60
    return [df.iloc[r]["title"] for r, _ in sorted(col.items(), key=lambda x: -x[1])[:limit]]


# ═══════════════════════════════════════════════════════════════
# SECTION 2 -- RECOMMENDATION / COSINE BENCHMARKS
# ═══════════════════════════════════════════════════════════════

# ── OLD: sklearn sparse cosine (20 000 dims) ─────────────────

def bench_cosine_old(tfidf_matrix, query_indices, runs=100):
    from sklearn.metrics.pairwise import cosine_similarity as cs
    # warm-up
    for idx in query_indices:
        cs(tfidf_matrix[idx], tfidf_matrix)
    times = []
    for _ in range(runs):
        for idx in query_indices:
            t0 = time.perf_counter()
            cs(tfidf_matrix[idx], tfidf_matrix).flatten()
            times.append((time.perf_counter() - t0) * 1000)
    return times


# ── NEW: FAISS IndexFlatIP on LSA 256-dim dense vectors ──────

def build_ann_index(tfidf_matrix, n_components=256):
    """
    TruncatedSVD (LSA) + L2-normalize + FAISS IndexFlatIP.
    Inner product on L2-normalized vectors == cosine similarity.
    """
    print(f"  Running TruncatedSVD ({n_components} components)...")
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=5)
    lsa = svd.fit_transform(tfidf_matrix)            # sparse → dense (n, 256)
    lsa_norm = normalize(lsa, norm="l2").astype("float32")
    svd_ms = (time.perf_counter() - t0) * 1000
    print(f"  SVD done in {svd_ms:.0f}ms  shape={lsa_norm.shape}")

    try:
        import faiss
        t0 = time.perf_counter()
        dim   = lsa_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(lsa_norm)
        idx_ms = (time.perf_counter() - t0) * 1000
        print(f"  FAISS IndexFlatIP built in {idx_ms:.0f}ms  "
              f"({index.ntotal} vectors, dim={dim})  [faiss-cpu {faiss.__version__}]")
        return lsa_norm, svd, index, "faiss"
    except ImportError:
        print("  faiss-cpu not installed -- using numpy fallback")
        return lsa_norm, svd, lsa_norm, "numpy"


def ann_scores(lsa_norm, ann_index, backend, query_idx):
    """Single query: return cosine score array for all titles."""
    q = lsa_norm[query_idx : query_idx + 1]   # (1, 256) float32

    if backend == "faiss":
        n = ann_index.ntotal
        scores, indices = ann_index.search(q, n)
        out = np.zeros(n, dtype="float32")
        out[indices[0]] = scores[0]
        return out
    else:
        return np.dot(ann_index, q[0])


def bench_cosine_new(lsa_norm, ann_index, backend, query_indices, runs=100):
    # warm-up
    for idx in query_indices:
        ann_scores(lsa_norm, ann_index, backend, idx)
    times = []
    for _ in range(runs):
        for idx in query_indices:
            t0 = time.perf_counter()
            ann_scores(lsa_norm, ann_index, backend, idx)
            times.append((time.perf_counter() - t0) * 1000)
    return times


# ═══════════════════════════════════════════════════════════════
# Benchmark machinery
# ═══════════════════════════════════════════════════════════════

QUERIES = [
    "the", "dark", "love", "money heist",
    "strnger thngs", "br", "house of", "nat", "comedy", "inception",
]

REC_TITLES = ["Stranger Things", "The Crown", "Breaking Bad", "Dark"]


def run_search_bench(fn, runs):
    for _ in range(2):
        for q in QUERIES: fn(q)
    times = []
    for _ in range(runs):
        for q in QUERIES:
            t0 = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - t0) * 1000)
    return times


def stats(d):
    p = quantiles(d, n=100)
    return dict(p50=median(d), p95=p[94], p99=p[98], mn=min(d), mx=max(d))


def row(label, s):
    print(f"  {label:<42}  p50={s['p50']:7.3f}ms  "
          f"p95={s['p95']:7.3f}ms  max={s['mx']:7.3f}ms")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    S = "=" * 72
    print(S)
    print("  Smart Content Recommender -- Full Latency Benchmark")
    print(S)

    # Load data
    t0 = time.perf_counter()
    df  = pd.read_pickle(os.path.join(MODELS_DIR, "processed_data.pkl"))
    with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "rb") as f:
        mat = pickle.load(f)
    print(f"  Loaded {len(df)} titles in {(time.perf_counter()-t0)*1000:.0f}ms\n")

    # ── BENCHMARK 1: Title Search ──────────────────────────────
    print(S)
    print(f"  BENCHMARK 1 -- Title Search  ({len(QUERIES)} query types)")
    print(S)

    t0 = time.perf_counter()
    pairs, sk, tok, stok, tri = build_search_indexes(df)
    bms = (time.perf_counter() - t0) * 1000
    print(f"  Search index built in {bms:.1f}ms\n")

    print("  OLD: O(n) iterrows + SequenceMatcher  [5 runs] ...")
    od = run_search_bench(lambda q: search_old(df, q), 5)
    os_ = stats(od)

    print("  NEW: 3-stage index  [500 runs] ...")
    nd = run_search_bench(lambda q: search_new(df, pairs, sk, tok, stok, tri, q), 500)
    ns = stats(nd)

    print()
    row("OLD -- iterrows + SequenceMatcher", os_)
    row("NEW -- bisect + token + trigram",   ns)
    sp50 = os_["p50"] / max(ns["p50"], 0.001)
    sp95 = os_["p95"] / max(ns["p95"], 0.001)
    u1 = 100 * sum(1 for d in nd if d < 1.0) / len(nd)
    u5 = 100 * sum(1 for d in nd if d < 5.0) / len(nd)
    print(f"\n  Speedup: {sp50:.0f}x at p50   {sp95:.0f}x at p95")
    print(f"  NEW: {u1:.0f}% calls <1ms   {u5:.0f}% calls <5ms")

    # ── BENCHMARK 2: Cosine Similarity ────────────────────────
    print(f"\n{S}")
    print(f"  BENCHMARK 2 -- Recommendation Cosine Similarity")
    print(f"  TF-IDF matrix: {mat.shape}  nnz={mat.nnz}  (99.9% sparse)")
    print(S)

    t2i = {t.lower(): i for i, t in enumerate(df["title"])}
    query_indices = [t2i[t.lower()] for t in REC_TITLES if t.lower() in t2i]
    print(f"  Query titles: {REC_TITLES[:len(query_indices)]}\n")

    print("  Building FAISS index...")
    lsa_norm, svd, ann_index, backend = build_ann_index(mat)
    print()

    print(f"  OLD: sparse sklearn cosine_similarity ({mat.shape[1]} dims)  [100 runs] ...")
    old_cos = bench_cosine_old(mat, query_indices, runs=100)
    ocs = stats(old_cos)

    print(f"  NEW: FAISS IndexFlatIP on LSA (256 dims)  [500 runs] ...")
    new_cos = bench_cosine_new(lsa_norm, ann_index, backend, query_indices, runs=500)
    ncs = stats(new_cos)

    print()
    row(f"OLD -- sparse cosine ({mat.shape[1]} dims)", ocs)
    row(f"NEW -- FAISS/LSA (256 dims, backend={backend})", ncs)
    cp50 = ocs["p50"] / max(ncs["p50"], 0.001)
    cp95 = ocs["p95"] / max(ncs["p95"], 0.001)
    cu5_old = 100 * sum(1 for d in old_cos if d < 5.0) / len(old_cos)
    cu5_new = 100 * sum(1 for d in new_cos if d < 5.0) / len(new_cos)
    print(f"\n  Speedup: {cp50:.0f}x at p50   {cp95:.0f}x at p95")
    print(f"  OLD calls <5ms: {cu5_old:.0f}%   NEW calls <5ms: {cu5_new:.0f}%")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{S}")
    print("  FINAL SUMMARY")
    print(S)
    print(f"  Dataset                    : {len(df)} titles")
    print(f"  TF-IDF shape               : {mat.shape}  (99.9% sparse)")
    print()
    print(f"  --- SEARCH ---")
    print(f"  Before  p50                : {os_['p50']:.1f}ms   (O(n) iterrows + SequenceMatcher)")
    print(f"  After   p50                : {ns['p50']:.3f}ms   (3-stage index)  [{sp50:.0f}x faster]")
    print(f"  After   p95                : {ns['p95']:.3f}ms")
    print(f"  Calls under 5ms            : {u5:.0f}%")
    print()
    print(f"  --- RECOMMENDATION (cosine) ---")
    print(f"  Before  p50                : {ocs['p50']:.2f}ms   (sklearn sparse, {mat.shape[1]} dims)")
    print(f"  After   p50                : {ncs['p50']:.3f}ms   (FAISS/LSA, 256 dims)  [{cp50:.0f}x faster]")
    print(f"  After   p95                : {ncs['p95']:.3f}ms")
    print(f"  Calls under 5ms            : {cu5_new:.0f}%")
    print(S)


if __name__ == "__main__":
    main()
