
"""
preprocess.py  --  DIMACS road network  ->  CSV graphs for the C++ shortest-path code.

Dataset : 9th DIMACS Implementation Challenge, USA-road-d.LKS (Great Lakes, TIGER/Line)
Input   : USA-road-d.LKS.gr(.gz)   arcs:   "a <u> <v> <distance>"
          USA-road-d.LKS.co(.gz)   coords: "v <id> <lon*1e6> <lat*1e6>"   (optional but needed for A*)

Pipeline
  1. Parse arcs (and coordinates).
  2. Clean: drop self-loops and non-positive weights, merge parallel arcs (keep minimum weight).
  3. Keep the largest (weakly) connected component.
  4. For every requested size N, take the first N vertices of a BFS from a fixed seed vertex
     and keep the induced subgraph. A BFS ball is always connected; the same seed is used for
     all sizes, so the subgraphs are nested and the experiment is reproducible.
  5. Relabel vertices 0..N-1 (BFS order) and write CSV files.
  6. Generate source-destination query pairs for each size.
  7. Write statistics (vertices, edges, average degree, max degree) and, if coordinates exist,
     the scale factor that keeps the haversine heuristic admissible.

Outputs in --outdir
  graph_<N>_edges.csv    u,v,w            directed arcs, 0-indexed (road arcs come in both directions)
  graph_<N>_nodes.csv    id,orig_id,lat,lon   (lat/lon only when --co is given)
  queries_<N>.csv        source,target
  stats.csv              per-graph statistics
  heuristic.json         haversine admissibility scale (only with --co)

Requires: numpy, scipy      Usage example:
  python3 preprocess.py --gr USA-road-d.LKS.gr.gz --co USA-road-d.LKS.co.gz --outdir data
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components  # used only for connectivity, not shortest paths

# ----------------------------------------------------------------------------- IO helpers
def open_text(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def read_gr(path):
    """Return n, U, V, W (1-indexed ids as in the file)."""
    n = None
    U, V, W = [], [], []
    with open_text(path) as f:
        for line in f:
            c = line[0]
            if c == "a":
                _, u, v, w = line.split()
                U.append(int(u)); V.append(int(v)); W.append(int(w))
            elif c == "p":
                n = int(line.split()[2])
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    if n is None:
        n = int(max(U.max(), V.max()))
    return n, U, V, W


def read_co(path, n):
    """Return lat, lon arrays in degrees, indexed by 1-based id (index 0 unused)."""
    lat = np.full(n + 1, np.nan)
    lon = np.full(n + 1, np.nan)
    with open_text(path) as f:
        for line in f:
            if line[0] == "v":
                _, i, x, y = line.split()
                lon[int(i)] = int(x) / 1e6
                lat[int(i)] = int(y) / 1e6
    return lat, lon


# ----------------------------------------------------------------------------- cleaning
def clean_arcs(U, V, W, n):
    """Remove self-loops / non-positive weights, merge parallel arcs by min weight."""
    stats = {"raw_arcs": int(len(U))}
    keep = (U != V) & (W > 0)
    stats["removed_self_loops_or_nonpositive"] = int((~keep).sum())
    U, V, W = U[keep], V[keep], W[keep]

    key = U * (n + 1) + V
    order = np.lexsort((W, key))          # sort by key, then by weight ascending
    key, U, V, W = key[order], U[order], V[order], W[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]       # first occurrence has the minimum weight
    stats["merged_parallel_arcs"] = int((~first).sum())
    U, V, W = U[first], V[first], W[first]
    stats["clean_arcs"] = int(len(U))
    return U, V, W, stats


def largest_component(U, V, n):
    """Boolean mask over 1..n (index 0 unused) of vertices in the largest weakly connected component."""
    g = csr_matrix((np.ones(len(U), dtype=np.int8), (U, V)), shape=(n + 1, n + 1))
    ncomp, labels = connected_components(g, directed=True, connection="weak")
    labels[0] = -1                                  # dummy vertex 0
    counts = np.bincount(labels[1:], minlength=ncomp)
    big = int(np.argmax(counts))
    mask = labels == big
    mask[0] = False
    return mask, int(ncomp) - 1, int(counts[big])   # -1: dummy vertex 0 is its own component


# ----------------------------------------------------------------------------- CSR + BFS
def build_csr(U, V, n):
    """Adjacency in CSR form (indptr, indices) for BFS."""
    order = np.argsort(U, kind="stable")
    indices = V[order]
    indptr = np.zeros(n + 2, dtype=np.int64)
    np.add.at(indptr, U + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, indices


def bfs_ball(indptr, indices, seed, size):
    """First `size` vertices reached by BFS from `seed` (deterministic order)."""
    seen = {seed}
    order = [seed]
    dq = deque([seed])
    while dq and len(order) < size:
        u = dq.popleft()
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            if v not in seen:
                seen.add(v)
                order.append(v)
                dq.append(v)
                if len(order) >= size:
                    break
    return np.asarray(order, dtype=np.int64)


# ----------------------------------------------------------------------------- geometry
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------- writers
def write_edges(path, u, v, w):
    with open(path, "w") as f:
        f.write("u,v,w\n")
        np.savetxt(f, np.column_stack((u, v, w)), fmt="%d", delimiter=",")


def write_nodes(path, orig, lat=None, lon=None):
    n = len(orig)
    with open(path, "w") as f:
        if lat is None:
            f.write("id,orig_id\n")
            np.savetxt(f, np.column_stack((np.arange(n), orig)), fmt="%d", delimiter=",")
        else:
            f.write("id,orig_id,lat,lon\n")
            for i in range(n):
                f.write(f"{i},{orig[i]},{lat[i]:.6f},{lon[i]:.6f}\n")


def graph_stats(n_vertices, u, n_mapped):
    """Degree statistics. Road arcs are symmetric, so out-degree == degree."""
    deg = np.bincount(u, minlength=n_mapped)
    return {
        "vertices": int(n_vertices),
        "arcs": int(len(u)),
        "undirected_edges": int(len(u) // 2),
        "avg_degree": round(float(len(u)) / n_vertices, 4),
        "max_degree": int(deg.max()),
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gr", required=True, help="DIMACS .gr file (.gz ok)")
    ap.add_argument("--co", default=None, help="DIMACS .co coordinate file (.gz ok); needed for A*")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000, 50000, 100000])
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (seed vertex + query pairs)")
    ap.add_argument("--queries", type=int, default=10, help="source-destination pairs per size (>=5)")
    ap.add_argument("--full", action="store_true", help="also write the full cleaned largest component")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print(f"[1/6] reading {args.gr}")
    n, U, V, W = read_gr(args.gr)
    print(f"      nodes={n:,} arcs={len(U):,}  ({time.time()-t0:.1f}s)")

    lat = lon = None
    if args.co:
        print(f"      reading {args.co}")
        lat, lon = read_co(args.co, n)
        missing = int(np.isnan(lat[1:]).sum())
        print(f"      coordinates loaded, missing={missing}")
    else:
        print("      WARNING: no --co file given -> no coordinates, A* heuristic unavailable")

    print("[2/6] cleaning arcs")
    U, V, W, cstats = clean_arcs(U, V, W, n)
    for k, v in cstats.items():
        print(f"      {k}: {v:,}")

    print("[3/6] largest connected component")
    mask, ncomp, lcc_size = largest_component(U, V, n)
    print(f"      components={ncomp:,}  largest={lcc_size:,} vertices")
    keep = mask[U] & mask[V]
    U, V, W = U[keep], V[keep], W[keep]

    print("[4/6] building adjacency (CSR)")
    indptr, indices = build_csr(U, V, n)

    # Seed vertex: random vertex of the LCC, fixed by --seed. Same seed for all sizes -> nested subgraphs.
    rng = np.random.default_rng(args.seed)
    lcc_ids = np.flatnonzero(mask)
    seed_vertex = int(rng.choice(lcc_ids))
    print(f"      seed vertex (orig id) = {seed_vertex}")

    # sort arcs by (u, v) once so subgraph weights can be pulled via CSR order
    order = np.argsort(U, kind="stable")
    Us, Vs, Ws = U[order], V[order], W[order]

    stats_rows = []
    heur = None

    sizes = sorted(set(s for s in args.sizes if s <= lcc_size))
    skipped = [s for s in args.sizes if s > lcc_size]
    if skipped:
        print(f"      skipped sizes larger than LCC: {skipped}")

    print(f"[5/6] extracting subgraphs {sizes}")
    for N in sizes:
        t1 = time.time()
        ball = bfs_ball(indptr, indices, seed_vertex, N)         # original ids in BFS order
        remap = np.full(n + 1, -1, dtype=np.int64)
        remap[ball] = np.arange(len(ball))

        sel = remap[Us] >= 0
        sel &= remap[Vs] >= 0
        su, sv, sw = remap[Us[sel]], remap[Vs[sel]], Ws[sel]
        o = np.lexsort((sv, su))                                  # sorted by (u, v) for tidy files
        su, sv, sw = su[o], sv[o], sw[o]

        write_edges(os.path.join(args.outdir, f"graph_{N}_edges.csv"), su, sv, sw)
        if lat is not None:
            write_nodes(os.path.join(args.outdir, f"graph_{N}_nodes.csv"), ball, lat[ball], lon[ball])
        else:
            write_nodes(os.path.join(args.outdir, f"graph_{N}_nodes.csv"), ball)

        # query pairs: distinct random vertices (subgraph is connected, so every pair is reachable)
        qr = np.random.default_rng(args.seed + N)
        q = []
        while len(q) < max(args.queries, 5):
            s, d = (int(x) for x in qr.integers(0, len(ball), size=2))
            if s != d:
                q.append((s, d))
        with open(os.path.join(args.outdir, f"queries_{N}.csv"), "w") as f:
            f.write("source,target\n")
            for s, d in q:
                f.write(f"{s},{d}\n")

        st = graph_stats(len(ball), su, len(ball))
        st["size_label"] = N
        stats_rows.append(st)
        print(f"      N={N:>7,}: arcs={st['arcs']:>9,} avg_deg={st['avg_degree']:.3f} "
              f"max_deg={st['max_degree']}  ({time.time()-t1:.1f}s)")

    if args.full:
        print("      writing full largest component (can be large)")
        ids = np.flatnonzero(mask)
        remap = np.full(n + 1, -1, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        fu, fv = remap[Us], remap[Vs]
        write_edges(os.path.join(args.outdir, "graph_full_edges.csv"), fu, fv, Ws)
        if lat is not None:
            write_nodes(os.path.join(args.outdir, "graph_full_nodes.csv"), ids, lat[ids], lon[ids])
        else:
            write_nodes(os.path.join(args.outdir, "graph_full_nodes.csv"), ids)
        st = graph_stats(len(ids), fu, len(ids))
        st["size_label"] = "full"
        stats_rows.append(st)

    # Heuristic admissibility: need  h(u,v) <= true cost.  With h = scale * haversine(u,v),
    # admissible iff scale <= min over arcs of  w / haversine(arc).  (Then it is also consistent.)
    if lat is not None:
        print("[6/6] heuristic scale factor (haversine)")
        hv = haversine_m(lat[Us], lon[Us], lat[Vs], lon[Vs])
        ok = hv > 0
        ratio = Ws[ok] / hv[ok]
        heur = {
            "formula": "h(v) = scale * haversine_meters(v, target)",
            "scale_admissible": float(ratio.min()),
            "ratio_min": float(ratio.min()),
            "ratio_p01": float(np.percentile(ratio, 1)),
            "ratio_median": float(np.median(ratio)),
            "ratio_max": float(ratio.max()),
            "arcs_with_zero_haversine": int((~ok).sum()),
            "note": "Use scale_admissible (or smaller) in C++ to guarantee admissibility/consistency. "
                    "ratio_median ~1 means weights are metres.",
        }
        with open(os.path.join(args.outdir, "heuristic.json"), "w") as f:
            json.dump(heur, f, indent=2)
        print(f"      weight/haversine ratio: min={heur['ratio_min']:.4f} "
              f"median={heur['ratio_median']:.4f} max={heur['ratio_max']:.2f}")
    else:
        print("[6/6] skipped heuristic (no coordinates)")

    with open(os.path.join(args.outdir, "stats.csv"), "w") as f:
        f.write("graph,vertices,arcs,undirected_edges,avg_degree,max_degree\n")
        for s in stats_rows:
            f.write(f"{s['size_label']},{s['vertices']},{s['arcs']},{s['undirected_edges']},"
                    f"{s['avg_degree']},{s['max_degree']}\n")
    with open(os.path.join(args.outdir, "cleaning_report.json"), "w") as f:
        json.dump({**cstats, "components": ncomp, "largest_component_vertices": lcc_size,
                   "seed_vertex_orig_id": seed_vertex, "rng_seed": args.seed}, f, indent=2)

    print(f"done in {time.time()-t0:.1f}s  ->  {args.outdir}/")


if __name__ == "__main__":
    sys.exit(main())
