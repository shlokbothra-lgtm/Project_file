#if 0
#!/usr/bin/env python3
"""
plot_results.py -- plots comparing Dijkstra and A* from the Part E results.

Usage:  python3 plot_results.py [results_dir]        (default: data/results)
Reads   <results_dir>/partE_summary.csv   (and partE_per_query.csv for the spread plot)
Writes  PNG files into <results_dir>/plots/ :
          time_vs_size.png, explored_vs_size.png, edges_vs_size.png,
          astar_vs_dijkstra.png, all_in_one.png, time_per_query.png
Requires: pandas, matplotlib
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "results")
plot_dir = os.path.join(res_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(os.path.join(res_dir, "partE_summary.csv"))
style = {"Dijkstra": dict(color="tab:blue", marker="o", ls="-"),
         "A*": dict(color="tab:red", marker="s", ls="--")}
algos = [a for a in ("Dijkstra", "A*") if a in set(df.algorithm)]


def line_plot(ax, column, ylabel, title):
    for a in algos:
        d = df[df.algorithm == a].sort_values("vertices")
        ax.plot(d.vertices, d[column], label=a, lw=2, ms=7, **style[a])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size (number of vertices)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


panels = [("avg_time_ms", "Average execution time (ms)", "Execution time vs graph size", "time_vs_size.png"),
          ("avg_vertices_explored", "Average vertices explored", "Vertices explored vs graph size",
           "explored_vs_size.png"),
          ("avg_edges_examined", "Average edges examined", "Edges examined vs graph size", "edges_vs_size.png")]
for col, ylab, title, fname in panels:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    line_plot(ax, col, ylab, title)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, fname), dpi=160)
    plt.close(fig)

# A* relative to Dijkstra (only if both algorithms are present)
if len(algos) == 2:
    piv = df.pivot(index="vertices", columns="algorithm")
    sizes = piv.index.values
    ratio_v = piv["avg_vertices_explored"]["A*"] / piv["avg_vertices_explored"]["Dijkstra"]
    ratio_e = piv["avg_edges_examined"]["A*"] / piv["avg_edges_examined"]["Dijkstra"]
    speedup = piv["avg_time_ms"]["Dijkstra"] / piv["avg_time_ms"]["A*"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(sizes, ratio_v, "s-", color="tab:green", lw=2, label="vertices explored")
    a1.plot(sizes, ratio_e, "^--", color="tab:orange", lw=2, label="edges examined")
    a1.axhline(1.0, color="gray", lw=1, ls=":")
    a1.set_xscale("log")
    a1.set_ylim(0, 1.1)
    a1.set_xlabel("Graph size (number of vertices)")
    a1.set_ylabel("A* / Dijkstra")
    a1.set_title("Search effort of A* relative to Dijkstra\n(below 1 = A* does less work)")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend()
    a2.plot(sizes, speedup, "o-", color="tab:purple", lw=2)
    a2.axhline(1.0, color="gray", lw=1, ls=":")
    a2.set_xscale("log")
    a2.set_xlabel("Graph size (number of vertices)")
    a2.set_ylabel("Dijkstra time / A* time")
    a2.set_title("Execution-time speed-up of A*\n(above 1 = A* is faster)")
    a2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "astar_vs_dijkstra.png"), dpi=160)
    plt.close(fig)

# one figure with the three main panels
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (col, ylab, title, _) in zip(axes, panels):
    line_plot(ax, col, ylab, title)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "all_in_one.png"), dpi=160)
plt.close(fig)

# spread over queries: per-query time of each algorithm at every size (box plot)
pq_path = os.path.join(res_dir, "partE_per_query.csv")
if os.path.exists(pq_path):
    pq = pd.read_csv(pq_path)
    sizes = sorted(pq.graph_size.unique())
    fig, ax = plt.subplots(figsize=(max(6.4, 1.4 * len(sizes) + 3), 4.4))
    width = 0.35
    for k, a in enumerate(algos):
        data = [pq[(pq.graph_size == s) & (pq.algorithm == a)].time_ms.values for s in sizes]
        pos = [i + (k - (len(algos) - 1) / 2) * width for i in range(len(sizes))]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor=style[a]["color"], alpha=0.5)
        ax.plot([], [], color=style[a]["color"], lw=6, alpha=0.5, label=a)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yscale("log")
    ax.set_xlabel("Graph size label (vertices)")
    ax.set_ylabel("Execution time per query (ms)")
    ax.set_title("Spread of execution time over the queries")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "time_per_query.png"), dpi=160)
    plt.close(fig)

print("plots written to", plot_dir)
for f in sorted(os.listdir(plot_dir)):
    print("  ", f)
#endif