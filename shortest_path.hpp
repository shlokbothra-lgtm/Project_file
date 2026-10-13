 // shortest_path.hpp -- Dijkstra (Part C) and A* (Part D) as a header, used by the Part E runner.
// This is the same code as in partC_dijkstra.cpp / partD_astar.cpp, so the experiments measure
// exactly the algorithms that were implemented in Parts C and D.
// Everything is written from scratch: only std::priority_queue / std::vector are used.
//
// Metric definitions (identical in both algorithms):
//   explored      = vertices settled (popped with final distance; stale queue entries skipped)
//   edgesExamined = adjacency entries scanned (relaxation attempts) from settled vertices
//   timeMs        = the search itself incl. dist/parent allocation and path reconstruction;
//                   excludes graph loading and the one-off heuristic setup
//
// Dijkstra complexity: O((V + E) log V). Each vertex is settled once, each arc scanned once, each
// successful relaxation pushes one heap entry (<= E+1 entries), push/pop cost O(log V).
// A*: same worst case; the heuristic only shrinks the explored region. h == 0 gives Dijkstra.
#pragma once

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <queue>
#include <vector>

#include "graph.hpp"

constexpr long long INF = LLONG_MAX / 4;

struct PathResult {
    bool found = false;            // false => destination unreachable (disconnected pair)
    long long cost = -1;
    std::vector<int> path;         // S ... D
    long long explored = 0;
    long long edgesExamined = 0;
    double timeMs = 0.0;
};

// Explicit path reconstruction: follow parent[] from the destination back to the source, reverse.
inline std::vector<int> reconstructPath(const std::vector<int>& parent, int s, int t) {
    std::vector<int> p;
    for (int v = t; v != -1; v = parent[v]) {
        p.push_back(v);
        if (v == s) break;
    }
    if (p.empty() || p.back() != s) return {};
    std::reverse(p.begin(), p.end());
    return p;
}

inline double elapsedMs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- Dijkstra: binary min-heap, lazy deletion, early exit when the destination is settled ----
inline PathResult dijkstra(const Graph& g, int s, int t) {
    PathResult r;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<long long> dist(g.n, INF);
    std::vector<int> parent(g.n, -1);
    using Item = std::pair<long long, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        ++r.explored;
        if (u == t) { r.found = true; break; }
        for (const Edge& e : g.adj[u]) {
            ++r.edgesExamined;
            long long nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                parent[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
    if (r.found) {
        r.cost = dist[t];
        r.path = reconstructPath(parent, s, t);
    }
    r.timeMs = elapsedMs(t0);
    return r;
}

// ---- Haversine heuristic: h(v) = floor(scale * haversine_metres(v, target)) ----
// scale = min over all arcs of  w / haversine(arc)  (times 1-1e-9 against rounding) =>
// admissible (triangle inequality of great-circle distance) and consistent; the floor keeps both
// properties because the weights are integers. Computed from the loaded graph, so the guarantee
// holds for every subgraph whatever unit the weights are in.
class HaversineHeuristic {
public:
    explicit HaversineHeuristic(const Graph& g, double scaleOverride = -1.0) {
        const double deg = 3.14159265358979323846 / 180.0;   // pi (M_PI is not defined on MinGW with -std=c++17)
        latr_.resize(g.n); lonr_.resize(g.n); cosl_.resize(g.n);
        for (int v = 0; v < g.n; ++v) {
            latr_[v] = g.lat[v] * deg;
            lonr_[v] = g.lon[v] * deg;
            cosl_[v] = std::cos(latr_[v]);
        }
        double minRatio = 1e300;
        for (int u = 0; u < g.n; ++u)
            for (const Edge& e : g.adj[u]) {
                double hv = dist(u, e.to);
                if (hv > 0.0) minRatio = std::min(minRatio, e.w / hv);
            }
        if (minRatio == 1e300) minRatio = 0.0;
        minRatio_ = minRatio;
        scale_ = (scaleOverride > 0.0) ? scaleOverride : minRatio * (1.0 - 1e-9);
    }
    double scale() const { return scale_; }
    double minRatio() const { return minRatio_; }
    void setTarget(int t) { target_ = t; }
    long long h(int v) const { return static_cast<long long>(std::floor(scale_ * dist(v, target_))); }

private:
    static constexpr double R = 6371008.8;
    double dist(int a, int b) const {
        double sp = std::sin((latr_[b] - latr_[a]) / 2), sl = std::sin((lonr_[b] - lonr_[a]) / 2);
        double x = sp * sp + cosl_[a] * cosl_[b] * sl * sl;
        return 2.0 * R * std::asin(std::min(1.0, std::sqrt(x)));
    }
    std::vector<double> latr_, lonr_, cosl_;
    double scale_ = 0.0, minRatio_ = 0.0;
    int target_ = 0;
};

// ---- A*: f(v) = g(v) + h(v); ties broken towards larger g; stale test on g ----
struct AStarItem {
    long long f, g;
    int u;
    bool operator>(const AStarItem& o) const {
        if (f != o.f) return f > o.f;
        return g < o.g;
    }
};

inline PathResult astar(const Graph& g, int s, int t, HaversineHeuristic& hz) {
    PathResult r;
    auto t0 = std::chrono::steady_clock::now();

    hz.setTarget(t);
    std::vector<long long> dist(g.n, INF);
    std::vector<int> parent(g.n, -1);
    std::priority_queue<AStarItem, std::vector<AStarItem>, std::greater<AStarItem>> pq;

    dist[s] = 0;
    pq.push({hz.h(s), 0, s});
    while (!pq.empty()) {
        AStarItem it = pq.top();
        pq.pop();
        int u = it.u;
        if (it.g > dist[u]) continue;
        ++r.explored;
        if (u == t) { r.found = true; break; }
        for (const Edge& e : g.adj[u]) {
            ++r.edgesExamined;
            long long ng = it.g + e.w;
            if (ng < dist[e.to]) {
                dist[e.to] = ng;
                parent[e.to] = u;
                pq.push({ng + hz.h(e.to), ng, e.to});
            }
        }
    }
    if (r.found) {
        r.cost = dist[t];
        r.path = reconstructPath(parent, s, t);
    }
    r.timeMs = elapsedMs(t0);
    return r;
}

// Independent check: consecutive path vertices are adjacent and the weights add up to `cost`.
inline bool validatePath(const Graph& g, const std::vector<int>& path, int s, int t, long long cost) {
    if (path.empty() || path.front() != s || path.back() != t) return false;
    long long sum = 0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        long long best = -1;
        for (const Edge& e : g.adj[path[i]])
            if (e.to == path[i + 1] && (best < 0 || e.w < best)) best = e.w;
        if (best < 0) return false;
        sum += best;
    }
    return sum == cost;
}