 // partD_astar.cpp -- Part D: A* algorithm, f(v) = g(v) + h(v) (stand-alone program).
//
// Build:  g++ -O2 -std=c++17 partD_astar.cpp -o partD_astar
// Usage:  ./partD_astar <edges.csv> <nodes.csv> (--query S D | --queries queries.csv)
//                       [--repeat R] [--scale X] [--out results.csv] [--paths paths.csv]
//   nodes.csv MUST contain lat/lon (the heuristic needs coordinates).
//   --query S D     one source-destination pair; prints the complete path
//   --queries f     every pair in a query file (header "source,target"); prints a table + averages
//   --repeat R      time each search R times and report the mean (default 3)
//   --scale X       override the heuristic scale (default: computed from the graph, admissible)
//   --out / --paths default: <edges prefix>_astar_results.csv / _astar_paths.csv
//
// Same metrics and definitions as Dijkstra (partC_dijkstra.cpp):
//   vertices explored = vertices settled (popped, stale entries skipped)
//   edges examined    = adjacency entries scanned from settled vertices
//   execution time    = the search itself (incl. dist/parent allocation, path reconstruction);
//                       graph loading and the one-off heuristic setup are excluded
//
// ---- Heuristic --------------------------------------------------------------------------
//   h(v) = floor( scale * haversine_metres(v, destination) )
// Why it fits: vertices are real road intersections with latitude/longitude, and the straight
// (great-circle) distance is a natural lower bound on distance travelled along roads.
// Admissible: great-circle distance is a metric, so by the triangle inequality
//   haversine(v,D) <= sum of haversine over the arcs of ANY v->D path.
// If scale <= min over all arcs of  w(u,v)/haversine(u,v)  then  scale*haversine(arc) <= w(arc)
// for every arc, hence h(v) <= true remaining cost: it never overestimates.
// Consistent: h(u) <= w(u,v) + h(v), because
//   scale*hav(u,D) <= scale*hav(u,v) + scale*hav(v,D) <= w(u,v) + scale*hav(v,D).
// The floor keeps both properties because w is an integer (floor(w + x) = w + floor(x)).
// Consequence of consistency: a vertex is settled at most once and the first time the destination
// is popped its cost is optimal. The scale is computed from the loaded graph (one pass over the
// arcs), so the guarantee holds for every subgraph whatever unit the weights are in.
// Complexity: same worst case as Dijkstra, O((V + E) log V); the heuristic only shrinks the
// region actually explored. With h == 0 the algorithm is exactly Dijkstra.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <vector>

#include "graph.hpp"

constexpr long long INF = LLONG_MAX / 4;

struct PathResult {
    bool found = false;           // false => destination unreachable (disconnected pair)
    long long cost = -1;
    std::vector<int> path;        // S ... D
    long long explored = 0;
    long long edgesExamined = 0;
    double timeMs = 0.0;
};

static std::vector<int> reconstructPath(const std::vector<int>& parent, int s, int t) {
    std::vector<int> p;
    for (int v = t; v != -1; v = parent[v]) {
        p.push_back(v);
        if (v == s) break;
    }
    if (p.empty() || p.back() != s) return {};
    std::reverse(p.begin(), p.end());
    return p;
}

// ------------------------------------------------------------------------------------------
// Haversine heuristic with precomputed radians / cos(lat) per vertex.
// ------------------------------------------------------------------------------------------
class HaversineHeuristic {
public:
    HaversineHeuristic(const Graph& g, double scaleOverride) {
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
        scale_ = (scaleOverride > 0.0) ? scaleOverride : minRatio * (1.0 - 1e-9);  // margin vs rounding
    }
    double scale() const { return scale_; }
    double minRatio() const { return minRatio_; }
    void setTarget(int t) { target_ = t; }
    long long h(int v) const { return static_cast<long long>(std::floor(scale_ * dist(v, target_))); }

private:
    static constexpr double R = 6371008.8;                       // mean Earth radius, metres
    double dist(int a, int b) const {
        double sp = std::sin((latr_[b] - latr_[a]) / 2), sl = std::sin((lonr_[b] - lonr_[a]) / 2);
        double x = sp * sp + cosl_[a] * cosl_[b] * sl * sl;
        return 2.0 * R * std::asin(std::min(1.0, std::sqrt(x)));
    }
    std::vector<double> latr_, lonr_, cosl_;
    double scale_ = 0.0, minRatio_ = 0.0;
    int target_ = 0;
};

// ------------------------------------------------------------------------------------------
// A*: min-priority queue ordered by f = g + h (ties: larger g first, i.e. the deeper vertex).
// Stale entries are detected through g (not a closed set), so the code stays correct even if a
// non-consistent heuristic is plugged in (vertices would then simply be re-opened).
// ------------------------------------------------------------------------------------------
struct QItem {
    long long f, g;
    int u;
    bool operator>(const QItem& o) const {            // "a > b" = a comes out of the heap later
        if (f != o.f) return f > o.f;
        return g < o.g;
    }
};

static PathResult astar(const Graph& g, int s, int t, HaversineHeuristic& hz) {
    PathResult r;
    auto t0 = std::chrono::steady_clock::now();

    hz.setTarget(t);
    std::vector<long long> dist(g.n, INF);            // g(v): best known cost from the source
    std::vector<int> parent(g.n, -1);
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;

    dist[s] = 0;
    pq.push({hz.h(s), 0, s});
    while (!pq.empty()) {
        QItem it = pq.top();
        pq.pop();
        int u = it.u;
        if (it.g > dist[u]) continue;                 // stale entry
        ++r.explored;
        if (u == t) { r.found = true; break; }
        for (const Edge& e : g.adj[u]) {
            ++r.edgesExamined;
            long long ng = it.g + e.w;
            if (ng < dist[e.to]) {
                dist[e.to] = ng;
                parent[e.to] = u;
                pq.push({ng + hz.h(e.to), ng, e.to}); // f = g + h
            }
        }
    }
    if (r.found) {
        r.cost = dist[t];
        r.path = reconstructPath(parent, s, t);
    }
    r.timeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

static bool validatePath(const Graph& g, const std::vector<int>& path, int s, int t, long long cost) {
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

// ------------------------------------------------------------------------------------------
static std::string prefixOf(const std::string& p) {
    const std::string suf = "_edges.csv";
    if (p.size() > suf.size() && p.compare(p.size() - suf.size(), suf.size(), suf) == 0)
        return p.substr(0, p.size() - suf.size());
    return p;
}

static std::vector<std::pair<int, int>> loadQueries(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open query file: " + path);
    std::vector<std::pair<int, int>> q;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        size_t c = line.find(',');
        if (c == std::string::npos) throw std::runtime_error("bad query row: " + line);
        q.push_back({std::stoi(line.substr(0, c)), std::stoi(line.substr(c + 1))});
    }
    return q;
}

static std::string joinPath(const std::vector<int>& p, const Graph* g = nullptr) {
    std::ostringstream o;
    for (size_t i = 0; i < p.size(); ++i) o << (i ? " " : "") << (g ? g->orig_id[p[i]] : p[i]);
    return o.str();
}

int main(int argc, char** argv) {
    std::string edgesPath, nodesPath, queriesPath, outPath, pathsPath;
    int qs = -1, qd = -1, repeat = 3, positional = 0;
    double scaleOverride = -1.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--query" && i + 2 < argc)        { qs = std::stoi(argv[++i]); qd = std::stoi(argv[++i]); }
        else if (a == "--queries" && i + 1 < argc) queriesPath = argv[++i];
        else if (a == "--repeat" && i + 1 < argc)  repeat = std::max(1, std::stoi(argv[++i]));
        else if (a == "--scale" && i + 1 < argc)   scaleOverride = std::stod(argv[++i]);
        else if (a == "--out" && i + 1 < argc)     outPath = argv[++i];
        else if (a == "--paths" && i + 1 < argc)   pathsPath = argv[++i];
        else if (positional == 0)                  { edgesPath = a; ++positional; }
        else if (positional == 1)                  { nodesPath = a; ++positional; }
        else { std::cerr << "unexpected argument: " << a << "\n"; return 1; }
    }
    if (edgesPath.empty() || nodesPath.empty() || (qs < 0 && queriesPath.empty())) {
        std::cerr << "usage: " << argv[0] << " <edges.csv> <nodes.csv> (--query S D | --queries q.csv)"
                  << " [--repeat R] [--scale X] [--out f.csv] [--paths f.csv]\n";
        return 1;
    }
    if (outPath.empty())   outPath = prefixOf(edgesPath) + "_astar_results.csv";
    if (pathsPath.empty()) pathsPath = prefixOf(edgesPath) + "_astar_paths.csv";

    Graph g;
    std::vector<std::pair<int, int>> queries;
    try {
        g.load(edgesPath, nodesPath);
        queries = (qs >= 0) ? std::vector<std::pair<int, int>>{{qs, qd}} : loadQueries(queriesPath);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    if (!g.has_coords) {
        std::cerr << "error: " << nodesPath << " has no lat/lon columns; A* needs coordinates\n";
        return 2;
    }
    for (auto [s, d] : queries)
        if (s < 0 || d < 0 || s >= g.n || d >= g.n) {
            std::cerr << "error: query (" << s << "," << d << ") outside 0.." << g.n - 1 << "\n";
            return 2;
        }

    auto th = std::chrono::steady_clock::now();
    HaversineHeuristic hz(g, scaleOverride);
    double setupMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th).count();

    std::ofstream res(outPath), pth(pathsPath);
    if (!res || !pth) { std::cerr << "error: cannot write output files\n"; return 3; }
    res << "query,source,target,algorithm,found,cost,vertices_explored,edges_examined,time_ms,path_vertices\n";
    pth << "query,algorithm,cost,path_vertices,path,path_orig_ids\n";

    const bool single = (queries.size() == 1);
    std::cout << "A* | graph: " << g.n << " vertices, " << g.m << " arcs | " << queries.size()
              << " queries, time averaged over " << repeat << " runs\n"
              << "heuristic: haversine, scale = " << std::setprecision(6) << hz.scale()
              << " (min weight/haversine ratio = " << hz.minRatio() << "), setup " << std::fixed
              << std::setprecision(1) << setupMs << " ms\n";
    if (scaleOverride > 0 && scaleOverride > hz.minRatio())
        std::cout << "WARNING: --scale exceeds the admissible bound; A* may return non-optimal paths\n";
    std::cout << "\n";
    if (!single)
        std::cout << std::left << std::setw(4) << "#" << std::setw(9) << "src" << std::setw(9) << "dst"
                  << std::right << std::setw(12) << "cost" << std::setw(11) << "explored" << std::setw(12)
                  << "edges" << std::setw(11) << "time_ms" << "\n";

    double sumT = 0, sumV = 0, sumE = 0, sumC = 0;
    int nFound = 0, invalid = 0;
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        auto [s, d] = queries[qi];
        PathResult r;
        double tsum = 0;
        for (int k = 0; k < repeat; ++k) {
            PathResult cur = astar(g, s, d, hz);
            tsum += cur.timeMs;
            if (k == repeat - 1) r = cur;
        }
        r.timeMs = tsum / repeat;
        if (r.found && !validatePath(g, r.path, s, d, r.cost)) {
            std::cerr << "INVALID PATH on query " << qi << "\n";
            ++invalid;
        }
        res << qi << "," << s << "," << d << ",A*," << (r.found ? 1 : 0) << "," << r.cost << "," << r.explored
            << "," << r.edgesExamined << "," << std::fixed << std::setprecision(4) << r.timeMs << ","
            << r.path.size() << "\n";
        pth << qi << ",A*," << r.cost << "," << r.path.size() << "," << joinPath(r.path) << ","
            << joinPath(r.path, &g) << "\n";
        if (r.found) { ++nFound; sumT += r.timeMs; sumV += r.explored; sumE += r.edgesExamined; sumC += r.cost; }

        if (single) {
            std::cout << "Source " << s << " -> destination " << d << "\n";
            if (!r.found) {
                std::cout << "  no path: source and destination are disconnected\n";
            } else {
                std::cout << "  shortest-path cost : " << r.cost << "\n"
                          << "  path length        : " << r.path.size() << " vertices\n"
                          << "  vertices explored  : " << r.explored << "\n"
                          << "  edges examined     : " << r.edgesExamined << "\n"
                          << std::fixed << std::setprecision(4)
                          << "  execution time     : " << r.timeMs << " ms\n"
                          << "  path               : " << joinPath(r.path) << "\n";
            }
        } else {
            std::cout << std::left << std::setw(4) << qi << std::setw(9) << s << std::setw(9) << d << std::right
                      << std::setw(12) << r.cost << std::setw(11) << r.explored << std::setw(12)
                      << r.edgesExamined << std::fixed << std::setprecision(4) << std::setw(11) << r.timeMs
                      << (r.found ? "" : "   (no path)") << "\n";
        }
    }
    if (!single) {
        double k = std::max(1, nFound);
        std::cout << "\nAverage over " << nFound << " reachable queries: cost " << std::fixed
                  << std::setprecision(1) << sumC / k << ", explored " << sumV / k << ", edges " << sumE / k
                  << ", time " << std::setprecision(4) << sumT / k << " ms\n";
    }
    std::cout << "Path validation : " << (invalid ? "FAILED" : "all returned paths valid") << "\n"
              << "Results written : " << outPath << "\nPaths written   : " << pathsPath << "\n";
    return invalid ? 4 : 0;
}