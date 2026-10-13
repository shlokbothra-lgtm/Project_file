// partE_experiments.cpp -- Part E: Dijkstra vs A* on every graph size, averaged over the queries.
//
// Build:  g++ -O2 -std=c++17 partE_experiments.cpp -o partE_experiments
// Usage:  ./partE_experiments [data_dir] [--repeat R] [--out-dir DIR]
//   data_dir   folder produced by preprocess.py (default "data"); needs, for each size N:
//              graph_N_edges.csv, graph_N_nodes.csv (with lat/lon for A*), queries_N.csv
//   --repeat R each search is timed R times after one untimed warm-up run; the mean is used (default 5)
//   --out-dir  where results are written (default <data_dir>/results)
//
// Outputs (in the out dir):
//   partE_per_query.csv   every single query for both algorithms
//   partE_summary.csv     one row per (graph size, algorithm): averages over the queries
//   partE_table.md        the Part E table (and the A*-vs-Dijkstra comparison) in Markdown
//   partE_run_info.txt    settings, compiler, sizes, heuristic scales (for reproducibility)
//
// Reproducibility: the graphs, the seed vertex and the query pairs are fixed by preprocess.py
// (--seed). Costs, paths, vertices explored and edges examined are fully deterministic; only the
// execution times vary slightly between runs / machines, hence the repeat-and-average.
//
// Correctness checks done for every query: both returned paths are re-validated edge by edge, and
// the Dijkstra and A* costs must be equal. Any failure makes the program exit with status 4.
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "shortest_path.hpp"

namespace fs = std::filesystem;

struct Segment {
    std::string label;
    long long order;
    std::string edges, nodes, queries;
};

struct Agg {                                   // accumulates averages over reachable queries
    int n = 0;
    double cost = 0, explored = 0, edges = 0, timeMs = 0, pathLen = 0;
    void add(const PathResult& r) {
        ++n; cost += r.cost; explored += r.explored; edges += r.edgesExamined;
        timeMs += r.timeMs; pathLen += r.path.size();
    }
    double avg(double x) const { return n ? x / n : 0.0; }
};

struct SummaryRow {
    std::string label;
    long long vertices, arcs;
    std::string algo;
    int queries, reachable;
    double avgCost, avgExplored, avgEdges, avgTime, avgPath, scale;
};

static bool endsWith(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
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

int main(int argc, char** argv) {
    std::string dataDir = "data", outDir;
    int repeat = 5;
    bool dirSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--repeat" && i + 1 < argc)       repeat = std::max(1, std::stoi(argv[++i]));
        else if (a == "--out-dir" && i + 1 < argc) outDir = argv[++i];
        else if (!dirSet && a[0] != '-')           { dataDir = a; dirSet = true; }
        else { std::cerr << "usage: " << argv[0] << " [data_dir] [--repeat R] [--out-dir DIR]\n"; return 1; }
    }
    fs::path dir = dataDir;
    if (!fs::is_directory(dir)) { std::cerr << "error: not a directory: " << dir << "\n"; return 1; }
    fs::path out = outDir.empty() ? dir / "results" : fs::path(outDir);
    fs::create_directories(out);

    // ---- discover segments: graph_<N>_edges.csv ------------------------------------------------
    const std::string pre = "graph_", suf = "_edges.csv";
    std::vector<Segment> segs;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string name = entry.path().filename().string();
        if (name.size() <= pre.size() + suf.size() || name.compare(0, pre.size(), pre) != 0 || !endsWith(name, suf))
            continue;
        Segment s;
        s.label = name.substr(pre.size(), name.size() - pre.size() - suf.size());
        bool numeric = !s.label.empty() &&
                       std::all_of(s.label.begin(), s.label.end(), [](unsigned char c) { return std::isdigit(c); });
        if (!numeric) continue;                                   // skip e.g. graph_full: no query file
        s.order = std::stoll(s.label);
        s.edges = entry.path().string();
        s.nodes = (dir / (pre + s.label + "_nodes.csv")).string();
        s.queries = (dir / ("queries_" + s.label + ".csv")).string();
        segs.push_back(s);
    }
    std::sort(segs.begin(), segs.end(), [](const Segment& a, const Segment& b) { return a.order < b.order; });
    if (segs.empty()) { std::cerr << "no graph_<N>_edges.csv files in " << dir << "\n"; return 1; }

    std::ofstream perq((out / "partE_per_query.csv").string());
    perq << "graph_size,query,source,target,algorithm,found,cost,vertices_explored,edges_examined,"
            "time_ms,path_vertices\n";

    std::vector<SummaryRow> rows;
    std::ostringstream info;
    int failures = 0;
    const char* names[2] = {"Dijkstra", "A*"};

    std::cout << "Part E | repeat = " << repeat << " (+1 warm-up) | output: " << out << "\n\n";

    for (const Segment& seg : segs) {
        Graph g;
        std::vector<std::pair<int, int>> queries;
        try {
            g.load(seg.edges, seg.nodes);
            queries = loadQueries(seg.queries);
        } catch (const std::exception& e) {
            std::cerr << "[" << seg.label << "] skipped: " << e.what() << "\n";
            ++failures;
            continue;
        }
        if (queries.size() < 5)
            std::cerr << "[" << seg.label << "] WARNING: only " << queries.size()
                      << " queries; the assignment requires at least 5\n";
        const bool useAStar = g.has_coords;
        HaversineHeuristic* hz = useAStar ? new HaversineHeuristic(g) : nullptr;
        if (!useAStar) std::cerr << "[" << seg.label << "] no coordinates: A* skipped\n";

        Agg agg[2];
        int mismatches = 0, invalid = 0;
        for (size_t qi = 0; qi < queries.size(); ++qi) {
            auto [s, d] = queries[qi];
            if (s < 0 || d < 0 || s >= g.n || d >= g.n) { std::cerr << "bad query id\n"; ++failures; continue; }
            PathResult r[2];
            for (int a = 0; a < (useAStar ? 2 : 1); ++a) {
                auto run = [&]() { return a == 0 ? dijkstra(g, s, d) : astar(g, s, d, *hz); };
                run();                                             // warm-up (not timed)
                double tsum = 0;
                for (int k = 0; k < repeat; ++k) {
                    PathResult cur = run();
                    tsum += cur.timeMs;
                    if (k == repeat - 1) r[a] = cur;
                }
                r[a].timeMs = tsum / repeat;
                if (r[a].found && !validatePath(g, r[a].path, s, d, r[a].cost)) {
                    std::cerr << "[" << seg.label << "] INVALID PATH (" << names[a] << ", query " << qi << ")\n";
                    ++invalid;
                }
                if (r[a].found) agg[a].add(r[a]);
                perq << seg.label << "," << qi << "," << s << "," << d << "," << names[a] << ","
                     << (r[a].found ? 1 : 0) << "," << r[a].cost << "," << r[a].explored << ","
                     << r[a].edgesExamined << "," << std::fixed << std::setprecision(4) << r[a].timeMs << ","
                     << r[a].path.size() << "\n";
            }
            if (useAStar && (r[0].found != r[1].found || r[0].cost != r[1].cost)) {
                ++mismatches;
                std::cerr << "[" << seg.label << "] COST MISMATCH query " << qi << ": " << r[0].cost << " vs "
                          << r[1].cost << "\n";
            }
        }
        failures += mismatches + invalid;

        for (int a = 0; a < (useAStar ? 2 : 1); ++a)
            rows.push_back({seg.label, g.n, g.m, names[a], (int)queries.size(), agg[a].n, agg[a].avg(agg[a].cost),
                            agg[a].avg(agg[a].explored), agg[a].avg(agg[a].edges), agg[a].avg(agg[a].timeMs),
                            agg[a].avg(agg[a].pathLen), a == 1 ? hz->scale() : -1.0});

        info << "size " << seg.label << ": " << g.n << " vertices, " << g.m << " arcs, " << queries.size()
             << " queries";
        if (useAStar) info << ", heuristic scale " << std::setprecision(6) << hz->scale();
        info << "\n";
        std::cout << std::left << std::setw(8) << seg.label << ": " << queries.size() << " queries, ";
        if (useAStar)
            std::cout << "avg explored Dijkstra " << std::fixed << std::setprecision(0) << agg[0].avg(agg[0].explored)
                      << " vs A* " << agg[1].avg(agg[1].explored) << std::setprecision(4) << " | avg time "
                      << agg[0].avg(agg[0].timeMs) << " vs " << agg[1].avg(agg[1].timeMs) << " ms"
                      << ((mismatches || invalid) ? "   <-- PROBLEM" : "") << "\n";
        else
            std::cout << "Dijkstra only\n";
        delete hz;
    }

    // ---- summary CSV ------------------------------------------------------------------------------
    {
        std::ofstream f((out / "partE_summary.csv").string());
        f << "graph_size,vertices,arcs,algorithm,queries,reachable,avg_cost,avg_vertices_explored,"
             "avg_edges_examined,avg_time_ms,avg_path_vertices,heuristic_scale\n";
        for (const auto& r : rows) {
            f << r.label << "," << r.vertices << "," << r.arcs << "," << r.algo << "," << r.queries << ","
              << r.reachable << "," << std::fixed << std::setprecision(2) << r.avgCost << "," << r.avgExplored
              << "," << r.avgEdges << "," << std::setprecision(4) << r.avgTime << "," << std::setprecision(2)
              << r.avgPath << ",";
            if (r.scale >= 0) f << std::setprecision(6) << r.scale;
            f << "\n";
        }
    }

    // ---- Markdown table(s) --------------------------------------------------------------------------
    {
        std::ofstream f((out / "partE_table.md").string());
        f << "| Graph Size (vertices) | Algorithm | Avg Shortest-Path Cost | Avg Vertices Explored | "
             "Avg Edges Examined | Avg Execution Time (ms) |\n|---:|---|---:|---:|---:|---:|\n";
        for (const auto& r : rows)
            f << "| " << r.vertices << " | " << r.algo << " | " << std::fixed << std::setprecision(1) << r.avgCost
              << " | " << r.avgExplored << " | " << r.avgEdges << " | " << std::setprecision(4) << r.avgTime << " |\n";

        f << "\n**A\\* relative to Dijkstra** (averages over the same queries)\n\n"
             "| Graph Size (vertices) | Vertices explored (A\\*/Dijkstra) | Edges examined (A\\*/Dijkstra) | "
             "Time speed-up (Dijkstra/A\\*) | Heuristic scale |\n|---:|---:|---:|---:|---:|\n";
        for (size_t i = 0; i + 1 < rows.size(); ++i)
            if (rows[i].algo == "Dijkstra" && rows[i + 1].algo == "A*" && rows[i].label == rows[i + 1].label) {
                const auto &d = rows[i], &a = rows[i + 1];
                f << "| " << d.vertices << " | " << std::fixed << std::setprecision(3)
                  << (d.avgExplored ? a.avgExplored / d.avgExplored : 0) << " | "
                  << (d.avgEdges ? a.avgEdges / d.avgEdges : 0) << " | " << std::setprecision(2)
                  << (a.avgTime ? d.avgTime / a.avgTime : 0) << "x | " << std::setprecision(4) << a.scale << " |\n";
            }
    }

    // ---- run info -----------------------------------------------------------------------------------
    {
        std::ofstream f((out / "partE_run_info.txt").string());
        f << "Part E run information\n"
          << "data dir          : " << dir.string() << "\n"
          << "timing            : mean of " << repeat << " runs per search after 1 untimed warm-up\n"
          << "compiler          : " << __VERSION__ << "\n"
#ifdef __OPTIMIZE__
          << "optimisation      : enabled (-O2 or higher)\n"
#else
          << "optimisation      : DISABLED (compile with -O2 for meaningful timings)\n"
#endif
          << "metric definitions: explored = vertices settled; edges examined = adjacency entries scanned\n"
          << info.str();
    }

    std::cout << "\nWritten to " << out << ":\n  partE_summary.csv, partE_per_query.csv, partE_table.md, "
              << "partE_run_info.txt\n"
              << (failures ? "RESULT: PROBLEMS DETECTED (see messages above)\n"
                           : "RESULT: all paths valid, Dijkstra and A* costs identical on every query\n");
    return failures ? 4 : 0;
}