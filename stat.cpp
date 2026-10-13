// partB_stats.cpp -- Part B: build adjacency list from CSV and report graph statistics.
//
// Build:  g++ -O2 -std=c++17 partB_stats.cpp -o partB_stats
// Run  :  ./partB_stats data/graph_10000_edges.csv data/graph_10000_nodes.csv [stats_out.csv]
//         (nodes file is optional; stats_out.csv is appended to if given)
#include <chrono>
#include <iomanip>
#include <iostream>

#include "graph.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <edges.csv> [nodes.csv] [stats_out.csv]\n";
        return 1;
    }
    std::string edgesPath = argv[1];
    std::string nodesPath = (argc >= 3) ? argv[2] : "";
    std::string outPath   = (argc >= 4) ? argv[3] : "";

    Graph g;
    auto t0 = std::chrono::steady_clock::now();
    try {
        g.load(edgesPath, nodesPath);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    auto t1 = std::chrono::steady_clock::now();
    double loadMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    GraphStats s = g.stats();

    std::cout << "Graph file            : " << edgesPath << "\n"
              << "Representation        : adjacency list (vector<vector<Edge>>)\n"
              << "Coordinates loaded    : " << (g.has_coords ? "yes" : "no") << "\n"
              << "Number of vertices    : " << s.vertices << "\n"
              << "Number of arcs (dir.) : " << s.arcs << "\n"
              << "Graph is symmetric    : " << (s.symmetric ? "yes" : "no") << "\n"
              << "Number of edges       : " << s.undirected_edges
              << (s.symmetric ? "  (undirected: arcs / 2)" : "  (directed)") << "\n"
              << std::fixed << std::setprecision(4)
              << "Average degree        : " << s.avg_degree << "  (arcs / vertices)\n"
              << "Maximum degree        : " << s.max_degree << "\n"
              << "Minimum degree        : " << s.min_degree << "\n"
              << "Isolated vertices     : " << s.isolated << "\n"
              << "Weight min / max      : " << s.min_weight << " / " << s.max_weight << "\n"
              << std::setprecision(1)
              << "Load time (ms)        : " << loadMs << "\n";

    if (!outPath.empty()) {
        bool fresh = !std::ifstream(outPath).good();
        std::ofstream out(outPath, std::ios::app);
        if (fresh)
            out << "graph,vertices,arcs,edges,avg_degree,max_degree,min_degree,isolated,w_min,w_max\n";
        out << std::fixed << std::setprecision(4)
            << edgesPath << "," << s.vertices << "," << s.arcs << "," << s.undirected_edges << ","
            << s.avg_degree << "," << s.max_degree << "," << s.min_degree << "," << s.isolated << ","
            << s.min_weight << "," << s.max_weight << "\n";
    }
    return 0;
}