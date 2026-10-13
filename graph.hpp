// graph.hpp -- weighted graph stored as an adjacency list, loaded from the CSV files
// produced by preprocess.py.  Reused by Dijkstra (Part C) and A* (Part D).
//
// Files:
//   graph_<N>_edges.csv   header "u,v,w"                 directed arcs, 0-indexed
//   graph_<N>_nodes.csv   header "id,orig_id[,lat,lon]"  id = 0..N-1  (lat/lon optional)
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Edge {
    int to;      // head vertex
    int w;       // positive edge weight (DIMACS distance)
};

struct GraphStats {
    long long vertices = 0;
    long long arcs = 0;              // directed arcs stored in the adjacency list
    long long undirected_edges = 0;  // arcs/2 when the graph is symmetric
    bool symmetric = false;
    double avg_degree = 0.0;         // arcs / vertices
    int max_degree = 0;
    int min_degree = 0;
    long long isolated = 0;          // vertices with degree 0
    int min_weight = 0;
    int max_weight = 0;
};

class Graph {
public:
    int n = 0;                               // number of vertices
    long long m = 0;                         // number of directed arcs
    std::vector<std::vector<Edge>> adj;      // adjacency list: adj[u] = outgoing edges of u
    std::vector<double> lat, lon;            // coordinates in degrees (empty if not loaded)
    std::vector<long long> orig_id;          // original DIMACS id of each vertex (for reporting)
    bool has_coords = false;

    // Load nodes (optional) and edges. If nodesPath is empty, n = max vertex id + 1.
    void load(const std::string& edgesPath, const std::string& nodesPath = "") {
        if (!nodesPath.empty()) loadNodes(nodesPath);
        loadEdges(edgesPath);
    }

    GraphStats stats() const {
        GraphStats s;
        s.vertices = n;
        s.arcs = m;
        s.avg_degree = n ? static_cast<double>(m) / n : 0.0;
        s.min_degree = n ? static_cast<int>(adj[0].size()) : 0;
        s.min_weight = INT32_MAX;
        s.max_weight = 0;
        for (int u = 0; u < n; ++u) {
            int d = static_cast<int>(adj[u].size());
            s.max_degree = std::max(s.max_degree, d);
            s.min_degree = std::min(s.min_degree, d);
            if (d == 0) ++s.isolated;
            for (const Edge& e : adj[u]) {
                s.min_weight = std::min(s.min_weight, e.w);
                s.max_weight = std::max(s.max_weight, e.w);
            }
        }
        if (m == 0) s.min_weight = 0;
        s.symmetric = isSymmetric();
        s.undirected_edges = s.symmetric ? m / 2 : m;
        return s;
    }

    // True if every arc (u,v,w) has a matching reverse arc (v,u,w).
    bool isSymmetric() const {
        for (int u = 0; u < n; ++u)
            for (const Edge& e : adj[u]) {
                bool found = false;
                for (const Edge& r : adj[e.to])      // degrees are tiny in road graphs
                    if (r.to == u && r.w == e.w) { found = true; break; }
                if (!found) return false;
            }
        return true;
    }

private:
    // ---- tiny CSV helpers ---------------------------------------------------------------
    static void stripCR(std::string& s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    }
    // Split on ',' without allocating many strings.
    static std::vector<std::string> split(const std::string& line) {
        std::vector<std::string> out;
        size_t start = 0;
        while (true) {
            size_t p = line.find(',', start);
            if (p == std::string::npos) { out.push_back(line.substr(start)); break; }
            out.push_back(line.substr(start, p - start));
            start = p + 1;
        }
        return out;
    }

    void loadNodes(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open nodes file: " + path);
        std::string line;
        std::getline(in, line);                       // header
        stripCR(line);
        auto cols = split(line);
        has_coords = cols.size() >= 4;

        std::vector<long long> ids;
        while (std::getline(in, line)) {
            stripCR(line);
            if (line.empty()) continue;
            auto f = split(line);
            if (f.size() < 2) throw std::runtime_error("bad nodes row: " + line);
            long long id = std::stoll(f[0]);
            if (id != static_cast<long long>(orig_id.size()))
                throw std::runtime_error("node ids must be 0..N-1 in order, got " + f[0]);
            orig_id.push_back(std::stoll(f[1]));
            if (has_coords) {
                if (f.size() < 4) throw std::runtime_error("bad nodes row: " + line);
                lat.push_back(std::stod(f[2]));
                lon.push_back(std::stod(f[3]));
            }
        }
        n = static_cast<int>(orig_id.size());
        adj.assign(n, {});
    }

    void loadEdges(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open edges file: " + path);
        std::string line;
        std::getline(in, line);                       // header "u,v,w"

        const bool fixedN = n > 0;                    // n already set by the nodes file
        long long lineNo = 1;
        while (std::getline(in, line)) {
            ++lineNo;
            stripCR(line);
            if (line.empty()) continue;
            const char* p = line.c_str();
            char* end = nullptr;
            long u = std::strtol(p, &end, 10);
            if (*end != ',') throw std::runtime_error("bad edge row " + std::to_string(lineNo));
            long v = std::strtol(end + 1, &end, 10);
            if (*end != ',') throw std::runtime_error("bad edge row " + std::to_string(lineNo));
            long w = std::strtol(end + 1, &end, 10);

            if (u < 0 || v < 0) throw std::runtime_error("negative vertex id, row " + std::to_string(lineNo));
            if (w <= 0)         throw std::runtime_error("non-positive weight, row " + std::to_string(lineNo));
            if (fixedN) {
                if (u >= n || v >= n)
                    throw std::runtime_error("vertex id out of range, row " + std::to_string(lineNo));
            } else {
                int need = static_cast<int>(std::max(u, v)) + 1;
                if (need > static_cast<int>(adj.size())) adj.resize(need);
                n = std::max(n, need);
            }
            adj[u].push_back({static_cast<int>(v), static_cast<int>(w)});
            ++m;
        }
    }
};
