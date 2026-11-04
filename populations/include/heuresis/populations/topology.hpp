// Topologies: who sends to whom among islands, and who neighbours whom among cells.
//
// A topology is a directed graph on {0, …, k − 1}, given by each vertex's out-neighbours in increasing order
// (the order in which migrants are delivered). The standard ones of parallel evolutionary algorithms (Cantú-Paz
// 2000, Efficient and Accurate Parallel Genetic Algorithms, Kluwer; Alba and Tomassini 2002, "Parallelism and
// evolutionary algorithms", IEEE Trans. Evol. Comput. 6, 443–462; Alba and Dorronsoro 2008, Cellular Genetic
// Algorithms, Springer, ch. 2):
//   ring          i → i + 1 (and i − 1 when bidirectional);
//   torus         a rows × cols grid with wrap-around, von Neumann's four neighbours or Moore's eight;
//   complete      every vertex to every other;
//   star          the centre 0 to and from every other;
//   hypercube     2^d vertices, i ↔ i ⊕ 2^j;
//   isolated      no edges (independent runs).
// The graph's diameter bounds how fast a good individual can spread: under selection alone it reaches a
// vertex no sooner than the vertex's distance (populations/cellular.hpp).

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace heuresis::populations {

struct Topology {
  std::vector<std::vector<std::size_t>> out;

  std::size_t size() const { return out.size(); }

  static Topology isolated(std::size_t k) { return {std::vector<std::vector<std::size_t>>(k)}; }
  static Topology ring(std::size_t k, bool bidirectional = true) {
    Topology t = isolated(k);
    if (k < 2) return t;
    for (std::size_t i = 0; i < k; i++) {
      t.out[i].push_back((i + 1) % k);
      if (bidirectional && k > 2) t.out[i].push_back((i + k - 1) % k);
    }
    t.normalise();
    return t;
  }
  static Topology torus(std::size_t rows, std::size_t cols, bool moore = false) {
    Topology t = isolated(rows * cols);
    for (std::size_t r = 0; r < rows; r++)
      for (std::size_t c = 0; c < cols; c++) {
        auto& o = t.out[r * cols + c];
        for (int dr = -1; dr <= 1; dr++)
          for (int dc = -1; dc <= 1; dc++) {
            if ((dr == 0 && dc == 0) || (!moore && dr != 0 && dc != 0)) continue;
            const std::size_t rr = (r + rows + static_cast<std::size_t>(dr + 1) - 1) % rows;
            const std::size_t cc = (c + cols + static_cast<std::size_t>(dc + 1) - 1) % cols;
            const std::size_t v = rr * cols + cc;
            if (v != r * cols + c) o.push_back(v);
          }
      }
    t.normalise();
    return t;
  }
  static Topology complete(std::size_t k) {
    Topology t = isolated(k);
    for (std::size_t i = 0; i < k; i++)
      for (std::size_t j = 0; j < k; j++)
        if (i != j) t.out[i].push_back(j);
    return t;
  }
  static Topology star(std::size_t k) {
    Topology t = isolated(k);
    for (std::size_t i = 1; i < k; i++) {
      t.out[0].push_back(i);
      t.out[i].push_back(0);
    }
    return t;
  }
  static Topology hypercube(std::size_t d) {
    Topology t = isolated(std::size_t{1} << d);
    for (std::size_t i = 0; i < t.size(); i++)
      for (std::size_t j = 0; j < d; j++) t.out[i].push_back(i ^ (std::size_t{1} << j));
    t.normalise();
    return t;
  }

  // Sorts each list and removes repeats (small rings and tori produce them).
  void normalise() {
    for (auto& o : out) {
      std::sort(o.begin(), o.end());
      o.erase(std::unique(o.begin(), o.end()), o.end());
    }
  }
};

}  // namespace heuresis::populations
