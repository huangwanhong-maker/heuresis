// Archives of elites over a space of behaviours, and the measures of a repertoire.
//
// The search for quality and diversity asks not for the best solution but for the best solution of every kind
// (Pugh, Soros and Stanley 2016, "Quality diversity: a new frontier for evolutionary computation", Front. Robot. AI
// 3, 40). Each evaluation places its point in a space of behaviours B ⊂ ℝᵈ by its descriptors; B is cut into
// cells by a tessellation, and an archive keeps in each cell the best individual seen there (by Deb's rule on the
// first objective, so constraints and failures are handled as everywhere in heuresis). The archive is a function
// from cells to elites; offering an individual changes it only where the individual is better than the cell's
// elite.
//
// Tessellations:
// - a grid of rᵢ intervals on each axis of a box (Mouret and Clune 2015, "Illuminating search spaces by mapping
//   elites", arXiv:1504.04909, §3); descriptors outside the box fall in its boundary cells;
// - Voronoi cells of given sites, each descriptor in the cell of its nearest site (Vassiliades, Chatzilygeroudis
//   and Mouret 2018, "Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic
//   elites algorithm", IEEE Trans. Evol. Comput. 22, 623–630). Their CVT takes the sites as the centroids of k-means
//   on samples of B, so that cells have nearly equal volume in any dimension; k-means is tyche's to provide and is
//   not yet written, so the caller supplies the sites (any sites give a valid tessellation).
//
// Measures (Pugh et al. 2016; Mouret and Clune 2015, §4): coverage, the fraction of cells filled; the QD-score,
// Σ over filled feasible cells of (offset − f), with the offset an upper bound of the values so that each term is
// positive. Both never decrease as individuals are offered: a new cell adds a positive term, a replacement lowers
// f in its cell.

#pragma once

#include <heuresis/search/protocol.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::diversity {

using heuresis::search::Evaluation;

class Grid {
 public:
  Grid(std::vector<double> lower, std::vector<double> upper, std::vector<std::size_t> resolution) : lower_(std::move(lower)), upper_(std::move(upper)), resolution_(std::move(resolution)) {
    if (lower_.size() != upper_.size() || lower_.size() != resolution_.size() || lower_.empty()) throw std::invalid_argument("heuresis::diversity::Grid: bounds and resolutions of one dimension");
    cells_ = 1;
    for (std::size_t i = 0; i < resolution_.size(); i++) {
      if (resolution_[i] == 0 || !(upper_[i] > lower_[i])) throw std::invalid_argument("heuresis::diversity::Grid: an empty axis");
      cells_ *= resolution_[i];
    }
  }
  std::size_t cells() const { return cells_; }
  std::size_t dimension() const { return lower_.size(); }
  // The cell of a descriptor, the first axis fastest.
  std::size_t cell(std::span<const double> d) const {
    if (d.size() != lower_.size()) throw std::invalid_argument("heuresis::diversity::Grid::cell: a descriptor of another dimension");
    std::size_t c = 0, stride = 1;
    for (std::size_t i = 0; i < d.size(); i++) {
      const double t = (d[i] - lower_[i]) / (upper_[i] - lower_[i]) * static_cast<double>(resolution_[i]);
      const auto k = static_cast<std::size_t>(std::clamp(std::floor(t), 0.0, static_cast<double>(resolution_[i] - 1)));
      c += k * stride;
      stride *= resolution_[i];
    }
    return c;
  }
  // The centre of a cell.
  std::vector<double> centre(std::size_t c) const {
    std::vector<double> x(lower_.size());
    for (std::size_t i = 0; i < x.size(); i++) {
      const std::size_t k = c % resolution_[i];
      c /= resolution_[i];
      x[i] = lower_[i] + (static_cast<double>(k) + 0.5) * (upper_[i] - lower_[i]) / static_cast<double>(resolution_[i]);
    }
    return x;
  }
  const std::vector<double>& lower() const { return lower_; }
  const std::vector<double>& upper() const { return upper_; }
  const std::vector<std::size_t>& resolution() const { return resolution_; }

 private:
  std::vector<double> lower_, upper_;
  std::vector<std::size_t> resolution_;
  std::size_t cells_ = 0;
};

class Voronoi {
 public:
  explicit Voronoi(std::vector<std::vector<double>> sites) : sites_(std::move(sites)) {
    if (sites_.empty()) throw std::invalid_argument("heuresis::diversity::Voronoi: no sites");
  }
  std::size_t cells() const { return sites_.size(); }
  std::size_t dimension() const { return sites_[0].size(); }
  // The nearest site, ties to the lower index (a scan; a k-d tree would serve many thousands of sites).
  std::size_t cell(std::span<const double> d) const {
    std::size_t best = 0;
    double bd = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < sites_.size(); j++) {
      double s = 0;
      for (std::size_t i = 0; i < d.size(); i++) s += (d[i] - sites_[j][i]) * (d[i] - sites_[j][i]);
      if (s < bd) { bd = s; best = j; }
    }
    return best;
  }
  const std::vector<double>& site(std::size_t j) const { return sites_[j]; }

 private:
  std::vector<std::vector<double>> sites_;
};

template <class G, class Tessellation>
class Archive {
 public:
  using Individual = heuresis::search::Individual<G>;

  explicit Archive(Tessellation t) : tessellation_(std::move(t)), elites_(tessellation_.cells()) {}

  // Offers an individual; true when it filled or improved its cell. Failures and individuals without descriptors
  // are refused.
  bool offer(const Individual& x) {
    if (x.evaluation.failed() || x.evaluation.descriptors.size() != tessellation_.dimension()) return false;
    const std::size_t c = tessellation_.cell(x.evaluation.descriptors);
    auto& slot = elites_[c];
    if (slot && !heuresis::search::better(x.evaluation, slot->evaluation)) return false;
    if (!slot) filled_.push_back(c);
    slot = x;
    return true;
  }

  std::size_t cells() const { return elites_.size(); }
  std::size_t size() const { return filled_.size(); }
  double coverage() const { return static_cast<double>(filled_.size()) / static_cast<double>(elites_.size()); }
  // The filled cells, in the order they were first filled.
  const std::vector<std::size_t>& filled() const { return filled_; }
  const std::optional<Individual>& elite(std::size_t c) const { return elites_[c]; }
  const Tessellation& tessellation() const { return tessellation_; }

  double qd_score(double offset) const {
    double s = 0;
    for (auto c : filled_)
      if (elites_[c]->evaluation.feasible()) s += offset - elites_[c]->evaluation.value();
    return s;
  }

 private:
  Tessellation tessellation_;
  std::vector<std::optional<Individual>> elites_;
  std::vector<std::size_t> filled_;
};

}  // namespace heuresis::diversity
