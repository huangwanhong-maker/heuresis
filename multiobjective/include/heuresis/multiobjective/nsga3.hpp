// NSGA-III: selection by fronts and reference directions, for many objectives (Deb and Jain 2014, "An evolutionary
// many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I",
// IEEE Trans. Evol. Comput. 18, 577–601).
//
// In many objectives almost every point of a population is non-dominated, so crowding distance (NSGA-II) decides
// nearly all of selection and does it badly. NSGA-III replaces it by a set of reference directions (by default
// Das and Dennis's lattice, decomposition.hpp) and keeps the population spread among them:
//
// 1. Parents and offspring are joined and sorted into fronts; S is the smallest union of leading fronts holding at
//    least μ points, F_l its last front.
// 2. Normalisation (§IV.C): the ideal point z* (the least value of each objective over S) is subtracted; the extreme
//    point of axis j minimises the achievement function maxᵢ f′ᵢ/wᵢ with w = e_j (other weights 10⁻⁶); the
//    hyperplane through the m extreme points cuts the axes at the intercepts aᵢ, and f″ = f′/a. When the extreme
//    points are degenerate (the system is singular, or an intercept is not positive and finite), the largest value
//    of each objective over S stands in for the intercept, as Deb and Jain's own code does.
// 3. Association (§IV.D): each point of S goes to the reference line through the origin nearest to f″, at
//    perpendicular distance d⊥.
// 4. Niching (§IV.E): the points before F_l are kept; the remaining places are filled from F_l one at a time: take
//    the reference direction with the fewest kept points (ties at random), among those with points of F_l left;
//    if it has none kept, add its F_l point nearest the line, otherwise one of its F_l points at random.
//
// Offspring are bred from parents drawn uniformly (§IV.F), by the crossover and mutation given (η_c = 30, η_m = 20
// on real vectors). Generational under the ask–tell protocol, read in ticket order; failed evaluations are kept only
// when nothing else can fill the population. Constraints are handled by constrained dominance in step 1 (Jain and
// Deb 2014, part II, use the same order).

#pragma once

#include <heuresis/multiobjective/decomposition.hpp>
#include <heuresis/multiobjective/dominance.hpp>
#include <heuresis/multiobjective/kernels.hpp>
#include <heuresis/multiobjective/nsga2.hpp>
#include <heuresis/search/draws.hpp>

#include <calx/linalg/dense.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace heuresis::multiobjective {

// The points of S normalised as NSGA-III normalises them.
inline std::vector<Point> nsga3_normalise(const std::vector<Point>& s) {
  const std::size_t n = s.size(), m = s.empty() ? 0 : s[0].size();
  Point ideal(m, std::numeric_limits<double>::infinity()), worst(m, -std::numeric_limits<double>::infinity());
  for (const auto& p : s)
    for (std::size_t i = 0; i < m; i++) { ideal[i] = std::min(ideal[i], p[i]); worst[i] = std::max(worst[i], p[i]); }
  std::vector<Point> t(n, Point(m));
  for (std::size_t k = 0; k < n; k++)
    for (std::size_t i = 0; i < m; i++) t[k][i] = s[k][i] - ideal[i];
  calx::linalg::Matrix E(m);
  for (std::size_t j = 0; j < m; j++) {
    std::size_t best = 0;
    double best_asf = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < n; k++) {
      double asf = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < m; i++) asf = std::max(asf, t[k][i] / (i == j ? 1.0 : 1e-6));
      if (asf < best_asf) { best_asf = asf; best = k; }
    }
    for (std::size_t i = 0; i < m; i++) E(j, i) = t[best][i];
  }
  Point a(m);
  bool degenerate = false;
  try {
    calx::linalg::Matrix B(m);
    for (std::size_t i = 0; i < m; i++) B(i, 0) = 1;
    const auto X = calx::linalg::solve(E, B);
    for (std::size_t i = 0; i < m; i++) {
      a[i] = 1 / X(i, 0);
      if (!(a[i] > 1e-6) || !std::isfinite(a[i])) degenerate = true;
    }
  } catch (const std::runtime_error&) {
    degenerate = true;
  }
  if (degenerate)
    for (std::size_t i = 0; i < m; i++) a[i] = std::max(worst[i] - ideal[i], 1e-12);
  for (auto& p : t)
    for (std::size_t i = 0; i < m; i++) p[i] /= a[i];
  return t;
}

// The nearest reference line to each point, and the perpendicular distance to it.
inline void associate(const std::vector<Point>& points, const std::vector<Point>& references, std::vector<std::size_t>& line, std::vector<double>& distance) {
  line.assign(points.size(), 0);
  distance.assign(points.size(), std::numeric_limits<double>::infinity());
  for (std::size_t k = 0; k < points.size(); k++)
    for (std::size_t j = 0; j < references.size(); j++) {
      const auto& w = references[j];
      double ww = 0, pw = 0, pp = 0;
      for (std::size_t i = 0; i < w.size(); i++) { ww += w[i] * w[i]; pw += points[k][i] * w[i]; pp += points[k][i] * points[k][i]; }
      const double d = std::sqrt(std::max(0.0, pp - pw * pw / ww));
      if (d < distance[k]) { distance[k] = d; line[k] = j; }
    }
}

template <class G>
class Nsga3 {
 public:
  using Genotype = G;
  using Individual = heuresis::search::Individual<G>;
  using Candidate = heuresis::search::Candidate<G>;

  Nsga3(Kernels<G> kernels, std::vector<Point> references, std::size_t mu, heuresis::search::Draws draws = heuresis::search::Draws{})
      : kernels_(std::move(kernels)), references_(std::move(references)), mu_(mu), draws_(draws) {
    if (mu < 2 || references_.empty()) throw std::invalid_argument("heuresis::multiobjective::Nsga3: a population of at least two and some reference directions");
  }

  std::vector<Candidate> ask(std::size_t limit) {
    if (!batch_.active()) open();
    return batch_.take(limit);
  }

  void tell(std::span<const heuresis::search::Told> told) {
    for (const auto& t : told)
      if (batch_.accept(t)) {
        auto pool = population_;
        for (auto& x : batch_.individuals()) pool.push_back(std::move(x));
        batch_.close();
        select(std::move(pool));
        generation_++;
      }
  }

  bool stopped() const { return false; }
  void inject(std::span<const Individual> in) {
    auto pool = population_;
    pool.insert(pool.end(), in.begin(), in.end());
    select(std::move(pool));
  }
  std::vector<Individual> emigrants(std::size_t k) const {
    return std::vector<Individual>(population_.begin(), population_.begin() + static_cast<std::ptrdiff_t>(std::min(k, population_.size())));
  }

  const std::vector<Individual>& population() const { return population_; }
  const std::vector<Point>& references() const { return references_; }
  std::uint64_t generation() const { return generation_; }
  std::vector<Point> front() const {
    std::vector<Evaluation> e;
    for (const auto& x : population_) e.push_back(x.evaluation);
    const auto f = non_dominated_sort(e);
    std::vector<Point> out;
    if (!f.fronts.empty())
      for (auto i : f.fronts[0])
        if (!e[i].failed()) out.push_back(e[i].objectives);
    return out;
  }

  static constexpr std::uint32_t sample_tag = heuresis::search::tag("multiobjective.nsga3.sample");
  static constexpr std::uint32_t breed_tag = heuresis::search::tag("multiobjective.nsga3.breed");
  static constexpr std::uint32_t niche_tag = heuresis::search::tag("multiobjective.nsga3.niche");

 private:
  void open() {
    std::vector<Candidate> c;
    if (population_.empty()) {
      for (std::size_t i = 0; i < mu_; i++) {
        calx::random::Stream s = draws_(sample_tag, generation_, i);
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.initialise(s), {sample_tag, {}}});
      }
    } else {
      for (std::size_t pair = 0; c.size() < mu_; pair++) {
        calx::random::Stream s = draws_(breed_tag, generation_, pair);
        const std::size_t a = s.below(population_.size()), b = s.below(population_.size());
        auto children = kernels_.crossover(population_[a].genotype, population_[b].genotype, s);
        const std::vector<heuresis::search::Ticket> parents{population_[a].ticket, population_[b].ticket};
        c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.mutate(children.first, s), {breed_tag, parents}});
        if (c.size() < mu_) c.push_back({heuresis::search::make_ticket(draws_.island(), serial_++), kernels_.mutate(children.second, s), {breed_tag, parents}});
      }
    }
    batch_.open(std::move(c));
  }

  void select(std::vector<Individual> pool) {
    std::vector<Evaluation> e(pool.size());
    for (std::size_t i = 0; i < pool.size(); i++) e[i] = pool[i].evaluation;
    const auto fr = non_dominated_sort(e);
    std::vector<std::size_t> kept, last;
    for (const auto& f : fr.fronts) {
      std::vector<std::size_t> good;
      for (auto i : f)
        if (!e[i].failed()) good.push_back(i);
      if (kept.size() + good.size() <= mu_) {
        kept.insert(kept.end(), good.begin(), good.end());
        if (kept.size() == mu_) break;
      } else {
        last = good;
        break;
      }
    }
    if (!last.empty()) {
      std::vector<std::size_t> s = kept;
      s.insert(s.end(), last.begin(), last.end());
      std::vector<Point> pts;
      for (auto i : s) pts.push_back(e[i].objectives);
      const auto normal = nsga3_normalise(pts);
      std::vector<std::size_t> line;
      std::vector<double> dist;
      associate(normal, references_, line, dist);
      const std::size_t base = kept.size();  // last's q-th member is s[base + q]
      std::vector<std::size_t> count(references_.size(), 0);
      for (std::size_t k = 0; k < base; k++) count[line[k]]++;
      std::vector<char> taken(last.size(), 0);
      calx::random::Stream r = draws_(niche_tag, generation_, 0);
      std::vector<char> open_line(references_.size(), 1);
      while (kept.size() < mu_) {
        // the open directions with the fewest kept points
        std::size_t least = std::numeric_limits<std::size_t>::max();
        for (std::size_t j = 0; j < references_.size(); j++)
          if (open_line[j]) least = std::min(least, count[j]);
        std::vector<std::size_t> candidates;
        for (std::size_t j = 0; j < references_.size(); j++)
          if (open_line[j] && count[j] == least) candidates.push_back(j);
        const std::size_t jbar = candidates[r.below(candidates.size())];
        std::vector<std::size_t> members;  // positions in `last` associated with jbar, not taken
        for (std::size_t q = 0; q < last.size(); q++)
          if (!taken[q] && line[base + q] == jbar) members.push_back(q);
        if (members.empty()) { open_line[jbar] = 0; continue; }
        std::size_t pick = members[0];
        if (count[jbar] == 0) {
          for (auto q : members)
            if (dist[base + q] < dist[base + pick]) pick = q;
        } else {
          pick = members[r.below(members.size())];
        }
        taken[pick] = 1;
        kept.push_back(last[pick]);
        count[jbar]++;
      }
    }
    // failures last, if they are all there is
    for (std::size_t i = 0; i < e.size() && kept.size() < mu_; i++)
      if (e[i].failed()) kept.push_back(i);
    std::vector<Individual> next;
    for (auto i : kept) next.push_back(std::move(pool[i]));
    population_ = std::move(next);
  }

  Kernels<G> kernels_;
  std::vector<Point> references_;
  std::size_t mu_;
  heuresis::search::Draws draws_;
  heuresis::search::Batch<G> batch_;
  std::vector<Individual> population_;
  std::uint64_t generation_ = 0, serial_ = 0;
};

}  // namespace heuresis::multiobjective
