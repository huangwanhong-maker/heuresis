// Checks of heuresis/assessment/conformal: the quantile is the ⌈(n + 1)(1 − α)⌉-th smallest score, and the whole line when
// that exceeds n; over 200 replications the split conformal interval of a Gaussian process covers new runs within the band
// [1 − α, 1 − α + 1/(n + 1)] the theorem gives, plain and normalised; the normalised intervals are narrower where the noise
// is small and wider where it is large; screening keeps the true minimiser among the candidates and assesses fewer than all;
// a biased predictor's intervals, calibrated on unbiased runs elsewhere, are reported as covering less on what is screened.

#include <heuresis/assessment/conformal.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace heuresis::assessment;

namespace {
double f(double x) { return std::sin(3 * x) + 0.5 * x * x; }
double noise_sd(double x) { return 0.05 + 0.3 * std::fabs(x) / 2; }  // small near 0, large near ±2
}  // namespace

int main() {
  calx::verification::Suite s;
  {
    // Scores 1 … 19 (a predictor of 0 at runs of value i): n = 19, α = 0.1 gives the 18th smallest.
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    for (int i = 1; i <= 19; i++) x.push_back({static_cast<double>(i)}), y.push_back(i);
    const auto zero = [](std::span<const double>) { return Prediction{0, 1}; };
    s.check("the quantile is the ⌈(n + 1)(1 − α)⌉-th smallest score: the 18th of 19 at α = 0.1", calibrate(zero, x, y, 0.1).quantile() - 18, 0);
    const auto c = calibrate(zero, x, y, 0.02);
    s.require("when ⌈(n + 1)(1 − α)⌉ exceeds n the interval is the whole line, and says so", c(std::vector<double>{1.0}).whole_line && std::isinf(c.quantile()));
  }
  {
    // 200 replications: fit on 40 runs, calibrate on 99, test on 400 new runs.
    double plain = 0, normed = 0;
    double narrow_mid = 0, narrow_edge = 0;
    const int reps = 200;
    for (int r = 0; r < reps; r++) {
      calx::random::Stream g(500 + r, 1, 0, 0);
      auto draw = [&](std::vector<std::vector<double>>& xs, std::vector<double>& ys, int n) {
        for (int i = 0; i < n; i++) {
          const double x = g.uniform(-2, 2);
          xs.push_back({x}), ys.push_back(f(x) + noise_sd(x) * g.normal());
        }
      };
      std::vector<std::vector<double>> xt, xc, xs;
      std::vector<double> yt, yc, ys;
      draw(xt, yt, 40), draw(xc, yc, 99), draw(xs, ys, 400);
      const auto P = calibrate(gaussian_process(xt, yt, false), xc, yc, 0.1), N = calibrate(gaussian_process(xt, yt, true), xc, yc, 0.1);
      int in_p = 0, in_n = 0;
      for (std::size_t i = 0; i < xs.size(); i++) {
        const auto a = P(xs[i]), b = N(xs[i]);
        in_p += ys[i] >= a.lower && ys[i] <= a.upper, in_n += ys[i] >= b.lower && ys[i] <= b.upper;
      }
      plain += in_p / 400.0 / reps, normed += in_n / 400.0 / reps;
      if (r < 20) {
        const auto m = N(std::vector<double>{0.0}), e = N(std::vector<double>{1.9});
        narrow_mid += (m.upper - m.lower) / 20, narrow_edge += (e.upper - e.lower) / 20;
      }
    }
    // The theorem's band for n = 99, α = 0.1: [0.90, 0.91]; the mean over 200 × 400 runs within ±0.006 of it.
    s.require("split conformal, plain: mean coverage within the theorem's band [0.90, 0.91]", plain > 0.894 && plain < 0.916, std::to_string(plain));
    s.require("split conformal, normalised by the process's deviation: likewise", normed > 0.894 && normed < 0.916, std::to_string(normed));
    s.require("the normalised intervals are narrower where the noise is small than where it is large", narrow_mid < narrow_edge,
              std::to_string(narrow_mid) + " at 0 against " + std::to_string(narrow_edge) + " at 1.9");
  }
  {
    calx::random::Stream g(9, 2, 0, 0);
    std::vector<std::vector<double>> xt, xc, cand;
    std::vector<double> yt, yc, truth;
    for (int i = 0; i < 40; i++) {
      const double x = g.uniform(-2, 2);
      xt.push_back({x}), yt.push_back(f(x) + noise_sd(x) * g.normal());
    }
    for (int i = 0; i < 99; i++) {
      const double x = g.uniform(-2, 2);
      xc.push_back({x}), yc.push_back(f(x) + noise_sd(x) * g.normal());
    }
    for (int i = 0; i < 200; i++) {
      const double x = g.uniform(-2, 2);
      cand.push_back({x}), truth.push_back(f(x) + noise_sd(x) * g.normal());
    }
    const auto c = calibrate(gaussian_process(xt, yt), xc, yc, 0.05);
    const auto S = screen(cand, c, [&](std::size_t i) { return truth[i]; });
    std::size_t best = 0;
    for (std::size_t i = 1; i < truth.size(); i++)
      if (truth[i] < truth[best]) best = i;
    s.require("screening keeps the true minimiser among the candidates", S.best == best);
    s.require("... and assesses fewer than all of them", S.assessed.size() < cand.size() && !S.screened.empty(), std::to_string(S.assessed.size()) + " of 200 assessed");
    // A predictor biased by +1 where x < 0, calibrated on runs where x > 0 only: on the candidates it covers less.
    const auto biased = [&](std::span<const double> z) { return Prediction{f(z[0]) + (z[0] < 0 ? 1.0 : 0.0), 1.0}; };
    std::vector<std::vector<double>> xr;
    std::vector<double> yr;
    for (int i = 0; i < 99; i++) {
      const double x = g.uniform(0, 2);
      xr.push_back({x}), yr.push_back(f(x) + noise_sd(x) * g.normal());
    }
    const auto B = screen(cand, calibrate(biased, xr, yr, 0.1), [&](std::size_t i) { return truth[i]; });
    s.require("a predictor calibrated where it was right covers less on what it screens, and the screen says so", B.coverage() < 0.85, std::to_string(B.coverage()));
  }
  return s.done();
}
