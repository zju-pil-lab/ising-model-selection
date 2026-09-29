// Full-graph Metropolis sampling and full (N-1)-dimensional L1 estimators.
// Independent of the replica order parameters and local surrogate code.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

static double soft(double x, double t) { return std::copysign(std::max(std::abs(x) - t, 0.0), x); }

extern "C" void sample_ising(const int32_t *neighbors, const int8_t *signs, int N, int D,
                             double coupling, uint64_t seed, int burn, int gap, int count,
                             int initial, int8_t *output, double *diag) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> pick(0, N - 1);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<int8_t> s(N);
  for (int i = 0; i < N; i++)
    s[i] = (initial == 0 ? (rng() & 1 ? 1 : -1) : initial);
  std::vector<double> acceptance(2 * D + 1);
  for (int q = -D; q <= D; q++)
    acceptance[q + D] = std::min(1.0, std::exp(-2 * coupling * q));
  uint64_t attempts = 0, accepted = 0;
  auto sweeps = [&](int number) {
    for (int sweep = 0; sweep < number; sweep++)
      for (int t = 0; t < N; t++) {
        int i = pick(rng), field = 0;
        for (int k = 0; k < D; k++)
          field += signs[i * D + k] * s[neighbors[i * D + k]];
        int q = s[i] * field;
        if (q <= 0 || uniform(rng) < acceptance[q + D]) {
          s[i] = -s[i];
          accepted++;
        }
        attempts++;
      }
  };
  sweeps(burn);
  for (int mu = 0; mu < count; mu++) {
    sweeps(gap);
    std::copy(s.begin(), s.end(), output + mu * N);
  }
  diag[0] = double(accepted) / attempts;
  diag[1] = double(attempts);
}

static double logistic_loss(double h) {
  return std::log1p(std::exp(-2 * std::abs(h))) + std::max(-2 * h, 0.0);
}
static double kkt(const int8_t *X, int M, int P, const double *J,
                  const std::vector<double> &residual, double lam) {
  double violation = 0;
  for (int j = 0; j < P; j++) {
    const int8_t *x = X + j * M;
    double g = 0;
    for (int m = 0; m < M; m++)
      g -= x[m] * residual[m] / M;
    double v =
        J[j] == 0 ? std::max(std::abs(g) - lam, 0.0) : std::abs(g + std::copysign(lam, J[j]));
    violation = std::max(violation, v);
  }
  return violation;
}

extern "C" int fit_l1(const int8_t *X, int M, int P, double lam, int loss, double tol,
                      int max_sweeps, double *J, double *stats) {
  // X is feature-major; gauge-transformed response is exactly 1.
  std::fill(J, J + P, 0);
  std::vector<double> margin(M, 0), residual(M, 1), trial(M);
  double curvature = 1, meanloss = (loss == 0 ? .5 : std::log(2.0));
  int sweep = 0;
  for (; sweep < max_sweeps; sweep++) {
    for (int j = 0; j < P; j++) {
      const int8_t *x = X + j * M;
      double g = 0;
      for (int m = 0; m < M; m++)
        g -= x[m] * residual[m] / M;
      if (J[j] == 0 && std::abs(g) <= lam)
        continue;
      if (loss == 0) {
        double next = soft(J[j] - g, lam), delta = next - J[j];
        if (delta == 0)
          continue;
        for (int m = 0; m < M; m++)
          residual[m] -= x[m] * delta;
        J[j] = next;
      } else {
        double next = soft(J[j] - g / curvature, lam / curvature), delta = next - J[j];
        if (delta == 0)
          continue;
        double direction = g * delta + lam * (std::abs(next) - std::abs(J[j]));
        double step = 1, trialloss = 0;
        for (int ls = 0; ls < 50; ls++) {
          trialloss = 0;
          for (int m = 0; m < M; m++)
            trialloss += logistic_loss(margin[m] + step * delta * x[m]) / M;
          if (trialloss + lam * std::abs(J[j] + step * delta) <=
              meanloss + lam * std::abs(J[j]) + 1e-4 * step * direction + 2e-15)
            break;
          step *= .5;
        }
        J[j] = (step == 1 ? next : J[j] + step * delta);
        curvature = 0;
        for (int m = 0; m < M; m++) {
          margin[m] += step * delta * x[m];
          double t = std::tanh(margin[m]);
          residual[m] = 1 - t;
          curvature += (1 - t * t) / M;
        }
        meanloss = trialloss;
      }
    }
    double violation = kkt(X, M, P, J, residual, lam);
    if (violation < tol) {
      sweep++;
      break;
    }
  }
  double objective = 0;
  if (loss == 0) {
    for (double r : residual)
      objective += .5 * r * r / M;
  } else {
    for (double h : margin)
      objective += logistic_loss(h) / M;
  }
  for (int j = 0; j < P; j++)
    objective += lam * std::abs(J[j]);
  stats[0] = kkt(X, M, P, J, residual, lam);
  stats[1] = sweep;
  stats[2] = objective;
  return stats[0] <= tol ? 0 : 1;
}
