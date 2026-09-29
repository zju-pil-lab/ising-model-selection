// Exact d-dimensional Moreau-envelope logistic L1 estimator, Eq. (100).
// No full graph or high-dimensional regression is generated here.
#include <cmath>
#include <algorithm>
#include <vector>

static double soft(double x, double l) { return std::copysign(std::max(std::abs(x) - l, 0.0), x); }
static double prox(double b, double chi) {
  double lo = b, hi = b + 2 * chi,
         y = b + chi * (1 - std::tanh(b)) / (1 + chi / std::pow(std::cosh(b), 2));
  for (int k = 0; k < 30; k++) {
    double t = std::tanh(y), f = y - b - chi * (1 - t);
    if (std::abs(f) < 2e-14 * (1 + std::abs(y)))
      break;
    if (f > 0)
      hi = y;
    else
      lo = y;
    double n = y - f / (1 + chi * (1 - t * t));
    y = (n > lo && n < hi) ? n : 0.5 * (lo + hi);
  }
  return y;
}
static double eval(const double *x, const double *z, int m, int d, const double *j, double chi,
                   double sq, double *grad, double *hess, double *hs = nullptr,
                   double *rs = nullptr) {
  std::fill(grad, grad + d, 0);
  std::fill(hess, hess + d * d, 0);
  double obj = 0;
  for (int i = 0; i < m; i++) {
    double b = sq * z[i];
    for (int k = 0; k < d; k++)
      b += x[i * d + k] * j[k];
    double y = prox(b, chi), t = std::tanh(y), r = 1 - t, h = (1 - t * t) / (1 + chi * (1 - t * t));
    obj += 0.5 * chi * r * r + std::log1p(std::exp(-2 * std::abs(y))) + std::max(-2 * y, 0.0);
    if (hs)
      hs[i] = h;
    if (rs)
      rs[i] = r;
    for (int k = 0; k < d; k++) {
      grad[k] -= x[i * d + k] * r / m;
      for (int l = 0; l < d; l++)
        hess[k * d + l] += x[i * d + k] * x[i * d + l] * h / m;
    }
  }
  return obj / m;
}
static void inverse(double *a, double *out, int n) {
  std::fill(out, out + n * n, 0);
  for (int i = 0; i < n; i++)
    out[i * n + i] = 1;
  for (int i = 0; i < n; i++) {
    double diag = a[i * n + i];
    for (int k = 0; k < n; k++) {
      a[i * n + k] /= diag;
      out[i * n + k] /= diag;
    }
    for (int j = 0; j < n; j++)
      if (j != i) {
        double w = a[j * n + i];
        for (int k = 0; k < n; k++) {
          a[j * n + k] -= w * a[i * n + k];
          out[j * n + k] -= w * out[i * n + k];
        }
      }
  }
}
extern "C" void solve_batch(const double *X, const double *Z, int T, int M, int D, double chi,
                            double Q, double lam, double *J, double *stats) {
  double sq = std::sqrt(Q);
  for (int t = 0; t < T; t++) {
    const double *x = X + t * M * D;
    const double *z = Z + t * M;
    double *j = J + t * D;
    double g[8], h[64], v[8], delta[8], gg[8], hh[64];
    int iteration = 0;
    for (; iteration < 80; iteration++) {
      double obj = eval(x, z, M, D, j, chi, sq, g, h);
      for (int k = 0; k < D; k++) {
        obj += lam * std::abs(j[k]);
        v[k] = j[k];
      }
      for (int cd = 0; cd < 150; cd++) {
        double change = 0;
        for (int k = 0; k < D; k++) {
          double rho = h[k * D + k] * j[k] - g[k];
          for (int l = 0; l < D; l++)
            if (l != k)
              rho -= h[k * D + l] * (v[l] - j[l]);
          double next = soft(rho, lam) / h[k * D + k];
          change = std::max(change, std::abs(next - v[k]));
          v[k] = next;
        }
        if (change < 1e-13)
          break;
      }
      double size = 0, direction = 0;
      for (int k = 0; k < D; k++) {
        delta[k] = v[k] - j[k];
        size = std::max(size, std::abs(delta[k]));
        direction += g[k] * delta[k] + lam * (std::abs(v[k]) - std::abs(j[k]));
      }
      if (size < 2e-11)
        break;
      double step = 1;
      for (int ls = 0; ls < 30; ls++) {
        for (int k = 0; k < D; k++)
          v[k] = j[k] + step * delta[k];
        double next = eval(x, z, M, D, v, chi, sq, gg, hh);
        for (int k = 0; k < D; k++)
          next += lam * std::abs(v[k]);
        if (next <= obj + 1e-4 * step * direction + 1e-15)
          break;
        step *= 0.5;
      }
      for (int k = 0; k < D; k++)
        j[k] = v[k];
    }
    std::vector<double> hs(M), rs(M);
    eval(x, z, M, D, j, chi, sq, g, h, hs.data(), rs.data());
    int active[8], a = 0;
    double kkt = 0;
    for (int k = 0; k < D; k++) {
      if (std::abs(j[k]) > 1e-9) {
        active[a++] = k;
        kkt = std::max(kkt, std::abs(g[k] + std::copysign(lam, j[k])));
      } else
        kkt = std::max(kkt, std::max(std::abs(g[k]) - lam, 0.0));
    }
    double mat[64], inv[64];
    for (int k = 0; k < a; k++)
      for (int l = 0; l < a; l++)
        mat[k * a + l] = M * h[active[k] * D + active[l]];
    if (a)
      inverse(mat, inv, a);
    double e = 0, f = 0, raw = 0;
    for (int i = 0; i < M; i++) {
      double leverage = 0;
      for (int k = 0; k < a; k++)
        for (int l = 0; l < a; l++)
          leverage += x[i * D + active[k]] * inv[k * a + l] * x[i * D + active[l]];
      // Stein identity includes the response of the optimized active J to z.
      e += hs[i] - hs[i] * hs[i] * leverage;
      f += rs[i] * rs[i];
      raw -= z[i] * rs[i] / sq;
    }
    stats[t * 5] = e / M;
    stats[t * 5 + 1] = f / M;
    stats[t * 5 + 2] = raw / M;
    stats[t * 5 + 3] = kkt;
    stats[t * 5 + 4] = iteration;
  }
}
