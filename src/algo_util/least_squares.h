// least_squares — the pieces the nonlinear least-squares solvers share.
//
// Four solvers here are Levenberg-Marquardt: the panorama's bundle_adjust,
// SfM's bundle_adjust_sfm, global_position, and the photometric align. They
// build different systems -- one dense, two with a Schur complement over the
// points, one 6x6 per frame -- so there is no single solver to share without
// hiding what each one does. What they do share is below: how an outlier is
// down-weighted, and how the damping moves. One copy of each, so a fix or a
// new loss reaches all of them.
#pragma once

#include <algorithm>
#include <cmath>

namespace tglab {

// --- robust losses ------------------------------------------------------------------
//
// WHY A ROBUST LOSS IS NOT OPTIONAL. Squared error weights an observation by
// the square of how wrong it is, so one mismatched track -- one that survived
// RANSAC -- pulls the whole solve toward itself; on real data a few percent of
// outliers doubled a bundle adjustment's final RMS. A robust loss is solved
// as iteratively reweighted least squares: each residual's squared error is
// scaled by a weight that falls as the residual grows past `delta`.
//
//   Squared   1 everywhere: plain least squares.
//   Huber     1 inside, delta / |r| outside: quadratic near zero and linear
//             beyond, so a gross outlier pulls with a bounded force.
//   Cauchy    delta^2 / (delta^2 + r^2): falls off faster, so a gross outlier
//             pulls with almost none -- and a bad start can discount good
//             data, which is why Huber is the usual default.
enum class RobustLoss { Squared = 0, Huber = 1, Cauchy = 2 };

inline double RobustWeight(RobustLoss loss, double r, double delta) {
    if (loss == RobustLoss::Squared || !(delta > 0.0)) return 1.0;
    const double a = std::fabs(r);
    if (a <= delta) return 1.0;
    if (loss == RobustLoss::Huber) return delta / a;
    return delta * delta / (delta * delta + a * a);
}

// --- Levenberg-Marquardt damping -------------------------------------------------------
//
// The dial between Gauss-Newton (fast, can overshoot) and gradient descent
// (safe, slow): `lambda` is added to the system's diagonal, scaled by it.
// Raised when a step fails -- the system would not factor, or the cost went
// up -- and lowered when one succeeds. The factors and bounds belong to each
// solver, because they were tuned on each one's problems; this makes them
// one line of each solver instead of a scatter of literals.
struct LmDamping {
    double lambda;              // current damping
    double up;                  // times this after a step that made things worse
    double down;                // times this after one that made them better
    double floor;               // never below this
    double ceiling;             // past this no step will help: Fail() says stop
    double upSingular = 10.0;   // times this when the system would not factor

    // A step that made the cost worse. False once past the ceiling.
    bool Fail() {
        lambda *= up;
        return lambda <= ceiling;
    }
    // A damped system that still would not factor: damp harder, retry.
    bool FailSingular() {
        lambda *= upSingular;
        return lambda <= ceiling;
    }
    void Succeed() { lambda = std::max(floor, lambda * down); }
};

}  // namespace tglab
