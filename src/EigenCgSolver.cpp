/**
 * Copyright (c) 2014 Patrick Wieschollek
 * Copyright (c) 2025 Michael Tesch
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "EigenCgSolver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "stopwatch.hpp"
#include "cppoptlib/linesearch/hager_zhang.h"

namespace pwie
{

template <typename Func>
EigenCgSolver<Func>::EigenCgSolver(const Func & func)
  : ISolver<Func>(func)
{
}

template <typename Func>
typename EigenCgSolver<Func>::beta_rule
EigenCgSolver<Func>::to_beta_rule(const std::string & token)
{
  if (token == "prp") return BETA_PRP;
  if (token == "hz")  return BETA_HZ;
  throw std::runtime_error("eigencg: unknown beta rule '" + token + "'");
}

template <typename Func>
typename EigenCgSolver<Func>::search_rule
EigenCgSolver<Func>::to_search_rule(const std::string & token)
{
  if (token == "arc") return SEARCH_ARC;
  if (token == "hz")  return SEARCH_HZ;
  if (token == "awolfe") return SEARCH_AWOLFE;
  throw std::runtime_error("eigencg: unknown line search '" + token + "'");
}

template <typename Func>
typename EigenCgSolver<Func>::step_rule
EigenCgSolver<Func>::to_step_rule(const std::string & token)
{
  if (token == "sp") return STEP_SP;
  if (token == "bb") return STEP_BB;
  throw std::runtime_error("eigencg: unknown step rule '" + token + "'");
}

template <typename Func>
typename EigenCgSolver<Func>::restart_rule
EigenCgSolver<Func>::to_restart_rule(const std::string & token)
{
  if (token == "any")     return RESTART_ANY;
  if (token == "grow")    return RESTART_GROW;
  if (token == "project") return RESTART_PROJECT;
  throw std::runtime_error("eigencg: unknown restart rule '" + token + "'");
}

/*! \brief Adapts our Functor<> to what the vendored hager_zhang.h expects.
 *
 * That file is a verbatim copy of upstream and calls its objective as
 * f = function(x, &g), so the adaptation happens here rather than there.
 * Note it asks for value *and* gradient at every trial point, which is why an
 * HZ search costs more per iteration than the Armijo arc.
 */
template <typename Func>
struct HzFunction
{
  typedef typename Func::Scalar ScalarType;
  typedef typename Func::InputType VectorType;

  const Functor<Func> & _functor;

  HzFunction(const Functor<Func> & functor) : _functor(functor) {}

  ScalarType operator()(const VectorType & x, VectorType * g) const
  {
    if (g) {
      g->resize(x.rows());
      _functor.gradient(x, *g);
    }
    return _functor.f(x);
  }
};

template <typename Func>
void
EigenCgSolver<Func>::freeSet(const InputType & x, const JacobianType & g,
                             MaskType & free) const
{
  /* A variable is in the active set only when it sits on a bound *and* the
   * gradient pushes it further out; a bound the gradient wants to move away
   * from is not binding and stays free.  Iterates come out of clampToBox(), so
   * "on the bound" is exact equality, no tolerance needed. */
  free = !(((x.array() <= _lb.array()) && (g.array() > 0)) ||
           ((x.array() >= _ub.array()) && (g.array() < 0)));
}

template <typename Func>
void
EigenCgSolver<Func>::atBoundSet(const InputType & x, MaskType & atBound) const
{
  // A(x) of Hager & Zhang: on a bound, whatever the gradient wants.  The UA
  // holds these fixed and is allowed to add to them, never to free one.
  atBound = (x.array() <= _lb.array()) || (x.array() >= _ub.array());
}

template <typename Func>
typename EigenCgSolver<Func>::Scalar
EigenCgSolver<Func>::maxFeasibleStep(const InputType & x, const InputType & d) const
{
  Scalar amax = std::numeric_limits<Scalar>::max();
  for (int i = 0; i < x.rows(); i++) {
    Scalar room;
    if (d(i) > 0)
      room = (_ub(i) - x(i)) / d(i);
    else if (d(i) < 0)
      room = (_lb(i) - x(i)) / d(i);
    else
      continue;               // a pinned or motionless variable blocks nothing
    // an infinite bound leaves infinite room, and must not poison the min
    if (std::isfinite(room) && room < amax)
      amax = MAX(room, (Scalar)0);
  }
  return amax;
}

template <typename Func>
bool
EigenCgSolver<Func>::lineSearchAWolfe(const InputType & x, const JacobianType & g,
                                      const InputType & d, Scalar & alpha,
                                      Scalar & fval, InputType & xnew,
                                      JacobianType & gnew) const
{
  using std::abs;
  /* With a postStep projection (--rescale) the point this search evaluates is
   * not the point the solver takes, so a Wolfe point on the box arc is the
   * wrong target and the interval loop grinds looking for one: 84 trials an
   * iteration on iceberg --rescale, against 2.5 without it.  lineSearch()
   * refuses to shrink for the same reason, and it takes this case. */
  if (this->_post)
    return false;

  const Scalar f0 = fval;
  const Scalar delta = (Scalar)_c1, sigma = (Scalar)_c2;

  /* eps_k, the width of the band in which comparing function values is noise
   * and the approximate test takes over.  Hager & Zhang make it eps*C_k with
   * C_k a decaying average of |f| (TOMS 32 (2006) 113, (4.2)), and the decay
   * is the whole point: the band has to follow f down.  A fixed 1e-6|f_0| does
   * not, and it gated the approximate test out of all 4858 trials of an
   * iceberg run -- so that search was a strong Wolfe search wearing this
   * function's name, and sigma had to be driven to 0.01 to pay for it. */
  _awQ = 1 + (Scalar)_awDecay * _awQ;
  _awC += (abs(f0) - _awC) / _awQ;
  const Scalar epsk = (Scalar)_awEps * _awC;

  /* phi'(0) counts only the components the arc can actually move: one held
   * against its bound contributes nothing however large its gradient. */
  Scalar dphi0 = 0;
  for (int i = 0; i < x.rows(); i++) {
    const Scalar di = d(i);
    if (di == 0) continue;
    if ((x(i) <= _lb(i) && di < 0) || (x(i) >= _ub(i) && di > 0)) continue;
    dphi0 += g(i) * di;
  }
  if (!(dphi0 < 0))
    return false;
  _awSearches++;

  InputType xa(x.rows());
  JacobianType ga(x.rows());
  Scalar fa = f0, gda = 0, dphia = 0;
  bool haveD = false;

  auto arcSlope = [&]() {
    Scalar s = 0;
    for (int i = 0; i < x.rows(); i++)
      if (xa(i) > _lb(i) && xa(i) < _ub(i)) s += ga(i) * d(i);
    return s;
  };
  /* Three ways to price a trial.  A gradient costs about four values here, so
   * a trial a value alone can reject never asks for one: value() is the whole
   * cost of a step that overshot, and slope() is only spent where the
   * curvature condition has to be tested. */
  auto value = [&](Scalar a) {
    xa = x + a * d;
    clampToBox(xa);
    gda = g.dot(xa - x);
    fa = fEval(xa);
    haveD = false;
    _awTrials++;
  };
  auto slope = [&]() {
    gEval(xa, ga);
    dphia = arcSlope();
    haveD = true;
  };
  //! both at once, for a trial we expect to accept -- one fused sweep, where
  //! value() then slope() at the same point would be five f-units instead of four
  auto valueSlope = [&](Scalar a) {
    xa = x + a * d;
    clampToBox(xa);
    gda = g.dot(xa - x);
    fa = fgEval(xa, ga);
    dphia = arcSlope();
    haveD = true;
    _awTrials++;
  };
  /*! The two tests of HZ (4.1).  Neither is the *strong* Wolfe condition: the
   * decrease test already rules out the overshoot that |phi'| <= sigma|phi'(0)|
   * was there to catch, and asking for both bought a line minimizer at every
   * iteration and paid for it in trials.  The approximate branch replaces the
   * decrease test -- which is meaningless once f is flat to within eps_k -- by
   * an upper bound on the slope, and is valid only inside that band. */
  auto wolfe = [&]() {
    if (!std::isfinite(fa) || !haveD)
      return false;
    if (fa <= f0 + delta * gda && dphia >= sigma * dphi0)
      return true;
    if (fa > f0 + epsk)
      return false;
    if (dphia >= sigma * dphi0 && dphia <= (2 * delta - 1) * dphi0) {
      _awApprox++;
      return true;
    }
    return false;
  };
  auto commit = [&](Scalar a) {
    alpha = a; fval = fa; xnew = xa; gnew = ga;
  };

  /* The bracket [lo, hi].  lo keeps phi(lo) <= phi(0) + eps_k and phi'(lo) < 0;
   * hi is either a point with phi'(hi) >= 0, or -- and this is what lets a
   * value reject a trial on its own -- a point whose value is above the band,
   * which puts a minimizer below it just as surely. */
  Scalar lo = 0, flo = f0, dlo = dphi0;
  Scalar hi = 0, fhi = 0, dhi = 0;
  bool haveHi = false, hiHasD = false;
  Scalar a = (alpha > 0 && std::isfinite(alpha)) ? alpha : 1;

  /* I1, the QuadStep: one *value* at psi1*a, and if the quadratic through
   * phi(0), phi'(0) and that value is strongly convex, start from its
   * minimizer instead of from a.  This is where asa_cg's 1.36 standalone f
   * per iteration go, and it is what lets its search accept on the first
   * trial: the caller's Shanno-Phua guess carries no curvature along d, so
   * without this the search either accepts a poor point (at sigma 0.9) or
   * spends trials fixing the guess (at sigma 0.01). */
  const Scalar R = (Scalar)_awPsi1 * a;
  value(R);
  if (std::isfinite(fa) && fa <= f0) {
    const Scalar den = 2 * (fa - f0 - dphi0 * R);
    if (den > 0) {                        // strongly convex
      const Scalar q = -dphi0 * R * R / den;
      if (q > 0 && std::isfinite(q))
        a = q;
    }
  }

  /* B0-B3: expand until the bracket closes.  Every trial here needs phi' to
   * decide whether to go on, so every trial here is fused. */
  for (int i = 0; i < _maxExpand && !haveHi; i++) {
    valueSlope(a);
    if (wolfe()) { commit(a); return true; }
    if (!std::isfinite(fa) || fa > f0 + epsk || dphia >= 0) {
      hi = a; fhi = fa; dhi = dphia; haveHi = true;
      hiHasD = std::isfinite(fa) && dphia >= 0;
      break;
    }
    if (xa == x)
      return false;                                  // step underflowed
    /* The next trial by secant on phi' rather than a fixed ratio: both slopes
     * are already paid for, and a factor of 5 makes every overshoot a full
     * evaluation of a point nobody wanted. */
    Scalar next = (Scalar)_awRho * a;
    if (dphia > dlo) {
      const Scalar t = (lo * dphia - a * dlo) / (dphia - dlo);
      if (t > a)
        next = MIN(MAX(t, (Scalar)1.5 * a), (Scalar)_awRho * 2 * a);
    }
    lo = a; flo = fa; dlo = dphia;
    a = next;
  }
  if (!haveHi)
    return false;

  /* L0-L3 with secant2 (S1-S4): two secant steps a pass, the second on the
   * side that moved, so both ends improve rather than one end and a bisection.
   * Flattened into one probe per turn, with `sameSide` alternating the pair
   * the secant is taken over. */
  bool sameSide = false, forceBisect = false, movedHi = false, oldHasD = false;
  Scalar oldA = 0, oldD = 0, width = hi - lo;

  for (int i = 0; i < 40; i++) {
    Scalar c;
    const Scalar w = (Scalar)0.1 * (hi - lo);
    if (forceBisect)
      c = (lo + hi) / 2;
    else if (sameSide && oldHasD && (!movedHi || hiHasD)) {
      const Scalar b = movedHi ? hi : lo, db = movedHi ? dhi : dlo;
      c = (db != oldD) ? (oldA * db - b * oldD) / (db - oldD) : (lo + hi) / 2;
    }
    else if (hiHasD && dhi > dlo)
      c = (lo * dhi - hi * dlo) / (dhi - dlo);        // secant on phi'
    else if (std::isfinite(fhi)) {
      // no slope at hi: the quadratic through phi(lo), phi'(lo), phi(hi)
      const Scalar wid = hi - lo;
      const Scalar den = 2 * (fhi - flo - dlo * wid);
      c = den > 0 ? lo - dlo * wid * wid / den : (lo + hi) / 2;
    }
    else
      c = (lo + hi) / 2;
    c = MAX(lo + w, MIN(hi - w, c));
    if (!(c > lo && c < hi))
      return false;

    // value first; only a trial the value cannot reject is worth a gradient
    value(c);
    if (!std::isfinite(fa) || fa > f0 + epsk) {
      oldA = hi; oldD = dhi; oldHasD = hiHasD;
      hi = c; fhi = fa; dhi = 0; hiHasD = false;
      movedHi = true;
    }
    else {
      slope();
      if (wolfe()) { commit(c); return true; }
      if (xa == x)
        return false;
      if (dphia >= 0) {
        oldA = hi; oldD = dhi; oldHasD = hiHasD;
        hi = c; fhi = fa; dhi = dphia; hiHasD = true;
        movedHi = true;
      }
      else {
        oldA = lo; oldD = dlo; oldHasD = true;
        lo = c; flo = fa; dlo = dphia;
        movedHi = false;
      }
    }
    // L2: an interval that will not shrink gets bisected next turn
    const Scalar now = hi - lo;
    forceBisect = now > (Scalar)_awGamma * width;
    sameSide = !forceBisect && !sameSide;
    width = now;
  }
  return false;
}

template <typename Func>
bool
EigenCgSolver<Func>::lineSearchHz(const InputType & x, const JacobianType & g,
                                  const InputType & d, Scalar & alpha,
                                  Scalar & fval, InputType & xnew,
                                  JacobianType & gnew) const
{
  typedef cppoptlib::solver::linesearch::HagerZhang<HzFunction<Func>, 1> HZ;

  /* The HZ search knows nothing about the box: it walks the ray x + a*d and
   * its bracketing phase expands by a factor of 5 at a time, straight through
   * any face.  So it is only usable where the whole search can stay interior.
   * If the first trial step already leaves the box there is nothing for it to
   * find, and the arc search -- which handles faces properly -- takes over. */
  const Scalar amax = maxFeasibleStep(x, d);
  if (!(amax > alpha))
    return false;

  const HzFunction<Func> fn(_functor);
  InputType xh;
  JacobianType gh;
  Scalar fh = fval;
  const Scalar ah = HZ::Search(x, fval, g, d, fn, alpha, &xh, &fh, &gh);

  /* Reject a Wolfe point outside the box.  The evaluations spent getting there
   * are wasted, which is the price of asking an unconstrained search. */
  if (!(ah > 0) || ah > amax || !(fh < fval))
    return false;

  alpha = ah;
  fval = fh;
  xnew = xh;
  gnew = gh;
  return true;
}

template <typename Func>
bool
EigenCgSolver<Func>::lineSearch(const InputType & x, const JacobianType & g,
                                const InputType & d, Scalar & alpha,
                                Scalar & fval, Scalar fref, InputType & xnew) const
{
  const Scalar f0 = fval;
  if (!(g.dot(d) < 0))          // caller must hand us a descent direction
    return false;

  /* Search along the projection arc x(a) = P(x + a*d) rather than clipping the
   * step at the first box face.  Clipping costs an iteration per variable that
   * wants to saturate, and on these problems most of the rf ends up on the
   * bound, so the arc is worth far more than the extra bookkeeping.  The
   * Armijo test then has to use the true displacement P(x+a*d) - x, which is
   * shorter than a*d wherever the arc has bent.
   *
   * fref is the value the decrease is measured against: f0 for a monotone
   * search, or a nonmonotone reference (GLL) when the caller supplies one. */
  Scalar a = alpha;
  Scalar fa = 0, gda = 0;
  int nback = 0;
  while (true) {
    xnew = x + a * d;
    clampToBox(xnew);
    gda = g.dot(xnew - x);
    if (gda < 0) {
      fa = fEval(xnew);
      if (std::isfinite(fa) && fa <= fref + (Scalar)_c1 * gda)
        break;
    }
    else {
      // the arc has bent so far that it no longer descends: shorten it
      fa = std::numeric_limits<Scalar>::quiet_NaN();
    }
    if (++nback > _maxBacktrack)
      return false;
    /* minimizer of the quadratic through f0, gda and fa, safeguarded into
     * [0.1a, 0.5a] -- much faster than plain halving when the trial step was
     * wrong by orders of magnitude, which it is on iteration 0 */
    Scalar anew = a / 2;
    if (std::isfinite(fa)) {
      const Scalar denom = 2 * (fa - f0 - gda);
      if (denom > 0) {
        const Scalar q = -gda * a / denom;
        if (q > a / 10 && q < a / 2)
          anew = q;
      }
    }
    a = anew;
    if (!(a > 0) || (a * d).template lpNorm<Eigen::Infinity>() == 0)
      return false;           // step has underflowed to nothing
  }

  /* The first guess passing Armijo does not make it good, so when it is
   * accepted outright probe both ways: double while f keeps falling, and if
   * doubling never helped, halve while it keeps falling.  Either way the step
   * lands within a factor of two of the best on the ray whatever scale the
   * guess had, which is what the guess alone cannot supply -- 2|f|/|g.d| asks
   * for a decrease of 2f, an overshoot by f/f* on a cost bounded below at 0.
   *
   * Growing alone used to be enough to accept a step ~4x past the ray minimum
   * on iteration 0.  Under ampangle that is fatal rather than merely slow:
   * |rf| is exp(x), so an amplitude the overlong step slams onto its floor has
   * a gradient of exp(floor) in both its own coordinate and its phase, and
   * never comes back.  35 of 250 died in the first four iterations and the run
   * converged to 8.52 against asa_cg's 0.808; probing both ways kills none.
   *
   * Only when the box is the whole feasible set.  With a postStep projection
   * (--rescale) the point the search evaluated is not the point taken, so the
   * minimum along the box arc is the wrong target: shrinking to it picks
   * interior points where the answer wants the disc boundary (0.82 -> 4.37). */
  if (nback == 0) {
    InputType xtry(x.rows());
    int grew = 0;
    for (int i = 0; i < _maxExpand; i++) {
      const Scalar a2 = 2 * a;
      xtry = x + a2 * d;
      clampToBox(xtry);
      if (xtry == xnew)
        break;                // wholly clamped: a longer step is the same point
      const Scalar gd2 = g.dot(xtry - x);
      if (!(gd2 < 0))
        break;
      const Scalar f2 = fEval(xtry);
      if (!(std::isfinite(f2) && f2 < fa && f2 <= fref + (Scalar)_c1 * gd2))
        break;
      a = a2;
      fa = f2;
      gda = gd2;
      xnew = xtry;
      grew++;
    }
    if (!grew && !this->_post) {
      for (int i = 0; i < _maxExpand; i++) {
        const Scalar a2 = a / 2;
        xtry = x + a2 * d;
        clampToBox(xtry);
        const Scalar gd2 = g.dot(xtry - x);
        if (!(gd2 < 0))
          break;
        const Scalar f2 = fEval(xtry);
        if (!(std::isfinite(f2) && f2 < fa && f2 <= fref + (Scalar)_c1 * gd2))
          break;
        a = a2;
        fa = f2;
        gda = gd2;
        xnew = xtry;
      }
    }
  }

  alpha = a;
  fval = fa;
  return true;
}

/*! \brief The f-delta safety net: has the run stopped moving at all?
 *
 * Both modes stop on the projected gradient, which is what asa_cg stops on;
 * this is only a net against a run that is genuinely stuck, and it is meant to
 * be hard to trip.  Two things make it so:
 *
 * - It asks for progress over a long window of iterations, not from one
 *   iteration to the next.  A CG makes several negligible steps in a row while
 *   it turns a corner, and it was five of those that ended every asa_cg2 run:
 *   18 of 18 measured runs stopped here and none on the gradient, which is how
 *   bibop_kobzar2004 ampangle finished at -0.9802 against asa_cg's -0.9968.
 *   The window has to be long because these problems crawl legitimately: at a
 *   window of 20, bibop still quit at -0.9564 where the same run left to
 *   itself reached -0.9968.  A run that is really dead is caught by the
 *   null-step test in the line search instead, and much sooner.
 * - It measures the improvement relative to |f|, for the same reason --gtol is
 *   a fraction of the projected gradient at x0 (ac35342): a tolerance in the
 *   objective's own units silently changes meaning whenever the objective or
 *   the variables are rescaled.
 *
 * settings.tol is the fraction, from --ftol.  Returns true, once per window,
 * when the window improved the best value by less than tol*|f|.
 */
template <typename Func>
bool
EigenCgSolver<Func>::stalled(Scalar fbest, Scalar & fwin, int & left) const
{
  if (--left > 0)
    return false;
  left = _stallWindow;
  // |f| == 0 is as good as converged, and has no scale to be relative to
  const Scalar scale = std::abs(fwin) > 0 ? std::abs(fwin) : (Scalar)1;
  const bool stuck = !(fwin - fbest > settings.tol * scale);
  fwin = fbest;
  return stuck;
}

template <typename Func>
void
EigenCgSolver<Func>::internalSolve(InputType & x0)
{
  const int DIM = x0.rows();

  _lb = _functor.getLowerBound();
  _ub = _functor.getUpperBound();
  if (_lb.size() == 0) _lb = InputType::Constant(DIM, std::numeric_limits<Scalar>::lowest());
  if (_ub.size() == 0) _ub = InputType::Constant(DIM, std::numeric_limits<Scalar>::max());

  Assert(_lb.rows() == DIM, "lower bound size incorrect");
  Assert(_ub.rows() == DIM, "upper bound size incorrect");

  InputType x = x0;
  clampToBox(x);              // the seed itself may be outside the box
  this->postStep(x);
  clampToBox(x);

  _ngpaIters = _uaIters = _uEmpty = _nf = _ng = 0;
  _awTrials = _awApprox = _awSearches = 0;
  _awQ = _awC = 0;
  settings.numIters = 0;

  if (mode == MODE_PCG)
    solveSimple(x);
  else
    solveAsa(x);

  x0 = x;
}

/*! \brief Single-phase projected CG.  See the class comment for what this is.
 */
template <typename Func>
void
EigenCgSolver<Func>::solveSimple(InputType & x)
{
  const int DIM = x.rows();
  using std::abs;

  JacobianType g(DIM), pg(DIM), pg_old(DIM), gnew(DIM);
  InputType d = InputType::Zero(DIM);
  InputType xnew(DIM), xbest(DIM), s(DIM);
  MaskType free(DIM), free_old(DIM);
  std::vector<Scalar> fhist;      // for the nonmonotone reference, if asked

  Scalar f = fgEval(x, g);
  freeSet(x, g, free);
  free_old = free;

  Scalar fbest = f;
  xbest = x;
  Scalar alpha = 0;
  Scalar gd_old = 0;
  Scalar sy = 0, ss = 0;      // for the BB step
  bool restart = true;
  Scalar fwin = f;            // f-delta window state, see stalled()
  int stallLeft = _stallWindow;
  size_t iter = 0;
  Stopwatch<> stopwatch;

  const char * why = "iteration limit";
  Scalar pgnorm = 0;

  while (iter < settings.maxIter) {

    /* KKT measure, ||P(x - g) - x||_inf -- the same quantity asa_cg tests
     * grad_tol against, so the two solvers stop on the same footing */
    pgnorm =
      ((x - g).cwiseMax(_lb).cwiseMin(_ub) - x).template lpNorm<Eigen::Infinity>();
    if (pgnorm <= settings.gradTol) {
      why = "projected gradient below gradTol";
      break;
    }
    if (f <= objectiveLimit()) {
      why = "objective below objectiveLimit";
      break;
    }

    pg.array() = free.select(g.array(), Scalar(0));
    if (pg.template lpNorm<Eigen::Infinity>() == 0) {
      why = "all variables pinned at a bound";
      break;                  // nowhere left to go
    }

    /* What a change in the active set does to the accumulated conjugacy.  The
     * measured churn on these problems is severe -- on bebop_pp45 the set
     * moves in 85% of iterations -- so RESTART_ANY throws the direction away
     * almost every iteration; the other two rules exist to measure that. */
    const int changed = (free != free_old).count();
    if (changed) {
      if (restartRule == RESTART_ANY)
        restart = true;
      else if (restartRule == RESTART_GROW && changed > 1)
        restart = true;
    }

    if (restart) {
      d = -pg;
    }
    else {
      Scalar beta = 0;
      if (betaRule == BETA_HZ) {
        /* Hager-Zhang beta_N with their eta truncation (TOMS 32 (2006) 113).
         * Paired with SEARCH_HZ this is CG_DESCENT, ie what asa_cg runs
         * between its projection steps. */
        const JacobianType y = pg - pg_old;
        const Scalar dy = d.dot(y);
        if (dy > 0) {
          beta = (y - 2 * d * (y.squaredNorm() / dy)).dot(pg) / dy;
          const Scalar eta = -1 / (d.norm() * MIN((Scalar)0.01, pg_old.norm()));
          beta = MAX(beta, eta);
        }
      }
      else {
        // Polak-Ribiere+, plus Powell's restart when successive projected
        // gradients stop looking orthogonal.
        const Scalar denom = pg_old.squaredNorm();
        beta = denom > 0 ? pg.dot(pg - pg_old) / denom : 0;
        beta = MAX(beta, (Scalar)0);
        if (abs(pg.dot(pg_old)) > powellRestart * pg.squaredNorm())
          beta = 0;
      }
      d = -pg + beta * d;
      d.array() = free.select(d.array(), Scalar(0)); // never step a pinned variable
      if (d.dot(pg) >= 0) {   // not a descent direction after all
        d = -pg;
        restart = true;
      }
    }

    const Scalar gd = g.dot(d);
    /* Initial trial step.  STEP_SP is Shanno-Phua scaling of the previous
     * accepted step, the standard CG guess; STEP_BB is the Barzilai-Borwein
     * step, which is what the NGPA uses.  On the first iteration nothing is
     * known about the scale of x, so ask for a decrease of order |f| and let
     * the search expand or backtrack from there. */
    Scalar step = 0;
    if (stepRule == STEP_BB && sy > 0)
      step = ss / sy;
    else if (iter > 0 && alpha > 0 && gd_old != 0)
      step = alpha * gd_old / gd;
    if (!(step > 0) || !std::isfinite(step))
      step = abs(f) > 0 ? 2 * abs(f) / abs(gd) : 1 / d.template lpNorm<Eigen::Infinity>();
    if (!std::isfinite(step) || step <= 0)
      step = 1 / d.template lpNorm<Eigen::Infinity>();

    // nonmonotone reference: max of the last `memory` accepted values (GLL)
    Scalar fref = f;
    if (memory > 1 && !fhist.empty())
      fref = *std::max_element(fhist.begin(), fhist.end());

    const Scalar f_old = f;
    Scalar f_try = f;
    /* Interior steps can get the Wolfe search, steps that reach a face get the
     * arc: the CG's conjugacy assumes a Wolfe point, but only the arc can
     * cross a face, and on these problems most of the rf saturates. */
    bool haveGrad = false;
    bool searched = true;
    if (searchRule == SEARCH_AWOLFE &&
        lineSearchAWolfe(x, g, d, step, f_try, xnew, gnew))
      haveGrad = true;
    else if (searchRule == SEARCH_HZ &&
        lineSearchHz(x, g, d, step, f_try, xnew, gnew))
      haveGrad = true;
    else {
      f_try = f;
      searched = lineSearch(x, g, d, step, f_try, fref, xnew);
    }
    // a step that moves neither x nor f is not progress, whatever the search
    // says; see the same test in the UA
    if (searched && (xnew == x || !(f_try < fref)))
      searched = false;
    if (!searched) {
      if (!restart) {
        // a stale CG direction is the usual culprit; retry from -pg
        restart = true;
        continue;
      }
      why = "line search failed";
      break;
    }

    const InputType xprev = x;
    x = xnew;
    f = f_try;
    this->postStep(x);
    clampToBox(x);
    /* postStep may have moved x off the point the line search evaluated (only
     * when the functor has a projection), so f -- and any gradient the search
     * handed back -- have to be discarded and re-measured there.  Both at
     * once: that is one ensemble sweep rather than two. */
    const bool moved = (x != xnew);
    if (moved)
      haveGrad = false;
    s = x - xprev;            // the step actually taken, for the BB length

    pg_old = pg;
    if (haveGrad) {
      sy = s.dot(gnew - g);
      g = gnew;
    }
    else {
      const JacobianType g_old = g;
      if (moved)
        f = fgEval(x, g);
      else
        gEval(x, g);
      sy = s.dot(g - g_old);
    }
    ss = s.squaredNorm();
    free_old = free;
    freeSet(x, g, free);
    alpha = step;
    gd_old = gd;
    restart = false;

    iter++;
    settings.numIters = iter;

    if (memory > 1) {
      fhist.push_back(f);
      if ((int)fhist.size() > memory)
        fhist.erase(fhist.begin());
    }
    if (f < fbest) {
      fbest = f;
      xbest = x;
    }

    // the safety net; the stop that matters is the gradient test above.  It
    // is measured on the best value seen, so a nonmonotone search that is
    // genuinely exploring does not look stuck.
    if (stalled(fbest, fwin, stallLeft)) {
      why = "no improvement above tol";
      break;
    }

    InputType dir = alpha * d;
    if (this->checkConverged(iter, x, alpha, dir, g)) {
      why = "functor says converged";
      break;
    }

    if (settings.verbosity > 0) {
      stopwatch.stop();
      std::cout << "iteration " << iter << " " << f
                << " dt=" << stopwatch.elapsed()/1e3
                << " df=" << (f_old - f)
                << " step=" << alpha
                << " nfree=" << free.count()
                // ||pg|| separates a bad direction from a flat face when the
                // cost stops moving; nothing else in the line tells them apart
                << " pg=" << pg.norm()
                << std::endl;
      stopwatch.start();
    }
  }

  if (fbest < f)
    x = xbest;                // a nonmonotone search can end above its best

  settings.stopReason = why;
  if (settings.verbosity > 0)
    std::cout << "eigencg[simple]: stopped after " << settings.numIters
              << " iterations, f=" << fbest << ": " << why
              // the quantity the run wanted to stop on, against the tolerance
              // it wanted to beat: the only way to tell a converged run from
              // one the budget or the f-delta net cut off mid-descent
              << " (pg=" << pgnorm << " gradTol=" << settings.gradTol
              << " nf=" << _nf << " ng=" << _ng << ")\n"
              << "  awolfe: searches=" << _awSearches
              << " trials=" << _awTrials
              << " approx-accepts=" << _awApprox << "\n";
}

/*! \brief The Hager-Zhang active set algorithm, ASA.
 *
 * Structure and every constant here come from W. W. Hager and H. Zhang, SIAM
 * J. Optim. 17 (2006) 526-557: the ASA of their Figure 3.1, the NGPA of their
 * section 2, and the CBB initial step (I0-I4) and adaptive reference value
 * (R0-R4) of their appendix.  Parameter values are the ones they used in
 * section 6.  Their UA is CG_DESCENT; ours is the same CG machinery
 * MODE_PCG uses, run on the face NGPA hands over.
 */
template <typename Func>
void
EigenCgSolver<Func>::solveAsa(InputType & x)
{
  const int DIM = x.rows();
  using std::abs;

  // NGPA (section 6): alpha_min = 1e-20, alpha_max = 1e+20, eta = .5,
  // delta = 1e-4, M = 8.  ASA: mu = .1, rho = .5, n1 = 2, n2 = 1.
  // CBB (appendix): theta = .975, L = 3, A = 40, m = 4, g1 = M/L, g2 = A/M.
  const Scalar eta = 0.5, delta = (Scalar)_c1;
  const Scalar alphaMin = 1e-20, alphaMax = 1e20;
  const int Mmem = 8;
  const Scalar theta = 0.975;
  const int Lref = 3, Aref = 40, mcbb = 4;
  const Scalar gamma1 = Scalar(Mmem) / Lref, gamma2 = Scalar(Aref) / Mmem;
  const Scalar rho = 0.5;
  const int n1 = 2, n2 = 1;
  const Scalar Ualpha = 0.5, Ubeta = 1.5;
  Scalar mu = 0.1;

  /* U(x) decides everything below and is empty at every iterate here, so ASA
   * runs as its UA alone.  The class comment says why, and why that is not the
   * reason this trails asa_cg -- forcing the phases to alternate is measurably
   * worse.  Do not "fix" the switching rule without re-reading it. */

  // d^alpha and the norms
  auto projDirS = [&](const InputType & xx, const JacobianType & gg,
                      Scalar alpha, InputType & dd) {
    dd = (xx.array() - alpha * gg.array())
      .max(_lb.array()).min(_ub.array()) - xx.array();
  };
  auto normS = [&](const InputType & v) { return v.norm(); };
  auto normSg = [&](const JacobianType & v) { return v.norm(); };

  InputType d(DIM), d1(DIM), xnew(DIM), s(DIM), xbest(DIM), dcg(DIM);
  JacobianType g(DIM), gnew(DIM), yv(DIM), gI(DIM), pg(DIM), pg_old(DIM);
  MaskType atBound(DIM), atBound_prev(DIM), freeMask(DIM);
  std::vector<Scalar> fhist;

  Scalar f = fgEval(x, g);
  fhist.push_back(f);
  Scalar fbest = f;
  xbest = x;

  atBoundSet(x, atBound);
  atBound_prev = atBound;
  int nActivePrev = atBound.count();
  int sameActive = 0;         // consecutive iterations with A(x) unchanged

  // I0 / R0
  projDirS(x, g, 1, d1);
  Scalar abar = 1 / MAX(normS(d1), std::numeric_limits<Scalar>::min());
  abar = MIN(alphaMax, MAX(alphaMin, abar));
  int jcbb = 0;
  bool flag = true;
  bool firstNgpa = true;
  Scalar fr = f, fr_prev = f, fmin = f, fmaxmin = f;
  int acount = 0, lcount = 0;

  // UA (CG) state
  bool inUA = false;
  bool uaRestart = true;
  Scalar uaAlpha = 0, uaGdOld = 0;

  size_t iter = 0;
  Scalar fwin = f;            // f-delta window state, see stalled()
  int stallLeft = _stallWindow;
  Stopwatch<> stopwatch;
  const char * why = "iteration limit";
  Scalar pgInf = 0;

  /* "Restarting the NGPA" means x0 is the current iterate, so I0 and R0 run
   * again: fresh stepsize and cycle, and a reference value that has forgotten
   * the UA's iterates. */
  auto restartNgpa = [&]() {
    projDirS(x, g, 1, d1);
    abar = 1 / MAX(normS(d1), std::numeric_limits<Scalar>::min());
    abar = MIN(alphaMax, MAX(alphaMin, abar));
    jcbb = 0;
    flag = true;
    firstNgpa = true;
    fr = fr_prev = fmin = fmaxmin = f;
    acount = lcount = 0;
    fhist.assign(1, f);
  };

  while (iter < settings.maxIter) {

    /* Stop on the unpreconditioned KKT measure whatever the mode, so every
     * mode and asa_cg stop on the same quantity. */
    projDir(x, g, 1, d1);
    pgInf = d1.template lpNorm<Eigen::Infinity>();
    if (pgInf <= settings.gradTol) {
      why = "projected gradient below gradTol";
      break;
    }
    projDirS(x, g, 1, d1);    // the algorithm's own d^1, in scaled units
    if (f <= objectiveLimit()) {
      why = "objective below objectiveLimit";
      break;
    }

    const Scalar f_old = f;
    bool stepOk = false;
    Scalar alphaLS = 1;       // the line search step actually taken

    if (!inUA) {
      /* ---- NGPA, one iteration (section 2, steps 1-5) ---- */
      // I0: flag starts set at k = 0 only; every later iteration clears it, and
      // I1/I3 below set it again.  Without this reset flag stays set forever,
      // I4 fires every iteration, and the cycle in cyclic BB never happens.
      if (!firstNgpa)
        flag = false;
      firstNgpa = false;
      bool truncated = false;
      {
        // step 1: d_k = P(x_k - abar_k g_k) - x_k
        projDirS(x, g, abar, d);
        // I1: did the box truncate the step in any component?
        for (int i = 0; i < DIM && !truncated; i++) {
          const Scalar full = abar * abs(g(i));
          const Scalar got = abs(d(i));
          if (got > 0 && got < full)
            truncated = true;
        }
      }
      const Scalar gd = g.dot(d);
      if (!(gd < 0)) {
        why = "gradient projection direction is not descent";
        break;
      }

      // R1: update the reference value f^r
      const Scalar fmax = *std::max_element(fhist.begin(), fhist.end());
      if (lcount == Lref) {
        lcount = 0;
        const Scalar num = fmax - fmin, den = fmaxmin - fmin;
        fr = (den > 0 && num / den >= gamma1) ? fmaxmin : fmax;
      }
      else if (acount > Aref) {
        const Scalar den = fmax - f;
        fr = (fmax > f && den > 0 && (fr_prev - f) / den >= gamma2) ? fmax : fr_prev;
      }
      else
        fr = fr_prev;
      // R2: the value steps 3 and 4 measure the decrease against
      const Scalar fR = (jcbb == 0) ? fr : MIN(fmax, fr);

      // steps 3 and 4: alpha = 1, else the first eta^j that satisfies Armijo
      Scalar alpha = 1;
      Scalar ftry = 0;
      int nb = 0;
      while (true) {
        xnew = x + alpha * d;
        clampToBox(xnew);       // x + alpha*d is already feasible for alpha<=1
        ftry = fEval(xnew);
        if (std::isfinite(ftry) && ftry <= fR + delta * alpha * gd)
          break;
        if (++nb > _maxBacktrack)
          break;
        alpha *= eta;
      }
      if (nb > _maxBacktrack) {
        why = "NGPA line search failed";
        break;
      }
      alphaLS = alpha;

      // step 5, then our projection hook
      const InputType xprev = x;
      const JacobianType gprev = g;
      x = xnew;
      f = ftry;
      this->postStep(x);
      clampToBox(x);
      // both at the same point, so one sweep; see NGPA's twin below
      if (x != xnew)
        f = fgEval(x, g);
      else
        gEval(x, g);
      s = x - xprev;
      yv = g - gprev;

      // I2, I3, I4: the initial stepsize for the next NGPA iteration
      if (alphaLS == 1) jcbb++;
      else flag = true;
      if (truncated) flag = true;
      const Scalar sy = s.dot(yv);
      const Scalar sn = s.norm(), yn = yv.norm();
      if (jcbb >= mcbb || flag ||
          (sn > 0 && yn > 0 && sy / (sn * yn) >= theta)) {
        if (sy <= 0) {
          if (jcbb >= (int)(1.5 * mcbb)) {
            const Scalar t =
              MIN(x.template lpNorm<Eigen::Infinity>(), (Scalar)1) /
              MAX(d1.template lpNorm<Eigen::Infinity>(),
                  std::numeric_limits<Scalar>::min());
            abar = MIN(alphaMax, MAX(t, alphaLS));
            jcbb = 0;
          }
          // else keep the current abar and the cycle going
        }
        else {
          // BB step, s'.s / s'.y
          abar = MIN(alphaMax, MAX(alphaMin, s.squaredNorm() / sy));
          jcbb = 0;
        }
      }
      // R3
      if (alphaLS < 1) acount = 0;
      else acount++;
      _ngpaIters++;
      stepOk = true;
    }
    else {
      /* ---- UA: conjugate gradients on the face NGPA identified ---- */
      freeMask = !atBound;
      pg.array() = freeMask.select(g.array(), Scalar(0));
      if (pg.template lpNorm<Eigen::Infinity>() == 0) {
        // nothing free to move: hand back to the NGPA, which can leave a face
        inUA = false;
        uaRestart = true;
        restartNgpa();
        continue;
      }

      if (uaRestart) {
        dcg = -pg;
      }
      else {
        Scalar beta = 0;
        if (betaRule == BETA_HZ) {
          const JacobianType y = pg - pg_old;
          const Scalar dy = dcg.dot(y);
          if (dy > 0) {
            beta = (y - 2 * dcg * (y.squaredNorm() / dy)).dot(pg) / dy;
            const Scalar etaHZ = -1 / (dcg.norm() * MIN((Scalar)0.01, pg_old.norm()));
            beta = MAX(beta, etaHZ);
          }
        }
        else {
          const Scalar den = pg_old.squaredNorm();
          beta = den > 0 ? pg.dot(pg - pg_old) / den : 0;
          beta = MAX(beta, (Scalar)0);
          if (abs(pg.dot(pg_old)) > powellRestart * pg.squaredNorm())
            beta = 0;
        }
        dcg = -pg + beta * dcg;
        dcg.array() = freeMask.select(dcg.array(), Scalar(0));
        if (dcg.dot(pg) >= 0) {
          dcg = -pg;
          uaRestart = true;
        }
      }

      const Scalar gd = g.dot(dcg);
      Scalar step = 0;
      if (!uaRestart && uaAlpha > 0 && uaGdOld != 0)
        step = uaAlpha * uaGdOld / gd;
      if (!(step > 0) || !std::isfinite(step))
        step = abs(f) > 0 ? 2 * abs(f) / abs(gd)
                          : 1 / dcg.template lpNorm<Eigen::Infinity>();

      /* U1 wants the UA monotone, so the reference is f itself.  U4 wants a
       * Wolfe step whenever the UA is started; we get one from the HZ search
       * when the step stays interior, and otherwise fall back to the Armijo
       * arc, which is a documented deviation from the paper. */
      Scalar ftry = f;
      bool haveGrad = false;
      bool searched = true;
      if (searchRule == SEARCH_AWOLFE &&
          lineSearchAWolfe(x, g, dcg, step, ftry, xnew, gnew))
        haveGrad = true;
      else if (searchRule == SEARCH_HZ &&
          lineSearchHz(x, g, dcg, step, ftry, xnew, gnew))
        haveGrad = true;
      else {
        ftry = f;
        searched = lineSearch(x, g, dcg, step, ftry, f, xnew);
      }
      /* A search that reports success with a step so short that neither x nor
       * f moves is not progress, and has to be caught here: it leaves the
       * search happy and the iterate exactly where it was, so the UA spins on
       * the same face forever taking null steps.  That is what burbop_ur90
       * seed 1 did from iteration 1084 -- df exactly 0 and ||d^1||_inf frozen
       * at 3.8e-4, 7 orders above gradTol -- until the f-delta net ended it
       * 3.5e-3 short of asa_cg.  A face the UA cannot improve is the paper's
       * signal to hand back to the NGPA, which is the phase allowed to leave
       * it. */
      if (searched && (xnew == x || !(ftry < f)))
        searched = false;
      if (!searched) {
        if (!uaRestart) {
          uaRestart = true;   // stale direction; retry from -pg
          continue;
        }
        // the face is as good as we can make it: let the NGPA move the set
        inUA = false;
        uaRestart = true;
        restartNgpa();
        continue;
      }
      alphaLS = step;

      const InputType xprev = x;
      x = xnew;
      f = ftry;
      this->postStep(x);
      clampToBox(x);
      const bool moved = (x != xnew);
      if (moved)
        haveGrad = false;
      pg_old = pg;
      if (haveGrad)
        g = gnew;
      else if (moved)
        f = fgEval(x, g);     // one sweep for the pair, see above
      else
        gEval(x, g);
      s = x - xprev;
      uaAlpha = step;
      uaGdOld = gd;
      uaRestart = false;
      _uaIters++;
      stepOk = true;
    }

    if (!stepOk)
      break;

    iter++;
    settings.numIters = iter;

    // f^max memory (2.3), and R4
    fhist.push_back(f);
    if ((int)fhist.size() > Mmem)
      fhist.erase(fhist.begin());
    if (f < fmin) {           // Delta is the float separation, ie strictly less
      fmaxmin = fmin = f;
      lcount = 0;
    }
    else {
      lcount++;
      fmaxmin = MAX(fmaxmin, f);
    }
    fr_prev = fr;

    if (f < fbest) {
      fbest = f;
      xbest = x;
    }

    /* ---- the branching rules of Figure 3.1, evaluated at the new iterate ---- */
    projDirS(x, g, 1, d1);
    const Scalar d1nrm = normS(d1);
    atBound_prev = atBound;
    atBoundSet(x, atBound);
    const int nActive = atBound.count();
    sameActive = (atBound == atBound_prev).all() ? sameActive + 1 : 0;

    gI.array() = atBound.select(Scalar(0), g.array());
    const Scalar gInrm = normSg(gI);

    /* U(x) = {i : |g_i| >= ||d^1||^alpha and dist(x_i, bound) >= ||d^1||^beta}
     * with alpha = 1/2, beta = 3/2.  The paper writes x_i for the distance to
     * the bound, having set l = 0, u = infinity. */
    const Scalar uThreshG = std::pow(d1nrm, Ualpha);
    const Scalar uThreshX = std::pow(d1nrm, Ubeta);
    bool Uempty = true;
    for (int i = 0; i < DIM && Uempty; i++) {
      if (abs(g(i)) >= uThreshG &&
          MIN(x(i) - _lb(i), _ub(i) - x(i)) >= uThreshX)
        Uempty = false;
    }
    if (Uempty)
      _uEmpty++;

    if (!inUA) {
      // step 1a / 1b
      if (Uempty) {
        if (gInrm < mu * d1nrm)
          mu *= rho;
        else {
          inUA = true;
          uaRestart = true;
        }
      }
      else if (sameActive >= n1) {
        if (gInrm >= mu * d1nrm) {
          inUA = true;
          uaRestart = true;
        }
      }
    }
    else {
      // step 2a / 2b
      if (gInrm < mu * d1nrm) {
        inUA = false;         // subproblem solved: back to the NGPA
        uaRestart = true;
        restartNgpa();
      }
      else if (nActivePrev < nActive) {
        if (Uempty || nActive > nActivePrev + n2)
          uaRestart = true;   // restart the UA at x_k
        else {
          inUA = false;
          uaRestart = true;
          restartNgpa();
        }
      }
    }
    nActivePrev = nActive;

    // the safety net; the stop that matters is the gradient test above
    if (stalled(fbest, fwin, stallLeft)) {
      why = "no improvement above tol";
      break;
    }

    InputType dir = alphaLS * (inUA ? dcg : d);
    if (this->checkConverged(iter, x, alphaLS, dir, g)) {
      why = "functor says converged";
      break;
    }

    if (settings.verbosity > 0) {
      stopwatch.stop();
      std::cout << "iteration " << iter << " " << f
                << " dt=" << stopwatch.elapsed()/1e3
                << " df=" << (f_old - f)
                << " step=" << alphaLS
                << " phase=" << (inUA ? "UA" : "NGPA")
                << " nact=" << nActive
                // the quantity the run stops on, so a crawling cost can be
                // told from one that is simply finished
                << " pg=" << pgInf
                << " abar=" << abar
                << " U=" << (Uempty ? "0" : "1") << std::endl;
      stopwatch.start();
    }
  }

  if (fbest < f)
    x = xbest;                // the NGPA is nonmonotone by design

  settings.stopReason = why;
  if (settings.verbosity > 0)
    std::cout << "eigencg[asa]: stopped after " << settings.numIters
              << " iterations, f=" << fbest << ": " << why
              // see the note in solveSimple's summary
              << " (pg=" << pgInf << " gradTol=" << settings.gradTol
              << " ngpa=" << _ngpaIters << " ua=" << _uaIters
              << " Uempty=" << _uEmpty
              << " nf=" << _nf << " ng=" << _ng << ")\n"
              << "  awolfe: searches=" << _awSearches
              << " trials=" << _awTrials
              << " approx-accepts=" << _awApprox << "\n";
}

}

/* namespace pwie */
