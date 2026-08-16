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

#ifndef EIGENCGSOLVER_H_
#define EIGENCGSOLVER_H_
#include <string>
#include <vector>
#include "pcgcompat.hpp"
namespace pwie
{

/*! \brief Box-constrained minimization, native Eigen: ASA and a simpler CG.
 *
 * Solves  min f(x)  s.t.  lo <= x <= hi.  Two modes:
 *
 * MODE_ASA is the active set algorithm of W. W. Hager and H. Zhang, "A new
 * active set algorithm for box constrained optimization", SIAM J. Optim. 17
 * (2006) 526-557.  It alternates two phases under the switching rules of their
 * Figure 3.1: NGPA, a nonmonotone gradient projection phase using cyclic
 * Barzilai-Borwein steps (their appendix, I0-I4) with the adaptive reference
 * value (R0-R4), identifies the active constraints; the UA then runs
 * conjugate gradients on the face NGPA found.  The branch is decided by the
 * undecided index set U(x) and by whether ||g_I|| has fallen below mu||d^1||.
 *
 * MODE_PCG is the single-phase projected CG this class had first: active
 * set from the sign of the gradient, CG on the free variables, and a
 * restartRule saying what an active-set change does to the direction.
 *
 * On the pulse problems in example/, ASA's two phases never interleave.  The
 * undecided set
 *
 *     U(x) = {i : |g_i| >= ||d^1(x)||^(1/2) and dist_i >= ||d^1(x)||^(3/2)}
 *
 * is empty at every iterate of every run, so step 1a goes to the UA on
 * iteration 1 and only step 2a, ||g_I|| < mu||d^1||, could come back.  Two
 * reasons, both measured on bebop_pp45 --ctype ampangle:
 *
 * - The gradient test cannot be met once ||d^1|| < 1.  For a free component
 *   d^1_i = -g_i, so |g_i| <= ||d^1||, and ||d^1|| < ||d^1||^(1/2) there.
 *   That is arithmetic, not a property of the problem: it holds from about
 *   iteration 10 on.  The only components that do clear the threshold are the
 *   fixed ones (ampangle pins 250 dt variables at lo == hi, |g| ~ 1e6), and
 *   they have dist_i = 0, so they fail the second test.  The powers are the
 *   paper's, written for a model box with x, f and g all order 1.
 * - Step 2a cannot fire either.  d^1 and g_I differ only in the components
 *   that sit on a bound with the gradient pointing back inward, and here
 *   there are 1 to 6 of them out of ~490, contributing ~1e-5: measured
 *   ||g_I||/||d^1|| stays within 0.5% of 1 for the whole run, against
 *   mu = 0.1.
 *
 * So MODE_ASA is its UA on a face that only grows (250 active at the start,
 * 493 of 750 at iteration 2000).  That is NOT why it trails asa_cg, and the
 * obvious repair is a trap -- forcing the phases to alternate, by ignoring U
 * and switching on the active set alone, makes it monotonically worse:
 * ngpa = 499 gives 4.4 and ngpa = 651 gives 13.3, against 2.05 for the
 * ngpa = 1 default and 13.5 for MODE_PCG.  The optimum here is bang-bang
 * (asa_cg's own answer has 245 of 250 amplitudes hard against b1max, ours
 * 240), so pinning a saturated variable is right and re-deciding it wastes
 * the budget.  U(x) being empty is protecting this mode, not crippling it.
 *
 * What is left is that asa_cg reaches 0.96 where this reaches 2.05 at equal
 * function count, with the same shape of answer.  Its UA is CG_DESCENT with
 * an approximate Wolfe line search (~1.9 f and 1.05 g per iteration); ours is
 * PRP+ over an Armijo arc that backtracks about twice per iteration (~2.9 f,
 * 1.0 g).  Tried and measured worse, so not in the tree: beta_HZ (2.28),
 * the vendored HZ search, which mostly refuses because the step leaves the
 * box (2.70), a Wolfe curvature test on the arc (3.11 -- the existing
 * doubling loop is a better crude line minimizer than sigma = 0.9), and
 * three initial-step rules (no effect, the backtrack count barely moves).
 * The remaining gap is line search quality inside the UA.
 *
 * Earlier and also dropped: nondimensionalizing the U thresholds by an rms
 * box width made U never empty, so it stayed in NGPA forever, ~11% worse; a
 * per-variable diagonal preconditioner (P-ASA, section 6) left U empty.
 *
 * The verbose summary prints ngpa/ua/Uempty and the f and g counts, which is
 * how to check any of this on a new problem before concluding from a mode.
 *
 * This class exists at all because AsaCgSolver's iteration loop lives inside a
 * third-party C library with no per-iteration callback, so it cannot call
 * ISolver::postStep().  A parameterization whose feasible set is not a box
 * (cartesian --rescale: |rf| <= b1max is a disc) needs that hook.
 *
 * Written from the two published papers.  No code was taken from, and no part
 * of, the GPL asa_cg.c that AsaCgSolver wraps was read.  The vendored
 * hager_zhang.h is MIT, from upstream CppNumericalSolvers.
 */
template <typename Func>
class EigenCgSolver : public ISolver<Func>
{
  typedef typename Func::Scalar Scalar;
  typedef typename Func::InputType InputType;
  typedef typename Func::JacobianType JacobianType;
  typedef typename ISolver<Func>::HessianType HessianType;
  typedef Eigen::Array<bool, Eigen::Dynamic, 1> MaskType;

  InputType _ub;
  InputType _lb;
  using ISolver<Func>::settings;
  using ISolver<Func>::_functor;

public:

  //! which algorithm runs
  typedef enum {
    MODE_PCG,       //!< single-phase projected CG (--omethod pcg)
    MODE_ASA,       //!< Hager-Zhang NGPA + UA, exactly as printed (--omethod asa_cg2)
  } solve_mode;

  //! which conjugacy formula the CG uses
  typedef enum {
    BETA_PRP,       //!< Polak-Ribiere+, restarted on Powell's test
    BETA_HZ,        //!< Hager-Zhang beta_N; with SEARCH_HZ this is CG_DESCENT
  } beta_rule;

  //! which line search a CG step uses while it stays inside the box
  typedef enum {
    SEARCH_ARC,     //!< projection-arc Armijo everywhere
    SEARCH_HZ,      //!< Hager-Zhang Wolfe in the interior, arc on the faces
  } search_rule;

  //! how the first trial step of a CG line search is guessed
  typedef enum {
    STEP_SP,        //!< Shanno-Phua scaling of the previous step
    STEP_BB,        //!< Barzilai-Borwein, s's/s'y
  } step_rule;

  //! what a change in the active set does to the CG direction
  typedef enum {
    RESTART_ANY,    //!< restart on any change
    RESTART_GROW,   //!< restart only when the set grows by more than one
    RESTART_PROJECT,//!< never restart for this reason; reproject the direction
  } restart_rule;

  /*! Solver knobs, set before solve().  The defaults are the ones that
   * measured best over example/, on four starting-point conditions with five
   * seeds each, compared per-start against the same alternative:
   *
   * - RESTART_PROJECT beat RESTART_ANY on all six problems.  The active set
   *   here moves in most iterations, so restarting on every change threw the
   *   conjugacy away almost every step; reprojecting the old direction into
   *   the new free set instead keeps it, and on bebop_pp45 it also halves the
   *   iterations needed.
   * - STEP_SP beat STEP_BB (bebop 1 win in 10), and monotone beat memory = 8
   *   (bebop 1 in 10) -- the Barzilai-Borwein step and the nonmonotone
   *   reference belong to the gradient projection phase they were designed
   *   for, not to a CG line search.  memory = 8 does help sburbop_ur180
   *   (7 of 10), so it is worth trying per problem.
   * - MODE_ASA is better than MODE_PCG on saturation_relaxation (-7%)
   *   and sburbop_ur180 (-4%) and worse on hsn_inversion, so no mode
   *   dominates; MODE_PCG is the default for being the most consistent.
   */
  solve_mode mode = MODE_PCG;
  beta_rule betaRule = BETA_PRP;
  search_rule searchRule = SEARCH_ARC;
  step_rule stepRule = STEP_SP;
  restart_rule restartRule = RESTART_PROJECT;
  //! nonmonotone memory for MODE_PCG's line search; 1 is monotone.  (The
  //! NGPA inside MODE_ASA always uses the paper's M = 8.)
  int memory = 1;
  //! Powell restart threshold: restart when |pg.pg_old| > this * ||pg||^2.
  Scalar powellRestart = 0.2;

  static beta_rule to_beta_rule(const std::string & token);
  static search_rule to_search_rule(const std::string & token);
  static step_rule to_step_rule(const std::string & token);
  static restart_rule to_restart_rule(const std::string & token);

private:

  //! Armijo sufficient-decrease constant, the usual 1e-4.  Also the NGPA's
  //! delta, which Hager & Zhang set to the same value.
  static constexpr double _c1 = 1e-4;
  //! give up on a direction after this many backtracks and restart CG.
  static constexpr int _maxBacktrack = 50;
  //! how far one line search may grow the trial step (2^_maxExpand).
  static constexpr int _maxExpand = 12;
  //! consecutive non-improving iterations tolerated before declaring done.
  static constexpr int _maxStall = 5;

  //! counters, for the verbose summary: which phase spent the iterations
  mutable size_t _ngpaIters = 0;
  mutable size_t _uaIters = 0;
  mutable size_t _uEmpty = 0;
  //! work counters.  An HZ line search evaluates through HzFunction and so is
  //! not counted here; SEARCH_ARC, the default, is.
  mutable size_t _nf = 0;
  mutable size_t _ng = 0;

  Scalar fEval(const InputType & x) const { _nf++; return _functor.f(x); }
  void gEval(const InputType & x, JacobianType & g) const
  { _ng++; _functor.gradient(x, g); }

public:
  EigenCgSolver(const Func & func);
  virtual ~EigenCgSolver() {};
  void internalSolve(InputType & x0);

private:
  //! x <- P(x), the projection onto the box.  Cheap insurance: every iterate
  //! this solver hands back has been through here.
  inline void clampToBox(InputType & x) const {
    x = x.cwiseMax(_lb).cwiseMin(_ub);
  }
  //! d^alpha(x) = P(x - alpha*g) - x, the gradient projection direction.
  inline void projDir(const InputType & x, const JacobianType & g,
                      Scalar alpha, InputType & d) const {
    d = (x - alpha * g).cwiseMax(_lb).cwiseMin(_ub) - x;
  }
  //! free(i) == false for variables held at a bound by the gradient.  This is
  //! MODE_PCG's active set; it is not Hager & Zhang's A(x).
  void freeSet(const InputType & x, const JacobianType & g, MaskType & free) const;
  //! A(x) of the paper: simply the variables sitting on a bound, whatever the
  //! gradient is doing.  atBound(i) == true means i is in A(x).
  void atBoundSet(const InputType & x, MaskType & atBound) const;
  //! largest a >= 0 with x + a*d still inside the box; how far the interior
  //! reaches along d, and so how far an unconstrained ray search may go.
  Scalar maxFeasibleStep(const InputType & x, const InputType & d) const;
  //! backtracking Armijo search along the projection arc, accepting against
  //! fref (== fval for a monotone search); false if it failed.
  bool lineSearch(const InputType & x, const JacobianType & g, const InputType & d,
                  Scalar & alpha, Scalar & fval, Scalar fref, InputType & xnew) const;
  //! Hager-Zhang Wolfe search, valid only while the step stays interior.
  //! false if it failed or wandered outside the box, and then the caller
  //! falls back to lineSearch().  Hands back the gradient at the new point.
  bool lineSearchHz(const InputType & x, const JacobianType & g, const InputType & d,
                    Scalar & alpha, Scalar & fval, InputType & xnew,
                    JacobianType & gnew) const;
  //! the two solvers
  void solveSimple(InputType & x);
  void solveAsa(InputType & x);
};

} /* namespace pwie */

#include "CppNumericalSolvers/src/EigenCgSolver.cpp"

#endif /* EIGENCGSOLVER_H_ */
