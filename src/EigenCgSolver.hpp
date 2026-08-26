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
 * So MODE_ASA is its UA on a face that only grows, and that is what the
 * paper expects near a solution: Theorem 5.7 says that once the iterates
 * approach a stationary point satisfying the strong second-order condition,
 * "the ASA performs only the UA without restarts", and section 3 says U "is
 * almost always empty when we reach a neighborhood of a minimizer".  What is
 * wrong here is only that U is empty from iteration 1 rather than eventually,
 * for the units reason above.
 *
 * Two things this mode was accused of, and neither survived measurement.
 *
 * The first was that it converges to a worse point than asa_cg on
 * bibop_kobzar2004 --ctype ampangle: -0.991001 against -0.996850, medians of
 * three seeds.  Three seeds is not enough on that example.  Its own NOTES
 * say so -- two runs from the same start gave 0.9877 and 0.9704 -- and over
 * ten seeds the two solvers are the same result:
 *
 *   solver     median      mean       worst      n <= -0.9955
 *   asa_cg     -0.996840  -0.995161  -0.990524      7 of 10
 *   MODE_ASA   -0.996722  -0.994679  -0.980657      7 of 10
 *
 * The medians differ by 1.2e-4, well inside the 1e-3 this tree calls the
 * same result, and asa_cg draws -0.9905, -0.9914 and -0.9918 on three of
 * those ten seeds -- so -0.991 is not a place only this solver goes.  Both
 * distributions are heavy-tailed because the example's 25000-iteration
 * budget is short of what either solver needs; MODE_ASA left to run reaches
 * -0.996850 at 54392 iterations, which is asa_cg's answer to every digit.
 *
 * Tried on the strength of the three-seed number and then dropped: handing
 * the face back to the NGPA every N UA iterations, which is the only way
 * back once U is empty, since neither step 2a nor step 2b can fire.  It does
 * compress the tail (worst -0.9807 -> -0.9952, 9 of 10 seeds under -0.9955)
 * but it moves the median 3.3e-4 the wrong way and costs iterations on
 * iceberg ampangle, so it buys nothing the 1e-3 rule can see and it is a
 * deviation from Figure 3.1.  N = 800 was the best of 200/400/800/1600.  The
 * earlier attempt in the same direction -- ignoring U and switching on the
 * active set alone, 4.4 and 13.3 against 2.05 on bebop_pp45 ampangle -- was
 * a different rule and predates the lineSearch() fix below.
 *
 * The second accusation was the line search, and that was this comment's own
 * guess for a while.  It is wrong.  Measured on iceberg cartesian, which is
 * the clean case because both solvers reach the same minimizer so nothing
 * else confounds it: they agree on the cost to 1.1e-9 and on the active set
 * exactly -- the same 91 of 156 variables at a bound -- and asa_cg puts
 * ||P(x-g)-x||_inf under gradTol in 2080 iterations where this needs 19103.
 * Of the 19044 UA steps, 19030 satisfy the Wolfe curvature condition at
 * sigma = 0.9, and |g_new.s| / |g.s| is under 0.01 on 5566 of them, under
 * 0.1 on 3839 more and under 0.5 on 9587 more; 52 are worse than 0.5.  That
 * is an accurate line minimizer, not a bad one.
 *
 * Nor is it the conjugacy formula -- BETA_HZ, which is CG_DESCENT's beta_N,
 * with no restart at all, needs 16823, which is 12% and not the factor of 9
 * -- nor a periodic CG restart (18148 at a period of 66), nor the active
 * set: the frozen components carry almost none of the KKT residual, so the
 * NGPA has nothing to free there, and forcing it in costs iterations (21397
 * at a hand-back period of 50).  What remains is the asymptotic rate of this
 * CG against CG_DESCENT's on a very ill-conditioned face, and closing that
 * means writing CG_DESCENT, which is the code this class exists in order not
 * to read.  So iceberg cartesian is ~5x asa_cg's wall time, stopping on the
 * f-delta net at ~14000 iterations rather than on the gradient at ~19100 --
 * but the pulse it hands back is asa_cg's, and it has been asa_cg's since
 * about iteration 2000.  That is a known and unfixed cost, not a wrong
 * answer.
 *
 * Some of it has since been closed: lineSearch() now probes shorter steps as
 * well as longer ones.  On bebop_pp45 --ctype ampangle that alone takes
 * MODE_ASA from 1.267 to 0.80773 and MODE_PCG from 8.518 to 0.80773, both
 * stopping on gradTol, against asa_cg's 0.8078 and ipopt's 0.8076.  It is also
 * why the three initial-step rules measured as no effect -- the accepted step
 * barely depends on the guess once the search can move either way.  See the
 * comment there.
 *
 * Earlier and also dropped: nondimensionalizing the U thresholds by an rms
 * box width made U never empty, so it stayed in NGPA forever, ~11% worse; a
 * per-variable diagonal preconditioner (P-ASA, section 6) left U empty.
 *
 * The verbose summary prints ngpa/ua/Uempty and the f and g counts, which is
 * how to check any of this on a new problem before concluding from a mode.
 * It also prints the final pg against gradTol, which is the one line that
 * separates a run that converged from one the budget or the f-delta net cut
 * off mid-descent.  Both accusations above would have looked different with
 * that number in hand: bibop stops on the iteration limit with pg four
 * orders above gradTol, so it had not converged to anything.
 *
 * Both modes stop on the projected gradient, ||P(x - g) - x||_inf <= gradTol,
 * which is the quantity asa_cg stops on.  The f-delta test is only a safety
 * net over a long window -- see stalled() for why it has to be, and for what
 * it used to cost.  A run that is really dead shows up as a step that moves
 * neither x nor f, and the line search callers treat that as a failure rather
 * than accepting it: in the UA it hands the face back to the NGPA.
 *
 * This class exists at all because AsaCgSolver's iteration loop lives inside a
 * third-party C library with no per-iteration callback, so it cannot call
 * ISolver::postStep().  A parameterization whose feasible set is not a box
 * (cartesian --rescale: |rf| <= b1max is a disc) needs that hook.
 *
 * A clean-room implementation, written from the two published papers.  The
 * vendored hager_zhang.h is MIT, from upstream CppNumericalSolvers.
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
    SEARCH_AWOLFE,  //!< Hager-Zhang's search along the projection arc
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
  /*! Curvature constant of the approximate-Wolfe search.  CG_DESCENT runs at
   * 0.9; this is 0.1 on the measurement below, and the point of the number is
   * that it is no longer load-bearing.  Median f-units over three seeds
   * (a gradient or a fused pair is 4 f-units), against asa_cg:
   *
   *   example                       asa_cg   sigma .9   .1     .01
   *   iceberg cartesian               5172      9626   7058   13287
   *   burbop_ur90 cartesian          19755     16050  14576   12636
   *   sburbop_ur180_short            2256       3355   3055    6814
   *   iceberg ampangle, gtol 1e-8    47300     29011  33674   66239
   *
   * 0.1 wins the paired comparison against 0.9 on 7 of those 10 runs and has
   * the better worst case (1.36x asa_cg against 1.86x), so it is the default;
   * 0.9 costs about 6% more overall and either is defensible.  What is no
   * longer defensible is 0.01, which was this constant before the approximate
   * test was fixed: it is 60% worse than 0.1 here, and it was 0.01 only
   * because the search it belonged to was a *strong* Wolfe search that could
   * not otherwise find an accurate enough step.
   */
  static constexpr double _c2 = 0.1;
  //! the approximate-Wolfe search, all from Hager & Zhang, ACM TOMS 32 (2006)
  //! 113: eps and the decay of the C_k average that eps_k is built from,
  //! their bracket expansion factor rho, and gamma, the shrink one pass of
  //! the interval loop has to achieve before the next one bisects instead.
  static constexpr double _awEps = 1e-6;
  static constexpr double _awDecay = 0.7;
  static constexpr double _awRho = 5;
  //! psi1 of their initial-step rule I1, the QuadStep's value probe
  static constexpr double _awPsi1 = 0.1;
  static constexpr double _awGamma = 0.66;
  //! how far one line search may grow the trial step (2^_maxExpand).
  static constexpr int _maxExpand = 12;
  //! how many iterations the f-delta safety net measures progress over.
  static constexpr int _stallWindow = 200;

  //! counters, for the verbose summary: which phase spent the iterations
  mutable size_t _ngpaIters = 0;
  mutable size_t _uaIters = 0;
  mutable size_t _uEmpty = 0;
  //! work counters.  An HZ line search evaluates through HzFunction and so is
  //! not counted here; SEARCH_ARC, the default, is.
  mutable size_t _nf = 0;
  mutable size_t _ng = 0;
  //! how many trials the approximate-Wolfe search spent, and how many of its
  //! acceptances the approximate branch is responsible for.  The second number
  //! is the one that says whether this is an approximate Wolfe search at all.
  mutable size_t _awTrials = 0, _awApprox = 0, _awSearches = 0;
  //! Q_k and C_k of (4.2), the decaying average |f| that eps_k is a fraction
  //! of.  State of the run, not of one line search.
  mutable Scalar _awQ = 0, _awC = 0;

  Scalar fEval(const InputType & x) const { _nf++; return _functor.f(x); }
  void gEval(const InputType & x, JacobianType & g) const
  { _ng++; _functor.gradient(x, g); }
  /*! \brief  value and gradient at the same x, in one sweep.
   *
   * For the places this solver wants both at a point it has just moved to.
   * The functor shares the spin-ensemble propagation between them, so this
   * costs about what gEval alone costs; the value it returns is bit-identical
   * to fEval(x) at the same x, so nothing downstream can tell the difference
   * except the clock.
   */
  Scalar fgEval(const InputType & x, JacobianType & g) const
  { _nf++; _ng++; return _functor.valgrad(x, g); }

public:
  EigenCgSolver(const Func & func);
  virtual ~EigenCgSolver() {};
  void internalSolve(InputType & x0);

private:
  /*! Options spells "no objective limit" as -infinity, and this tree builds
   * with -ffast-math, which folds that constant to a *positive* denormal.  An
   * untouched limit therefore reads as zero and stops any problem whose cost
   * goes negative on the first iteration -- bebop_skinner2003 and
   * sburbop_ur180 both did.  Nothing sets a target that tiny on purpose.
   * (The clean fix is a finite default in pcgcompat.hpp's Options.)
   */
  inline Scalar objectiveLimit() const {
    const double lim = settings.objectiveLimit;
    return (lim > 0 && lim < 1e-300) ? -std::numeric_limits<Scalar>::max()
                                     : (Scalar)lim;
  }
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
  /*! \brief  approximate-Wolfe line search along the projection arc.
   *
   * The arc search satisfies sufficient decrease only, which leaves the step
   * within a factor of two of the ray minimum; CG wants the curvature
   * condition as well.  This is Hager & Zhang's line search -- the QuadStep
   * I1, bracket B0-B3, update U0-U3 and secant2 S1-S4 of ACM TOMS 32 (2006)
   * 113 -- on the arc, so both conditions hold at the returned point.
   *
   * It was none of those things when it was written, and the measurement that
   * says so is one counter.  Its approximate branch was gated behind a *fixed*
   * eps_k = 1e-6|f_0| and still asked for the strong |phi'| <= sigma|phi'(0)|,
   * so on an iceberg run it decided 0 of 4858 trials: what ran was a strong
   * Wolfe search at sigma = 0.01, and sigma was 0.01 because that was the only
   * way such a search could find an accurate step.  Three things followed
   * from fixing it, priced in f-units on iceberg (a gradient or a fused pair
   * is 4 f-units, 919 iterations of the old search cost 19658 of them):
   *
   * - the real test.  eps_k decays with |f| (their C_k average), and the
   *   approximate branch bounds the slope above instead of keeping the strong
   *   test.  On its own that made sigma harmless -- 6832 iterations at 0.9
   *   against 6699 before -- but no cheaper.
   * - I1, the QuadStep.  One *value* at psi1*alpha and a quadratic through it
   *   picks the first trial, and that is where the iterations went: 6832 ->
   *   1602 at sigma 0.9, and 1.14 gradients an iteration, which is asa_cg's
   *   1.25.  A Shanno-Phua guess carries no curvature along d, so without
   *   this the search either accepted a poor point or paid trials to fix it.
   * - values before gradients.  A trial whose value is above the band cannot
   *   be accepted under either test and puts a minimizer below it, so it
   *   never asks for the gradient a fused evaluation would have paid for.
   *   Same iterates, same answer, 16-36% fewer f-units.
   *
   * 5.29 trials and 20.88 f-units an iteration became 2.8 and 6.3, against
   * asa_cg's 2.65 and 6.37.  What is left of the old gap is iteration count,
   * which is the CG's asymptotic rate and is discussed above.
   */
  bool lineSearchAWolfe(const InputType & x, const JacobianType & g, const InputType & d,
                        Scalar & alpha, Scalar & fval, InputType & xnew,
                        JacobianType & gnew) const;

  bool lineSearchHz(const InputType & x, const JacobianType & g, const InputType & d,
                    Scalar & alpha, Scalar & fval, InputType & xnew,
                    JacobianType & gnew) const;
  //! f-delta safety net, called once per accepted iteration.  fwin and left
  //! are the caller's window state, started at (f0, _stallWindow).
  bool stalled(Scalar fbest, Scalar & fwin, int & left) const;
  //! the two solvers
  void solveSimple(InputType & x);
  void solveAsa(InputType & x);
};

} /* namespace pwie */

#include "CppNumericalSolvers/src/EigenCgSolver.cpp"

#endif /* EIGENCGSOLVER_H_ */
