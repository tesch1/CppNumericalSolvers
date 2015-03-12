/**
 * Copyright (c) 2014-2015 Patrick Wieschollek
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

#ifndef NEWTONDESCENTSOLVER_H_
#define NEWTONDESCENTSOLVER_H_
#include "ISolver.hpp"
namespace pwie
{

template <typename Func>
class NewtonDescentSolver : public ISolver<Func>
{
  typedef typename Func::Scalar Scalar;
  typedef typename Func::InputType InputType;
  typedef typename Func::JacobianType JacobianType;
  typedef typename Functor<Func>::HessianType HessianType;

  using ISolver<Func>::settings;
  using ISolver<Func>::_functor;

public:
  NewtonDescentSolver(const Func & func);
  void internalSolve(InputType & x0);
};

} /* namespace pwie */

#include "CppNumericalSolvers/src/NewtonDescentSolver.cpp"

#endif /* NEWTONDESCENTSOLVER_H_ */
