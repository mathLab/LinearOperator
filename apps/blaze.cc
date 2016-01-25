/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <cstdlib>

#include "wrappers.h"

using namespace dealii;


int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  int n = std::atoi(argv[1]);
  int band = std::min(n/10+2, 50);
  unsigned int reps = std::atoi(argv[2]);

  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;


  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  BSparseMatrix A(n,n,band);

  for (int i=0; i<n; ++i)
    for (int j=std::max(0,i-band/2); j<std::min(i+band/2, n); ++j)
      A(i,j) = (double)std::rand()/(double)RAND_MAX;

  // Uncomment if you want to see the matrix
  // std::cout << A << std::endl;

  BVector v(n);
  BVector v1(n);

  blaze::randomize(static_cast<blaze::DynamicVector<double> &>(v), 0.0, 1.0);
  blaze::randomize(static_cast<blaze::DynamicVector<double> &>(v1), 0.0, 1.0);
  // v = Eigen::ArrayXd::Random(n);
  // v1 = Eigen::ArrayXd::Random(n);

  BVector b0(n);
  BVector b1(n);
  b0 = 0.0;
  b1 = 0.0;

  auto lo = blaze_lo(A);
  {
    v1 = v;
    timer.enter_subsection ("linear_operator");
    for (unsigned int i=0; i<reps; ++i)
      {
        v1 *= (double)i;
        b0 = lo*lo*lo*v;
      }
    timer.leave_subsection();
  }
  {
    v1 = v;
    timer.enter_subsection ("blaze");
    for (unsigned int i=0; i<reps; ++i)
      {
        v1 *= (double)i;
        static_cast<blaze::DynamicVector<double>&>(b1) = A*A*A*v;
      }
    timer.leave_subsection();
  }
}
