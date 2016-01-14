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
 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/linear_operator.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>

using namespace dealii;
class MatrixBenchmark
{
public:
  MatrixBenchmark (unsigned int size,
                   unsigned int reps);
  void run ();
private:
  void reset_vectors ();

  ConditionalOStream   pout;
  Triangulation<2>     triangulation;
  FE_Q<2>              fe;
  DoFHandler<2>        dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> matrix;
  Vector<double>       vector;
  Vector<double>       result;

  unsigned int reps;
  unsigned int size;
  TimerOutput timer;
};

MatrixBenchmark::MatrixBenchmark (unsigned int size,
                                  unsigned int reps)
  :
  pout(std::cout, reps == 1),
  fe (1),
  dof_handler (triangulation),
  reps(reps),
  size(size),
  timer ( std::cout,
          TimerOutput::summary,
          TimerOutput::wall_times)
{}

void MatrixBenchmark::reset_vectors()
{
  for (unsigned int i=0; i<vector.size(); ++i)
    {
      vector(i)=(double)i/(double)vector.size();
      result(i)=0.0;
    }
}

void MatrixBenchmark::run ()
{

  std::vector< unsigned int > repetitions;
  repetitions.push_back(size);
  repetitions.push_back(size);

  const Point<2> point_one (0,0);
  const Point<2> point_two (1,1);

  GridGenerator::subdivided_hyper_rectangle ( triangulation,
                                              repetitions,
                                              point_one,
                                              point_two);


  dof_handler.distribute_dofs (fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  matrix.reinit (sparsity_pattern);
  vector.reinit (dof_handler.n_dofs());
  result.reinit (dof_handler.n_dofs());

  QGauss<2>  quadrature_formula(2);
  MatrixCreator::create_laplace_matrix (
    dof_handler,
    quadrature_formula,
    matrix);
  
  auto S  = linear_operator<Vector<double>, Vector<double>>(matrix);
  
  std::string bar = "=";
  for (unsigned int i=0; i<40; ++i)
    bar += "=";
  bar += "\n";

  std::string b = "=";
  for (unsigned int i=0; i<40; ++i)
    b += "-";
  b += "\n";

  std::cout << bar
            << bar
            << "  Size: " << size
            << std::endl
            << "  Repetitions: " << reps
            << std::endl
            << "  Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "  Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << bar
            << bar
            << std::endl;

  pout << bar
       << "  SIMULATION" << std::endl
       << bar;

  pout << bar
       << "  simple vmult - STD" << std::endl
       << bar;
  reset_vectors();
  timer.enter_subsection ("simple vmult - STD");
  for (unsigned int i = 0; i<reps; ++i)
    matrix.vmult(result,vector);
  timer.leave_subsection();
  auto diff=result;

  pout << bar
       << "  simple vmult - LO" << std::endl
       << bar;
  reset_vectors();
  timer.enter_subsection ("simple vmult - LO");
  for (unsigned int i = 0; i<reps; ++i)
    S.vmult(result,vector);
  timer.leave_subsection();
  diff -=result;

  pout << "Diff norm = "
       << diff.linfty_norm()
       << std::endl
       << "  result norm = "
       << result.linfty_norm()
       << std::endl
       << "  vector norm = "
       << vector.linfty_norm()
       << std::endl
       << std::endl;

  pout << bar
       << "  operator - STD" << std::endl
       << bar;
  reset_vectors();
  timer.enter_subsection ("operator - STD");

  Vector<double> tmp1,tmp2;
  tmp1.reinit(vector, /*omit_zeroing_entries =*/ true);
  tmp2.reinit(vector, /*omit_zeroing_entries =*/ true);
  for (unsigned int j = 0; j<reps; ++j)
    {
      // matrix.vmult(tmp1, vector);
      // result.add(3.0, tmp1);
      // matrix.vmult(tmp2, tmp1);
      // matrix.vmult(tmp1, tmp2);
      // result.add(1.0, tmp1, 2.0, tmp2);
      matrix.vmult(tmp1, vector);
      result.add(3.0, tmp1);
      matrix.vmult(tmp2, tmp1);
      result.add(1.0, tmp2);
    } 

  timer.leave_subsection();
  diff=result;

  auto Id = identity_operator<Vector<double>>(S.reinit_range_vector);
  // auto F = (S*S + 2.0*S + 3.0 * Id )*S;
  auto F = (S + 3.0 * Id )*S;
  
  pout << bar
       << "  operator - LO" << std::endl
       << bar;
  reset_vectors();
  reset_vectors();
  timer.enter_subsection ("operator - LO");
  for (unsigned int i = 0; i<reps; ++i)
    {
      F.vmult(result,vector);
    }
  timer.leave_subsection();
  diff-=result;

  pout << "Diff norm = "
       << diff.linfty_norm()
       << std::endl
       << "  result norm = "
       << result.linfty_norm()
       << std::endl
       << "  vector norm = "
       << vector.linfty_norm()
       << std::endl
       << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cout << "Two integers are required."
                << std::endl
                << " USAGE: " << argv[0] << " [size] [repetitions]"
                << std::endl;
      exit(0);
    }
  int size = atoi (argv[1]);
  int reps = atoi (argv[2]);

  deallog.depth_console (2);
  MatrixBenchmark problem(size, reps);
  problem.run() ;
  return 0;
}