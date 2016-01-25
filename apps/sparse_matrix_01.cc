#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>

#include "wrappers.h"

#include <iostream>

using namespace dealii;

int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  unsigned int refinement = std::atoi(argv[1]);
  unsigned int reps = std::atoi(argv[2]);

  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(refinement);

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  dof_handler.initialize(triangulation, fe);

  std::cout << "n:    " << dof_handler.n_dofs() << std::endl;
  std::cout << "reps: " << reps << std::endl;

  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  SparseMatrix<double> matrix;
  matrix.reinit(sparsity_pattern);

  QGauss<2> quadrature(2);
  MatrixCreator::create_laplace_matrix(dof_handler, quadrature, matrix);

  Vector<double> x(dof_handler.n_dofs());
  for (unsigned int i = 0; i < x.size(); ++i)
    x(i) = i;

  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  timer.enter_subsection ("dealii_raw");
  Vector<double> tmp(dof_handler.n_dofs());
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      x = tmp;
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << "DEBUG" << std::endl;
  std::cout << x << std::endl;
#endif

  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] = i;

  timer.enter_subsection ("dealii_lo");
  const auto op = linear_operator(matrix);
  for (unsigned int i = 0; i < reps; ++i)
    {
      op.vmult(x, x);
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << x << std::endl;
#endif

  // Now copy the sparse matrix to eigen and blaze
  auto n = dof_handler.n_dofs();
  
  BSparseMatrix Bmatrix;
  copy(Bmatrix, matrix);
  ESparseMatrix Ematrix;
  copy(Ematrix, matrix);
  
  // And create two vectors
  BVector Bxx(n);
  EVector Ex(n);

  auto &Bx = static_cast<BVector::T&>(Bxx);
  
  // ============================================================ Blaze Raw  
  for (unsigned int i = 0; i < x.size(); ++i)
    Bx[i] = i;

  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*Bx;
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif

  
  // ============================================================ Eigen Raw  
  for (unsigned int i = 0; i < x.size(); ++i)
    Ex[i] = i;

  timer.enter_subsection ("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*Ex;
      Ex /= Ex.norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Ex << std::endl;
#endif

  
  // ============================================================ Blaze LO
  for (unsigned int i = 0; i < x.size(); ++i)
    Bx[i] = i;

  auto Blo = blaze_lo(Bmatrix);
  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Blo.vmult(Bxx,Bxx);
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif

  
  // ============================================================ Eigen LO
  for (unsigned int i = 0; i < x.size(); ++i)
    Ex[i] = i;

  auto Elo = eigen_lo(Ematrix);
  timer.enter_subsection ("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Elo.vmult(Ex,Ex);
      Ex /= Ex.norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << Ex << std::endl;
#endif
}
