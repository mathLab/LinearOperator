#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>

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

  timer.enter_subsection ("raw");
  Vector<double> tmp(dof_handler.n_dofs());
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult(x, tmp);
      tmp *= 3.;
      x += tmp;
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << "DEBUG" << std::endl;
  std::cout << x << std::endl;
#endif

  for (unsigned int i = 0; i < x.size(); ++i)
    x[i] = i;

  timer.enter_subsection ("linear_operator");
  for (unsigned int i = 0; i < reps; ++i)
    {
      const auto op = linear_operator(matrix);
      const auto id = identity_operator(op.reinit_range_vector);
      x = (op + 3.0 * id) * op * x;
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << x << std::endl;
#endif
}
