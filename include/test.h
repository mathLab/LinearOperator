#ifndef __test_h_
#define __test_h_

#include "wrappers.h"
#include "blaze_plugin.h"

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


using namespace dealii;

void create_sparse_matrix(SparseMatrix<double> &matrix,
                          const unsigned int refinement)
{

  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(refinement);

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler;
  dof_handler.initialize(triangulation, fe);

  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  static SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);

  matrix.reinit(sparsity_pattern);

  QGauss<2> quadrature(2);
  MatrixCreator::create_laplace_matrix(dof_handler, quadrature, matrix);
}

void create_full_matrix(FullMatrix<double> &matrix)
{
  unsigned int n=matrix.m();
  for (unsigned int i = 0; i < n; ++i)
    {
      for (unsigned int j = 0; j < n; ++j)
        matrix(i, j) = 1. + 1. / (i + 1) / (j + 1);
    }
}


template <typename VEC>
void reset_vector(VEC &x)
{
  for (unsigned int i=0; i<x.size(); ++i)
    {
      x[i] = i;
    }
}

double norm(const Vector<double> &x)
{
  return x.l2_norm();
}


double norm(const BVector &x)
{
  return std::sqrt(blaze::trans(x)*x);
}

double norm(const EVector &x)
{
  return x.norm();
}

double norm(const blaze::DynamicVector<double, false> &x)
{
  return std::sqrt(blaze::trans(x)*x);
}

template<typename VEC1, typename VEC2>
void check_vector(const VEC1 &ref, const VEC2 &src)
{
  double err = 0;
  for (unsigned int i=0; i<ref.size(); ++i)
    {
      err += (ref[i]-src[i])*(ref[i]-src[i]);
    }
  err = std::sqrt(err);
  if (err > 1e-10)
    {
      std::cout << "L2 Error: " << err << std::endl;

#ifdef DEBUG
      std::cout << "DEBUG left" << std::endl;
      std::cout << ref << std::endl;
      std::cout << "DEBUG right" << std::endl;
      std::cout << src << std::endl;
#endif
    }
}

#endif
