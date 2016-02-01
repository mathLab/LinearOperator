#ifndef __test_h_
#define __test_h_

// Blaze include files, plus our own wrapper
#include <blaze/Math.h>
#include "blaze_plugin.h"

// Eigen include files, including our plugin
#define EIGEN_MATRIX_PLUGIN "eigen_plugin.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

// Some typedefs for blaze
typedef blaze::CompressedMatrix<double, blaze::rowMajor> BSparseMatrix;
typedef blaze::CustomMatrix<double,blaze::aligned,blaze::unpadded,blaze::rowMajor> BFullMatrixShadow;
// BVector is a wrapper to blaze::DynamicVector, and it is declared in blaze_plugin.h

// Some typedefs for eigen
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> ESparseMatrix;
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EFullMatrixShadow;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> EVector;


/**
 * Copy from deal.II to blaze sparse matrix.
 */
void copy(BSparseMatrix &Bmatrix, const SparseMatrix<double> &matrix)
{
  auto n = matrix.m();
  Bmatrix.resize(n,n);
  Bmatrix.reserve(matrix.n_nonzero_elements());

  for (unsigned int i=0; i<n; ++i)
    {
      for (auto it = matrix.begin(i); it != matrix.end(i); ++it)
        {
          Bmatrix.append(i, it->column(), it->value());
        }
      Bmatrix.finalize(i);
    }
}

/**
 * Copy from deal.II to eigen sparse matrix.
 */
void copy(ESparseMatrix &Ematrix, const SparseMatrix<double> &matrix)
{
  auto n = matrix.m();
  Ematrix.resize(n,n);
  Ematrix.reserve(matrix.n_nonzero_elements());

  for (unsigned int i=0; i<n; ++i)
    {
      for (auto it = matrix.begin(i); it != matrix.end(i); ++it)
        {
          Ematrix.insert(i, it->column()) = it->value();
        }
    }
  Ematrix.makeCompressed();
}

/**
 * Given any Blaze type matrix, encapsulate it into a LinearOperator
 * object.
 */
template<typename MAT>
LinearOperator<BVector, BVector> blaze_lo(MAT &Bmatrix)
{
  LinearOperator<BVector, BVector> Blo;

  Blo.vmult = [&Bmatrix] (BVector &d, const BVector &s)
  {
    static_cast<BVector::T &>(d) = Bmatrix*s;
  };


  Blo.vmult_add = [&Bmatrix] (BVector &d, const BVector &s)
  {
    static_cast<BVector::T &>(d) += Bmatrix*s;
  };

  Blo.reinit_range_vector = [&Bmatrix] (BVector &v, bool fast)
  {
    v.resize(Bmatrix.rows(), fast);
    if (fast == false)
      v *= 0;
  };

  Blo.reinit_domain_vector = [&Bmatrix] (BVector &v, bool fast)
  {
    v.resize(Bmatrix.columns(), fast);
    if (fast == false)
      v *= 0;
  };
  return Blo;
}


/**
 * Given any Eigen type matrix, encapsulate it into a LinearOperator
 * object.
 */
template<typename MAT>
LinearOperator<EVector, EVector> eigen_lo(MAT &Ematrix)
{
  LinearOperator<EVector, EVector> Elo;

  Elo.vmult = [&Ematrix] (EVector &d, const EVector &s)
  {
    d = Ematrix*s;
  };


  Elo.vmult_add = [&Ematrix] (EVector &d, const EVector &s)
  {
    d += Ematrix*s;
  };

  Elo.reinit_range_vector = [&Ematrix] (EVector &v, bool fast)
  {
    v.resize(Ematrix.rows(), fast);
    if (fast == false)
      v *= 0;
  };

  Elo.reinit_domain_vector = [&Ematrix] (EVector &v, bool fast)
  {
    v.resize(Ematrix.cols(), fast);
    if (fast == false)
      v *= 0;
  };
  return Elo;
}

/**
 * Create a the stiffness sparse matrix associated to a Poisson
 * problem on a square with a fixed number of refinement cycles and
 * bilinear elements.
 */
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

/**
 * Create a full matrix with entries
 *
 * matrix(i, j) = 1. + 1. / (i + 1) / (j + 1)
 *
 */
void create_full_matrix(FullMatrix<double> &matrix)
{
  unsigned int n=matrix.m();
  for (unsigned int i = 0; i < n; ++i)
    {
      for (unsigned int j = 0; j < n; ++j)
        matrix(i, j) = 1. + 1. / (i + 1) / (j + 1);
    }
}

/**
 * Fill the given vector with x[i] = i.
 */
template <typename VEC>
void reset_vector(VEC &x)
{
  for (unsigned int i=0; i<x.size(); ++i)
    {
      x[i] = i;
    }
}

/**
 * Helper function to compute the L2 norm of a deal.II vector.
 */
double norm(const Vector<double> &x)
{
  return x.l2_norm();
}

/**
 * Helper function to compute the L2 norm of a blaze vector.
 */
double norm(const BVector &x)
{
  return std::sqrt(blaze::trans(x)*x);
}

/**
 * Helper function to compute the L2 norm of an eigen vector.
 */
double norm(const EVector &x)
{
  return x.norm();
}

/**
 * Helper function to compute the L2 norm of a native blaze vector.
 */
double norm(const blaze::DynamicVector<double, false> &x)
{
  return std::sqrt(blaze::trans(x)*x);
}

/**
 * Check that the given vectors are equal according to the L2 norm. If
 * they differ, just print the error in optimized mode, or the full
 * vectors in debug mode.
 */
template<typename VEC1, typename VEC2>
void check_vector(const VEC1 &ref,
                  const VEC2 &src,
                  const double tolerance=1e-10)
{
  double err = 0;
  for (unsigned int i=0; i<ref.size(); ++i)
    {
      err += (ref[i]-src[i])*(ref[i]-src[i]);
    }
  err = std::sqrt(err);
  if (err > tolerance)
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
