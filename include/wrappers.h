#ifndef __wrappers_h_
#define __wrappers_h_

// Some utilities to wrap Eigen and Blaze linear operators.

// Blaze include. All in one.
#include <blaze/Math.h>
#include "blaze_plugin.h"

#define EIGEN_MATRIX_PLUGIN "eigen_plugin.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>


// Some typedefs
typedef blaze::CompressedMatrix<double, blaze::rowMajor> BSparseMatrix;
typedef blaze::CustomMatrix<double,blaze::aligned,blaze::unpadded,blaze::rowMajor> BFullMatrixShadow;
// BVector is a wrapper to blaze::DynamicVector

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> ESparseMatrix;
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > EFullMatrixShadow;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> EVector;


/**
 * Copy from deal.II to blaze sparse matrix.
 */
void copy(BSparseMatrix &Bmatrix, const dealii::SparseMatrix<double> &matrix)
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
void copy(ESparseMatrix &Ematrix, const dealii::SparseMatrix<double> &matrix)
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
dealii::LinearOperator<BVector, BVector> blaze_lo(MAT &Bmatrix)
{
  dealii::LinearOperator<BVector, BVector> Blo;

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
dealii::LinearOperator<EVector, EVector> eigen_lo(MAT &Ematrix)
{
  dealii::LinearOperator<EVector, EVector> Elo;

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

#endif
