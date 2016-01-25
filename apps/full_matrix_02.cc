#include <deal.II/base/timer.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/vector_memory.templates.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

// Blaze include. All in one.
#include <blaze/Math.h>

#define EIGEN_MATRIX_PLUGIN "eigen_plugin.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "blaze_plugin.h"

using namespace dealii;

int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  unsigned int n = std::atoi(argv[1]);
  unsigned int reps = std::atoi(argv[2]);

  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;

  FullMatrix<double> matrix(n);
  Vector<double> x(n);
  for (unsigned int i = 0; i < n; ++i)
    {
      x[i] = i;
      for (unsigned int j = 0; j < n; ++j)
        matrix(i, j) = 1. + 1. / (i + 1) / (j + 1);
    }

  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  timer.enter_subsection("dealii_raw");
  Vector<double> tmp(n);
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult(x, tmp);
      x.add(3., tmp);
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << "DEBUG" << std::endl;
  std::cout << x << std::endl;
#endif

  for (unsigned int i = 0; i < n; ++i)
    x[i] = i;

  timer.enter_subsection("dealii_lo");
  const auto op = linear_operator(matrix);
  const auto reinit = op.reinit_range_vector;
  const auto step = (3.0 * identity_operator(reinit) + op) * op;
  
  for (unsigned int i = 0; i < reps; ++i)
    {
      step.vmult(x, x);
      x /= x.l2_norm();
    }
  timer.leave_subsection();

#ifdef DEBUG
  std::cout << x << std::endl;
#endif

  // Now do the same with Eigen. Let eigen interpret both x and the
  // matrix as its own objects

  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
    Ematrix(&matrix(0,0), n, n);

  typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> Evec;
  
  Evec Ex(n);

  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;
  
  timer.enter_subsection("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = (3*Ematrix*Ex+Ematrix*(Ematrix*Ex));
      Ex /= Ex.norm();
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Ex << std::endl << std::endl;
#endif
  // Now do the same but wrap it under a linear operator
  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;

  LinearOperator<Evec, Evec> Elo;
  
  Elo.vmult = [&Ematrix] (Evec &d, const Evec &s) {
    d = Ematrix*s;
  };
  
  Elo.vmult_add = [&Ematrix] (Evec &d, const Evec &s) {
    d += Ematrix*s;
  };

  Elo.reinit_range_vector = [&Ematrix] (Evec &v, bool fast)
  {
    v.resize(Ematrix.rows());
    if (fast == false)
      v *= 0;
  };

  Elo.reinit_domain_vector = [&Ematrix] (Evec &v, bool fast)
  {
    v.resize(Ematrix.cols());
    if (fast == false)
      v *= 0;
  };

  const auto Ereinit = Elo.reinit_range_vector;
  const auto Estep = (3.0 * identity_operator(Ereinit) + Elo) * Elo;
  
  timer.enter_subsection("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Estep.vmult(Ex, Ex);
      Ex /= Ex.norm();
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Ex << std::endl << std::endl;
#endif
  

  // Again, the same with Blaze. Let blaze interpret both the matrix as its own object
  typedef blaze::CustomVector<double,blaze::aligned,blaze::unpadded,blaze::columnVector> BVec;
  BVec Bx( &x(0), n);
  blaze::CustomMatrix<double,blaze::aligned,blaze::unpadded,blaze::rowMajor>
    Bmatrix( &matrix(0,0), n, n);

  
  for (unsigned int i = 0; i < n; ++i)
    Bx[i] = i;
  
  timer.enter_subsection("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = (3*Bmatrix*Bx+Bmatrix*Bmatrix*Bx);
      Bx /= std::sqrt(blaze::trans(Bx)*Bx);
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Bx << std::endl;
#endif

  // Again the same, but wrap it under a linear operator
  
  LinearOperator<BVector, BVector> Blo;

  Blo.vmult = [&Bmatrix] (BVector &d, const BVector &s) {
    static_cast<BVector::T&>(d) = Bmatrix*s;
  };

  
  Blo.vmult_add = [&Bmatrix] (BVector &d, const BVector &s) {
    static_cast<BVector::T&>(d) += Bmatrix*s;
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


  const auto Breinit = Blo.reinit_range_vector;
  const auto Bstep = (3.0 * identity_operator(Breinit) + Blo) * Blo;
  
  BVector wrappedBx(n);
  BVector::T &Bxx = static_cast<BVector::T&>(wrappedBx);

  for (unsigned int i = 0; i < n; ++i)
    Bxx[i] = i;
  
  timer.enter_subsection("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bstep.vmult(wrappedBx,wrappedBx);
      Bxx /= std::sqrt(blaze::trans(Bxx)*Bxx);
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Bxx << std::endl;
#endif
}
