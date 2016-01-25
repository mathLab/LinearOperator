#include <deal.II/base/timer.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>

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
      x = tmp;
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
  for (unsigned int i = 0; i < reps; ++i)
    {
      op.vmult(x, x);
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
  
  Eigen::Map<Evec> Ex(&x(0), n);

  for (unsigned int i = 0; i < n; ++i)
    Ex[i] = i;
  
  timer.enter_subsection("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*Ex;
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

  timer.enter_subsection("eigen_lo");
  Evec Etmp(n);
  for (unsigned int i = 0; i < reps; ++i)
    {
      Elo.vmult(Etmp, Ex);
      Ex = Etmp;
      Ex /= Ex.norm();
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Ex << std::endl << std::endl;
#endif
  

  // Again, the same with Blaze. Let blaze interpret both x and the
  // matrix as its own objects
  typedef blaze::CustomVector<double,blaze::aligned,blaze::unpadded,blaze::columnVector> BVec;
  BVec Bx( &x(0), n);
  blaze::CustomMatrix<double,blaze::aligned,blaze::unpadded,blaze::rowMajor>
    Bmatrix( &matrix(0,0), n, n);

  
  for (unsigned int i = 0; i < n; ++i)
    Bx[i] = i;
  
  timer.enter_subsection("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*Bx;
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


  BVector wrappedBx(n);
  BVector::T &Bxx = static_cast<BVector::T&>(wrappedBx);

  for (unsigned int i = 0; i < n; ++i)
    Bxx[i] = i;
  
  timer.enter_subsection("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Blo.vmult(wrappedBx,wrappedBx);
      Bxx /= std::sqrt(blaze::trans(Bxx)*Bxx);
    }
  timer.leave_subsection();
  
#ifdef DEBUG
  std::cout << Bxx << std::endl;
#endif
}
