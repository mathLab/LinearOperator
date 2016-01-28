#include "wrappers.h"
#include "test.h"

#include <iostream>

using namespace dealii;

int main(int argc, char *argv[])
{
  if (argc != 3)
    throw ExcMessage("Invalid number of command line parameters");
  unsigned int n = std::atoi(argv[1]);
  unsigned int reps = std::atoi(argv[2]);

  // Deal.II Full Matrix
  FullMatrix<double> matrix(n);
  create_full_matrix(matrix);

  // Blaze Full Matrix
  BFullMatrixShadow Bmatrix(&matrix(0,0), n, n);
  // Eigen Full Matrix
  EFullMatrixShadow Ematrix(&matrix(0,0), n, n);

  // And create temporary vectors
  Vector<double> x(n), ref(n);
  
  BVector Bxx(n);
  auto &Bx = static_cast<BVector::T&>(Bxx);
  
  EVector Ex(n);

  
  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  // ============================================================ Start Output

  std::cout << "Case 2 - FullMatrix" << std::endl;
  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;
  
  // ============================================================ deal.II RAW
  reset_vector(x);
  
  timer.enter_subsection ("dealii_raw");
  Vector<double> tmp(n);
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult(x, tmp);
      matrix.vmult(tmp, x);
      x = tmp;
      x /= norm(x);
    }
  timer.leave_subsection();

  ref = x;
  
  // ============================================================ deal.II LO  
  reset_vector(x);

  timer.enter_subsection ("dealii_lo");
  auto op = linear_operator(matrix);
  op = op*op*op;
  for (unsigned int i = 0; i < reps; ++i)
    {
      op.vmult(x, x);
      x /= norm(x);
    }
  timer.leave_subsection();

  check_vector(ref,x);

  // ============================================================ Blaze Raw
  reset_vector(Bx);
  
  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = Bmatrix*Bmatrix*Bmatrix*Bx;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Blaze LO
  reset_vector(Bx);
  
  auto Blo = blaze_lo(Bmatrix);
  Blo = Blo*Blo*Blo;
  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Blo.vmult(Bxx,Bxx);
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Eigen Raw a 
  reset_vector(Ex);
  
  if(n<300) {
    timer.enter_subsection ("eigen_raw");
    for (unsigned int i = 0; i < reps; ++i)
      {
	Ex = Ematrix*Ematrix*Ematrix*Ex;
	Ex /= norm(Ex);
      }
    timer.leave_subsection();
  }

  check_vector(ref,Ex);
  
  // ============================================================ Eigen Raw b
  reset_vector(Ex);
  
  timer.enter_subsection ("eigen_smart");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Ematrix*(Ematrix*(Ematrix*Ex));
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);
  
  // ============================================================ Eigen LO
  reset_vector(Ex);
  
  auto Elo = eigen_lo(Ematrix);
  Elo = Elo*Elo*Elo;
  timer.enter_subsection ("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Elo.vmult(Ex,Ex);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);
}
