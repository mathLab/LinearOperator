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
  auto &Bx = static_cast<BVector::T &>(Bxx);

  EVector Ex(n);


  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);

  // ============================================================ Start Output

  std::cout << "Case 3 - FullMatrix" << std::endl;
  std::cout << "n:    " << n << std::endl;
  std::cout << "reps: " << reps << std::endl;

  // ============================================================ deal.II RAW
  reset_vector(x);

  Vector<double> tmp(n);

  timer.enter_subsection ("dealii_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      matrix.vmult(tmp, x);
      matrix.vmult(x, tmp);
      x.add(3., tmp);
      x /= norm(x);
    }
  timer.leave_subsection();

  ref = x;

  // ============================================================ deal.II Slow LO
  reset_vector(x);

  const auto op = linear_operator(matrix);
  const auto reinit = op.reinit_range_vector;
  const auto step = (3.0 * identity_operator(reinit) + op) * op;

  timer.enter_subsection ("dealii_slowlo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      x = step * x;
      x /= norm(x);
    }
  timer.leave_subsection();

  check_vector(ref,x);

  // ============================================================ deal.II LO
  reset_vector(x);

  const auto step2 = (3.0 * identity_operator(reinit) + op) * op * x;

  timer.enter_subsection ("dealii_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      x = step2;
      x /= norm(x);
    }
  timer.leave_subsection();

  check_vector(ref,x);

  // ============================================================ Blaze Raw
  reset_vector(Bx);

  timer.enter_subsection ("blaze_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bx = 3 * Bmatrix * Bx + Bmatrix * Bmatrix * Bx;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Blaze Slow LO
  reset_vector(Bx);

  const auto Blo = blaze_lo(Bmatrix);
  const auto Breinit = Blo.reinit_range_vector;
  const auto Bstep = (3.0 * identity_operator(Breinit) + Blo) * Blo;

  timer.enter_subsection ("blaze_slowlo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bxx = Bstep * Bxx;
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Blaze LO
  reset_vector(Bx);

  const auto Bstep2 = (3.0 * identity_operator(Breinit) + Blo) * Blo * Bxx;

  timer.enter_subsection ("blaze_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Bstep2.apply(Bxx);
      Bx /= norm(Bx);
    }
  timer.leave_subsection();

  check_vector(ref,Bx);

  // ============================================================ Eigen Raw
  reset_vector(Ex);

  timer.enter_subsection ("eigen_raw");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = 3 * Ematrix * Ex + Ematrix * (Ematrix * Ex);
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);

  // ============================================================ Eigen Slow LO
  reset_vector(Ex);

  const auto Elo = eigen_lo(Ematrix);
  const auto Ereinit = Elo.reinit_range_vector;
  const auto Estep = (3.0 * identity_operator(Ereinit) + Elo) * Elo;

  timer.enter_subsection ("eigen_slowlo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Estep * Ex;
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);

  // ============================================================ Eigen LO
  reset_vector(Ex);

  const auto Estep2 = (3.0 * identity_operator(Ereinit) + Elo) * Elo * Ex;

  timer.enter_subsection ("eigen_lo");
  for (unsigned int i = 0; i < reps; ++i)
    {
      Ex = Estep2;
      Ex /= norm(Ex);
    }
  timer.leave_subsection();

  check_vector(ref,Ex);
}
