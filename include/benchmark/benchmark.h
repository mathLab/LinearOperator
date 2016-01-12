#ifndef _benchmark_h
#define _benchmark_h

namespace LinearSolvers
{
  template <class PreconditionerA, class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner (const TrilinosWrappers::BlockSparseMatrix  &S,
                              const TrilinosWrappers::BlockSparseMatrix  &Spre,
                              const PreconditionerMp                     &Mppreconditioner,
                              const PreconditionerA                      &Amgpreconditioner,
                              const bool                                  optimized = true,
                              const bool                                  amgyn = false)
      :
      stokes_matrix     (&S),
      stokes_preconditioner_matrix     (&Spre),
      mp_preconditioner (Mppreconditioner),
      a_preconditioner (Amgpreconditioner),
      optimized         (optimized),
      amgyn         (amgyn)
    {}
    void vmult (TrilinosWrappers::MPI::BlockVector       &dst,
                const TrilinosWrappers::MPI::BlockVector &src) const
    {
      if (optimized)
        {
          TrilinosWrappers::MPI::Vector utmp(src.block(0));
          SolverControl solver_control(5000, 1e-12);
          SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

          solver.solve(stokes_preconditioner_matrix->block(1,1),
                       dst.block(1), src.block(1),
                       mp_preconditioner);
          dst.block(1) *= -1.0;
          stokes_matrix->block(0,1).vmult(utmp, dst.block(1));
          utmp*=-1.0;

          utmp.add(src.block(0));

          if (amgyn)
            {
              SolverControl solver_control(5000, 1e-12);
              TrilinosWrappers::SolverCG solver(solver_control);
              solver.solve(stokes_preconditioner_matrix->block(0,0),
                           dst.block(0), utmp, a_preconditioner);
            }
          else
            stokes_preconditioner_matrix->block(0,0).vmult (dst.block(0), utmp);
        }
      else
        {
          TrilinosWrappers::MPI::Vector tmp(src.block(1));
          TrilinosWrappers::MPI::Vector utmp(src.block(0));

          SolverControl solver_control(5000, 1e-12);
          SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

          solver.solve(stokes_preconditioner_matrix->block(1,1),
                       tmp, src.block(1),
                       mp_preconditioner);

          stokes_matrix->block(0,1).vmult(utmp, tmp);

          stokes_preconditioner_matrix->block(0,0).vmult(dst.block(0), utmp);

          stokes_preconditioner_matrix->block(0,0).vmult_add(dst.block(0), src.block(0));

          solver.solve(stokes_preconditioner_matrix->block(1,1),
                       dst.block(1), src.block(1),
                       mp_preconditioner);
          dst.block(1) *= -1.0;
        }
    }
  private:
    const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_matrix;
    const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_preconditioner_matrix;
    const PreconditionerMp &mp_preconditioner;
    const PreconditionerA &a_preconditioner;
    const bool optimized;
    const bool amgyn;
  };
}

#endif // _benchmark_h
