/**
 * This benchmark is to test vmult with differents matrix size.
 */

#include "navier_stokes.h"

#include <deal.II/base/timer.h>

#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>

#include <deal.II/dofs/dof_renumbering.h>

#include "benchmark/benchmark.h"
#include "benchmark/utilities.h"

using namespace dealii;
using namespace deal2lkit;

class test_class : NavierStokes<2>
{
public:
  test_class (const RefinementMode refinement_mode) : NavierStokes<2>(refinement_mode) {};
  void run (bool linear_operator = true);

private:
  void setup_dofs ();
  void build_navier_stokes_preconditioner ();
  double vmult_linear_operators_with_null(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution);
  double vmult_linear_operators_from_zero(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution);
  double vmult_linear_operators_without_null(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution);
  double vmult_old_implementation_with_tricks(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution);
  double vmult_old_implementation_without_tricks(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution);
};

void test_class::build_navier_stokes_preconditioner ()
{
  if (rebuild_navier_stokes_preconditioner == false)
    return;

  assemble_navier_stokes_preconditioner ();

  std::vector<std::vector<bool> > constant_modes;
  FEValuesExtractors::Vector velocity_components(0);
  DoFTools::extract_constant_modes (*navier_stokes_dof_handler,
                                    navier_stokes_fe->component_mask(velocity_components),
                                    constant_modes);

  Mp_preconditioner.reset  (new TrilinosWrappers::PreconditionJacobi());
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.constant_modes = constant_modes;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  Mp_preconditioner->initialize (navier_stokes_preconditioner_matrix.block(1,1));
  Amg_preconditioner->initialize (navier_stokes_preconditioner_matrix.block(0,0),
                                  Amg_data);

  rebuild_navier_stokes_preconditioner = false;
}

void test_class::setup_dofs ()
{
  std::vector<unsigned int> navier_stokes_sub_blocks (3,0);
  navier_stokes_sub_blocks[2] = 1;
  navier_stokes_dof_handler->distribute_dofs (*navier_stokes_fe);
  DoFRenumbering::component_wise (*navier_stokes_dof_handler, navier_stokes_sub_blocks);

  std::vector<types::global_dof_index> navier_stokes_dofs_per_block (2);
  DoFTools::count_dofs_per_block (*navier_stokes_dof_handler, navier_stokes_dofs_per_block,
                                  navier_stokes_sub_blocks);

  const unsigned int n_u = navier_stokes_dofs_per_block[0],
                     n_p = navier_stokes_dofs_per_block[1];

  std::vector<IndexSet> navier_stokes_partitioning, navier_stokes_relevant_partitioning;
  IndexSet navier_stokes_relevant_set;
  {
    IndexSet navier_stokes_index_set = navier_stokes_dof_handler->locally_owned_dofs();
    navier_stokes_partitioning.push_back(navier_stokes_index_set.get_view(0,n_u));
    navier_stokes_partitioning.push_back(navier_stokes_index_set.get_view(n_u,n_u+n_p));

    DoFTools::extract_locally_relevant_dofs (*navier_stokes_dof_handler,
                                             navier_stokes_relevant_set);
    navier_stokes_relevant_partitioning.push_back(navier_stokes_relevant_set.get_view(0,n_u));
    navier_stokes_relevant_partitioning.push_back(navier_stokes_relevant_set.get_view(n_u,n_u+n_p));

  }

  {
    navier_stokes_constraints.clear ();
    navier_stokes_constraints.reinit (navier_stokes_relevant_set);

    DoFTools::make_hanging_node_constraints (*navier_stokes_dof_handler,
                                             navier_stokes_constraints);

    FEValuesExtractors::Vector velocity_components(0);
    //boundary_conditions.set_time(time_step*time_step_number);
    VectorTools::interpolate_boundary_values (*navier_stokes_dof_handler,
                                              0,
                                              boundary_conditions,
                                              navier_stokes_constraints,
                                              navier_stokes_fe->component_mask(velocity_components));

    navier_stokes_constraints.close ();
  }

  setup_navier_stokes_matrix (navier_stokes_partitioning, navier_stokes_relevant_partitioning);
  setup_navier_stokes_preconditioner (navier_stokes_partitioning,
                                      navier_stokes_relevant_partitioning);

  navier_stokes_rhs.reinit (navier_stokes_partitioning, navier_stokes_relevant_partitioning,
                            MPI_COMM_WORLD, true);
  navier_stokes_solution.reinit (navier_stokes_relevant_partitioning, MPI_COMM_WORLD);
  old_navier_stokes_solution.reinit (navier_stokes_solution);

  rebuild_navier_stokes_matrix              = true;
  rebuild_navier_stokes_preconditioner      = true;
}

void test_class::run(bool linear_operator)
{
  int dim = 2;
  unsigned int runs = 5;
  unsigned int refinement = 15;
  make_grid_fe ();
  // triangulation->refine_global (1);

  TrilinosWrappers::MPI::BlockVector u[5];
  TrilinosWrappers::MPI::BlockVector d[5];

  double norm_i[5]    = {0., 0., 0., 0., 0.};
  double norm_2[5]    = {0., 0., 0., 0., 0.};
  double mean_time[5] = {0., 0., 0., 0., 0.};
  double timer[5]     = {0., 0., 0., 0., 0.};


  for (unsigned int j = 0; j < refinement; ++j)
    {
      triangulation->refine_global (1);
      // }
      std::vector<unsigned int> navier_stokes_sub_blocks (dim+1,0);
      navier_stokes_sub_blocks[dim] = 1;
      navier_stokes_dof_handler->distribute_dofs (*navier_stokes_fe);
      DoFRenumbering::component_wise (*navier_stokes_dof_handler, navier_stokes_sub_blocks);

      std::vector<types::global_dof_index> navier_stokes_dofs_per_block (2);
      DoFTools::count_dofs_per_block (*navier_stokes_dof_handler, navier_stokes_dofs_per_block,
                                      navier_stokes_sub_blocks);

      const unsigned int n_u = navier_stokes_dofs_per_block[0],
                         n_p = navier_stokes_dofs_per_block[1];

      setup_dofs ();
      assemble_navier_stokes_system ();
      build_navier_stokes_preconditioner ();
      // unsigned int i = 500;
      for (unsigned int i = 100; i< 1000; i*=10)
        {
          for (unsigned int j =0; j<5; ++j)
            {
              norm_i[j]     = 0.;
              norm_2[j]     = 0.;
              mean_time[j]  = 0.;
              timer[j]      = 0.;
            }

          pcout << " | " << std::endl;
          pcout << " =======================================================================================================================================================================================" <<std::endl;
          pcout << " | " << std::setw(16) << "run"
                << " | " << std::setw(16) << "dim"
                << " | " << std::setw(16) << "reps"
                << " | " << std::setw(22) << "1. trick"
                << " | " << std::setw(22) << "2. no trick"
                << " | " << std::setw(22) << "3. LinOperator [basic]"
                << " | " << std::setw(22) << "4. LinOperator [null]"
                << " | " << std::setw(22) << "5. LinOperator [inv]"
                << " | " << std::endl;
          pcout << " =======================================================================================================================================================================================" <<std::endl;
          for (unsigned int n = 0; n < runs; ++n )
            {
              timer[0] = vmult_old_implementation_with_tricks(i, u[0]);
              timer[1] = vmult_old_implementation_without_tricks(i, u[1]);
              timer[2] = vmult_linear_operators_without_null(i, u[2]);
              timer[3] = vmult_linear_operators_with_null(i, u[3]);
              timer[4] = vmult_linear_operators_from_zero(i, u[4]);

              pcout << " | " << std::setw(16) << n
                    << " | " << std::setw(16) << n_p + n_u
                    << " | " << std::setw(16) << i
                    << " | " << std::setw(22) << timer[0]
                    << " | " << std::setw(22) << timer[1]
                    << " | " << std::setw(22) << timer[2]
                    << " | " << std::setw(22) << timer[3]
                    << " | " << std::setw(22) << timer[4]
                    << " | " << std::endl;


              for (unsigned int j = 0; j<5; ++j)
                {
                  mean_time[j] += timer[j];
                  if (n==0)
                    d[j] = u[0];
                  else
                    d[j] += u[0];

                  d[j] -= u[j];
                }
            }

          for (unsigned int j = 0; j< 5; ++j)
            {
              mean_time[j] /= runs;
              d[j] /= double(runs);
              norm_i[j] = d[j].linfty_norm();
              norm_2[j] = d[j].l2_norm();
            }

          pcout << " ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;

          pcout << " | " << std::setw(16) << "mean"
                << " | " << std::setw(16) << n_p + n_u
                << " | " << std::setw(16) << i
                << " | " << std::setw(22) << mean_time[0]
                << " | " << std::setw(22) << mean_time[1]
                << " | " << std::setw(22) << mean_time[2]
                << " | " << std::setw(22) << mean_time[3]
                << " | " << std::setw(22) << mean_time[4]
                << " | " << std::endl;

          pcout << " ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;

          pcout << " | " << std::setw(16) << "speed (ratio)"
                << " | " << std::setw(16) << " "
                << " | " << std::setw(16) << " "
                << " | " << std::setw(22) << mean_time[0]/mean_time[0]
                << " | " << std::setw(22) << mean_time[0]/mean_time[1]
                << " | " << std::setw(22) << mean_time[0]/mean_time[2]
                << " | " << std::setw(22) << mean_time[0]/mean_time[3]
                << " | " << std::setw(22) << mean_time[0]/mean_time[4]
                << " | " << std::endl;

          pcout << " ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
                << std::endl;

          pcout << " | " << std::setw(16) << "norm (inf)"
                << " | " << std::setw(16) << n_p + n_u
                << " | " << std::setw(16) << i
                << " | " << std::setw(22) << norm_i[0]
                << " | " << std::setw(22) << norm_i[1]
                << " | " << std::setw(22) << norm_i[2]
                << " | " << std::setw(22) << norm_i[3]
                << " | " << std::setw(22) << norm_i[4]
                << " | " << std::endl;

          pcout << " ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
                << std::endl;

          pcout << " | " << std::setw(16) << "norm (L2)"
                << " | " << std::setw(16) << n_p + n_u
                << " | " << std::setw(16) << i
                << " | " << std::setw(22) << norm_2[0]
                << " | " << std::setw(22) << norm_2[1]
                << " | " << std::setw(22) << norm_2[2]
                << " | " << std::setw(22) << norm_2[3]
                << " | " << std::setw(22) << norm_2[4]
                << " | " << std::endl;

          pcout << " =======================================================================================================================================================================================" <<std::endl<<std::endl;
        }

    }
  return ;
}

double test_class::vmult_old_implementation_with_tricks(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution)
{
  Timer timer(MPI_COMM_WORLD, /* sync_wall_time = */ true);
  TrilinosWrappers::MPI::BlockVector
  distributed_navier_stokes_solution (navier_stokes_rhs);
  distributed_navier_stokes_solution = navier_stokes_solution;

  const unsigned int
  start = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().first);
  const unsigned int
  end   = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().second);

  for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
    distributed_navier_stokes_solution(i) = 0;

  // PRECONDITIONER:
  Amg_preconditioner ->initialize(navier_stokes_preconditioner_matrix.block(0,0));
  Mp_preconditioner  ->initialize(navier_stokes_preconditioner_matrix.block(1,1));

  const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
        TrilinosWrappers::PreconditionJacobi>
        preconditioner (navier_stokes_matrix, navier_stokes_preconditioner_matrix,
                        *Mp_preconditioner, *Amg_preconditioner, true, false);

  /* BENCHMARK*/
  {
    auto v = distributed_navier_stokes_solution;
    for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
      {
        v(i)=1;
        distributed_navier_stokes_solution(i) = 1;
      }

    auto dim_max = distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size()-1;

    timer.start();
    for (unsigned int i = 0; i<reps ; ++i)
      {
        if (i < dim_max)
          {
            v(i) = 2;
            v(dim_max - i) = 2;
          }
        preconditioner.vmult(distributed_navier_stokes_solution, v);
      }
    timer.stop();
    solution = distributed_navier_stokes_solution;

    solution = distributed_navier_stokes_solution;
  }
  return timer.wall_time();
}

double test_class::vmult_old_implementation_without_tricks(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution)
{
  Timer timer(MPI_COMM_WORLD, /* sync_wall_time = */ true);
  TrilinosWrappers::MPI::BlockVector
  distributed_navier_stokes_solution (navier_stokes_rhs);
  distributed_navier_stokes_solution = navier_stokes_solution;

  const unsigned int
  start = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().first);
  const unsigned int
  end   = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().second);

  for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
    distributed_navier_stokes_solution(i) = 0;


  // PRECONDITIONER:
  Amg_preconditioner ->initialize(navier_stokes_preconditioner_matrix.block(0,0));
  Mp_preconditioner  ->initialize(navier_stokes_preconditioner_matrix.block(1,1));

  const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
        TrilinosWrappers::PreconditionJacobi>
        preconditioner_no (navier_stokes_matrix, navier_stokes_preconditioner_matrix,
                           *Mp_preconditioner, *Amg_preconditioner, false);


  /* BENCHMARK*/
  {
    auto v = distributed_navier_stokes_solution;
    for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
      {
        v(i)=1;
        distributed_navier_stokes_solution(i) = 1;
      }

    auto dim_max = distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size()-1;

    timer.start();
    for (unsigned int i = 0; i<reps ; ++i)
      {
        if (i < dim_max)
          {
            v(i) = 2;
            v(dim_max - i) = 2;
          }
        preconditioner_no.vmult(distributed_navier_stokes_solution, v);
      }
    timer.stop();
    solution = distributed_navier_stokes_solution;
  }
  return timer.wall_time();
}

double test_class::vmult_linear_operators_from_zero(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution)
{
  Timer timer(MPI_COMM_WORLD, /* sync_wall_time = */ true);
  TrilinosWrappers::MPI::BlockVector
  distributed_navier_stokes_solution (navier_stokes_rhs);
  distributed_navier_stokes_solution = navier_stokes_solution;

  const unsigned int
  start = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().first);
  const unsigned int
  end   = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().second);

  for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
    distributed_navier_stokes_solution(i) = 0;


  // PRECONDITIONER:
  Amg_preconditioner ->initialize(navier_stokes_preconditioner_matrix.block(0,0));
  Mp_preconditioner  ->initialize(navier_stokes_preconditioner_matrix.block(1,1));


  /*   NEW STYLE   */
  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,0) );
  auto ZeroP = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_preconditioner_matrix.block(1,1) );
  auto Amg   = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_preconditioner_matrix.block(0,0) );

  TrilinosWrappers::MPI::Vector utmp(distributed_navier_stokes_solution.block(0));


  SolverControl solver_control_pre(5000, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);

  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);
  auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);

  auto P00 = Amg;
  auto P01 = Amg * Bt * Schur_inv;
  auto P10 = 0 * B;
  auto P11 = -1 * Schur_inv;

  const auto Mat = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >(
  {
    {
      {{ A, Bt }} ,
      {{ B, ZeroP }}
    }
  } );

  const auto DiagInv = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >(
  {
    {
      {{ Amg, Bt }} ,
      {{ B, -1*Schur_inv }}
    }
  } );
  // const auto DiagInv = block_diagonal_operator<2 , TrilinosWrappers::MPI::BlockVector >(
  //       {{
  //         Amg, -1 * Schur_inv
  //       }} );

  const auto P_inv = block_back_substitution<TrilinosWrappers::MPI::BlockVector>(
                       Mat, DiagInv
                     );


  /* BENCHMARK*/
  {
    auto v = distributed_navier_stokes_solution;
    for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
      {
        v(i)=1;
        distributed_navier_stokes_solution(i) = 1;
      }

    P_inv.reinit_range_vector(solution,true);

    auto dim_max = distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size()-1;

    timer.start();
    for (unsigned int i = 0; i<reps ; ++i)
      {
        if (i < dim_max)
          {
            v(i) = 2;
            v(dim_max - i) = 2;
          }
        P_inv.vmult(solution, v);
      }
    timer.stop();

  }
  return timer.wall_time();
}

double test_class::vmult_linear_operators_without_null(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution)
{
  Timer timer(MPI_COMM_WORLD, /* sync_wall_time = */ true);
  TrilinosWrappers::MPI::BlockVector
  distributed_navier_stokes_solution (navier_stokes_rhs);
  distributed_navier_stokes_solution = navier_stokes_solution;

  const unsigned int
  start = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().first);
  const unsigned int
  end   = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().second);

  for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
    distributed_navier_stokes_solution(i) = 0;


  // PRECONDITIONER:
  Amg_preconditioner ->initialize(navier_stokes_preconditioner_matrix.block(0,0));
  Mp_preconditioner  ->initialize(navier_stokes_preconditioner_matrix.block(1,1));


  /*   NEW STYLE   */
  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,0) );
  auto ZeroP = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_preconditioner_matrix.block(1,1) );
  auto Amg   = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_preconditioner_matrix.block(0,0) );

  TrilinosWrappers::MPI::Vector utmp(distributed_navier_stokes_solution.block(0));


  SolverControl solver_control_pre(5000, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);

  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);
  // auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);

  auto P00 = Amg;
  auto P01 = Amg * Bt * Schur_inv;
  auto P10 = 0*B;
  auto P11 = -1 * Schur_inv;

  const auto P_inv = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });

  /* BENCHMARK*/
  {
    auto v = distributed_navier_stokes_solution;
    for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
      {
        v(i)=1;
        distributed_navier_stokes_solution(i) = 1;
      }

    P_inv.reinit_range_vector(solution,true);

    auto dim_max = distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size()-1;

    timer.start();
    for (unsigned int i = 0; i<reps ; ++i)
      {
        if (i < dim_max)
          {
            v(i) = 2;
            v(dim_max - i) = 2;
          }
        P_inv.vmult(solution, v);
      }
    timer.stop();
  }

  return timer.wall_time();
}

double test_class::vmult_linear_operators_with_null(unsigned int reps, TrilinosWrappers::MPI::BlockVector &solution)
{
  Timer timer(MPI_COMM_WORLD, /* sync_wall_time = */ true);
  TrilinosWrappers::MPI::BlockVector
  distributed_navier_stokes_solution (navier_stokes_rhs);
  distributed_navier_stokes_solution = navier_stokes_solution;

  const unsigned int
  start = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().first);
  const unsigned int
  end   = (distributed_navier_stokes_solution.block(0).size() +
           distributed_navier_stokes_solution.block(1).local_range().second);

  for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
    distributed_navier_stokes_solution(i) = 0;


  // PRECONDITIONER:
  Amg_preconditioner ->initialize(navier_stokes_preconditioner_matrix.block(0,0));
  Mp_preconditioner  ->initialize(navier_stokes_preconditioner_matrix.block(1,1));


  /*   NEW STYLE   */
  // SYSTEM MATRIX:
  auto A  = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,0) );
  auto Bt = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(0,1) );
  //  auto B =  transpose_operator(Bt);
  auto B     = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,0) );
  auto ZeroP = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_matrix.block(1,1) );

  auto Mp    = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_preconditioner_matrix.block(1,1) );
  auto Amg   = linear_operator< TrilinosWrappers::MPI::Vector >( navier_stokes_preconditioner_matrix.block(0,0) );

  TrilinosWrappers::MPI::Vector utmp(distributed_navier_stokes_solution.block(0));


  SolverControl solver_control_pre(5000, 1e-12);
  SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);

  auto Schur_inv = inverse_operator( Mp, solver_CG, *Mp_preconditioner);
  // auto A_inv     = inverse_operator( A, solver_CG, *Amg_preconditioner);

  auto P00 = Amg;
  auto P01 = Amg * Bt * Schur_inv;
  auto P10 = null_operator(B);
  auto P11 = -1 * Schur_inv;

  const auto P_inv = block_operator<2, 2, TrilinosWrappers::MPI::BlockVector >({{
      {{ P00, P01 }} ,
      {{ P10, P11 }}
    }
  });

  /* BENCHMARK*/
  {
    auto v = distributed_navier_stokes_solution;
    for (unsigned int i =0 ; i < distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size(); ++i)
      {
        v(i)=1;
        distributed_navier_stokes_solution(i) = 1;
      }

    P_inv.reinit_range_vector(solution,true);

    auto dim_max = distributed_navier_stokes_solution.block(0).size() + distributed_navier_stokes_solution.block(1).size()-1;

    timer.start();
    for (unsigned int i = 0; i<reps ; ++i)
      {
        if (i < dim_max)
          {
            v(i) = 2;
            v(dim_max - i) = 2;
          }
        P_inv.vmult(solution, v);
      }
    timer.stop();
  }

  return timer.wall_time();
}

int main (int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  deallog.depth_console (0);

  const int dim = 2;
  test_class test_problem(NavierStokes<2>::global_refinement);

  ParameterAcceptor::initialize("./parameters/benchmark.prm");
  ParameterAcceptor::prm.log_parameters(deallog);

  test_problem.run();

  return 0;
}
