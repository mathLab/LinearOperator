/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2014 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2009, 2010
 *         Timo Heister, University of Goettingen, 2009, 2010
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

namespace LA
{
  using namespace dealii::TrilinosWrappers;
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include "./linear_operator_parallel.h"
#include "./utilities.h"

#include <fstream>
#include <iostream>

namespace Step40
{
  using namespace dealii;


  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    ~LaplaceProblem ();

    void run ();

  private:
    void setup_system ();
    void assemble_system ();
    void solve ();
    void solve_lo ();
    void refine_grid ();

    MPI_Comm                                  mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    DoFHandler<dim>                           dof_handler;
    FE_Q<dim>                                 fe;

    IndexSet                                  locally_owned_dofs;
    IndexSet                                  locally_relevant_dofs;

    ConstraintMatrix                          constraints;
    ConstraintMatrix                          constraints_lo;

    LA::MPI::Vector                           system_rhs;
    LA::MPI::Vector                           system_rhs_lo;

    LA::SparseMatrix                          system_matrix;
    LA::SparseMatrix                          system_matrix_lo;

    LA::MPI::Vector                           locally_relevant_solution;


    ConditionalOStream                        pcout;
    // TimerOutput                               computing_timer;
  };




  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    fe (2),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0))
    //computing_timer (mpi_communicator,
    //                 pcout,
    // TimerOutput::summary,
    // TimerOutput::wall_times)
  {}



  template <int dim>
  LaplaceProblem<dim>::~LaplaceProblem ()
  {
    dof_handler.clear ();
  }



  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    // TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs (fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);
    system_rhs_lo.reinit (locally_owned_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);

    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close ();

    DynamicSparsityPattern dsp (locally_relevant_dofs);

    DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                     constraints, false);
    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);
    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);

    constraints_lo.clear ();
    constraints_lo.reinit (locally_relevant_dofs);
    constraints_lo.close ();

    DynamicSparsityPattern dsp_lo(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp_lo,
                                    constraints_lo,
                                    /*keep_constrained_dofs = */ false);

    SparsityTools::distribute_sparsity_pattern (dsp_lo,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);


    system_matrix_lo.reinit (locally_owned_dofs,
                             locally_owned_dofs,
                             dsp_lo,
                             mpi_communicator);

  }




  template <int dim>
  void LaplaceProblem<dim>::assemble_system ()
  {
    // TimerOutput::Scope t(computing_timer, "assembly");

    const QGauss<dim>  quadrature_formula(3);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit (cell);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
              const double
              rhs_value
                = (fe_values.quadrature_point(q_point)[1]
                   >
                   0.5+0.25*std::sin(4.0 * numbers::PI *
                                     fe_values.quadrature_point(q_point)[0])
                   ? 1 : -1);

              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                                         fe_values.shape_grad(j,q_point) *
                                         fe_values.JxW(q_point));

                  cell_rhs(i) += (rhs_value *
                                  fe_values.shape_value(i,q_point) *
                                  fe_values.JxW(q_point));
                }
            }

          cell->get_dof_indices (local_dof_indices);
          constraints_lo.distribute_local_to_global (cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix_lo,
                                                     system_rhs_lo);
          constraints.distribute_local_to_global (cell_matrix,
                                                  cell_rhs,
                                                  local_dof_indices,
                                                  system_matrix,
                                                  system_rhs);
        }

    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);
    system_matrix_lo.compress (VectorOperation::add);
    system_rhs_lo.compress (VectorOperation::add);
  }




  template <int dim>
  void LaplaceProblem<dim>::solve ()
  {
    // TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector
    completely_distributed_solution (locally_owned_dofs, mpi_communicator);

    SolverControl solver_control (dof_handler.n_dofs(), 1e-12);

    LA::SolverCG solver(solver_control);
    LA::PreconditionAMG preconditioner;

    LA::PreconditionAMG::AdditionalData data;

    preconditioner.initialize(system_matrix, data);

    solver.solve (system_matrix, completely_distributed_solution, system_rhs,
                  preconditioner);

    // pcout << "   Solved in " << solver_control.last_step()
    //       << " iterations." << std::endl;

    constraints.distribute (completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }

  template <int dim>
  void LaplaceProblem<dim>::solve_lo ()
  {
    // ConstraintLinearOperator< LA::MPI::Vector, LA::MPI::Vector, LA::SparseMatrix >
    //                     CM (constraints, system_matrix_lo, system_rhs_lo,locally_owned_dofs, mpi_communicator);
    auto rhs = constrained_rhs< LA::MPI::Vector, LA::MPI::Vector, LA::SparseMatrix >(constraints, system_matrix_lo, system_rhs_lo, locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    auto CM  = constrained_linear_operator< LA::MPI::Vector, LA::MPI::Vector, LA::SparseMatrix >(constraints, system_matrix_lo, locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    // TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector
    completely_distributed_solution (locally_owned_dofs, mpi_communicator);

    SolverControl solver_control (1000, 1e-12);

    // SolverCG solver(solver_control, mpi_communicator);
    SolverCG< LA::MPI::Vector > solver_new(solver_control);

    std_cxx11::shared_ptr< TrilinosWrappers::PreconditionAMG >    preconditioner;
    preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
    preconditioner->initialize(system_matrix);

    // auto M       = linear_operator< TrilinosWrappers::MPI::Vector >(system_matrix);
    auto M_inv   = inverse_operator(  CM,
                                      solver_new,
                                      *preconditioner);

    // system_rhs_lo = CM.get_constraint_rhs();
    M_inv.vmult(completely_distributed_solution, rhs);

    // pcout << "   Solved in " << solver_control.last_step()
    //       << " iterations." << std::endl;

    constraints.distribute (completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }



  template <int dim>
  void LaplaceProblem<dim>::refine_grid ()
  {
    // TimerOutput::Scope t(computing_timer, "refine");

    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(3),
                                        typename FunctionMap<dim>::type(),
                                        locally_relevant_solution,
                                        estimated_error_per_cell);
    parallel::distributed::GridRefinement::
    refine_and_coarsen_fixed_number (triangulation,
                                     estimated_error_per_cell,
                                     0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
  }

  template <int dim>
  void LaplaceProblem<dim>::run ()
  {
    pcout     << std::setw( 7 ) << "  "
              << std::setw( 3 )  << " | "
              << std::setw( 7 ) << "  "
              << std::setw( 3 )  << " | "
              << std::setw( 14 ) << "  "
              << std::setw( 14 ) << " WALLTIME "
              << std::setw( 14 ) << "  "
              << std::setw( 3 )  << " | "
              << std::setw( 14 ) << "  "
              << std::setw( 14 ) << " CPUTIME "
              << std::setw( 14 ) << "  "
              << std::endl << std::flush;

    pcout     << std::setw( 7 ) << " dim "
              << std::setw( 3 )  << " | "
              << std::setw( 7 ) << " rep "
              << std::setw( 3 )  << " | "
              << std::setw( 14 ) << " time "
              << std::setw( 14 ) << " LO time "
              << std::setw( 14 ) << " Efficiency "
              << std::setw( 3 )  << " | "
              << std::setw( 14 ) << " time "
              << std::setw( 14 ) << " LO time "
              << std::setw( 14 ) << " Efficiency "
              << std::endl << std::flush;

    pcout     << std::setw( 7 ) << "-------"
              << std::setw( 3 )  << "-+-"
              << std::setw( 7 ) << "-------"
              << std::setw( 3 )  << "-+-"
              << std::setw( 14 ) << "--------------"
              << std::setw( 14 ) << "--------------"
              << std::setw( 14 ) << "--------------"
              << std::setw( 3 )  << "-+-"
              << std::setw( 14 ) << "--------------"
              << std::setw( 14 ) << "--------------"
              << std::setw( 14 ) << "--------------"
              << std::endl << std::flush;

    const unsigned int n_cycles = 40;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        // pcout << "Cycle " << cycle << ':' << std::endl;


        if (cycle == 0)
          {
            GridGenerator::hyper_cube (triangulation);
            triangulation.refine_global (4);
          }
        else
          refine_grid ();

        setup_system ();

        // pcout << "   Number of active cells:       "
        //       << triangulation.n_global_active_cells()
        //       << std::endl
        //       << "   Number of degrees of freedom: "
        //       << dof_handler.n_dofs()
        //       << std::endl;

        assemble_system ();

        for (unsigned rep = 500; rep < 1000; rep*=10)
          {
            Timer timer(mpi_communicator, /* sync_wall_time = */ true);
            timer.start();
            for (unsigned int i = 0; i<rep ; i++)
              {
                solve ();
              }
            timer.stop();

            Timer timer_lo(mpi_communicator, /* sync_wall_time = */ true);
            timer_lo.start();
            for (unsigned int i = 0; i<rep ; i++)
              {
                solve_lo ();
              }
            timer_lo.stop();
            // compare_solution(solution, solution_lo, dof_handler.n_dofs(), 1e-6, false, false, "solution");

            pcout  << std::setw( 7 ) << dof_handler.n_dofs()
                   << std::setw( 3 )  << " | "
                   << std::setw( 7 ) << rep
                   << std::setw( 3 )  << " | "
                   << std::setw( 14 ) << timer.wall_time()
                   << std::setw( 14 ) << timer_lo.wall_time()
                   << std::setw( 14 ) << timer.wall_time()/timer_lo.wall_time()
                   << std::setw( 3 )  << " | "
                   << std::setw( 14 ) << timer()
                   << std::setw( 14 ) << timer_lo()
                   << std::setw( 14 ) << timer()/timer_lo()
                   << std::endl << std::flush;
          }
        // solve ();

        // if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
        //   {
        //     // TimerOutput::Scope t(computing_timer, "output");
        //     output_results (cycle);
        //   }

        // solve_lo ();

        // if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
        //   {
        //     // TimerOutput::Scope t(computing_timer, "output_lo");
        //     output_results_lo (cycle);
        //   }

        // computing_timer.print_summary ();
        // computing_timer.reset ();

        // pcout << std::endl;
      }
  }
}




int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step40;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      deallog.depth_console (0);

      {
        LaplaceProblem<2> laplace_problem_2d;
        laplace_problem_2d.run ();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
