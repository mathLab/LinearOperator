/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2014 by the deal.II authors
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
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include "linear_operator.h"
#include "utilities.h"

using namespace dealii;

template <int dim>
class Step6
{
public:
  Step6 ();
  ~Step6 ();

  void run ();

private:
  void setup_system ();

  void assemble_system ();
  void assemble_system_lo ();

  void solve ();
  void solve_lo ();

  void refine_grid ();

  Triangulation<dim>   triangulation;
  Triangulation<dim>   triangulation_lo;

  DoFHandler<dim>      dof_handler;
  DoFHandler<dim>      dof_handler_lo;

  FE_Q<dim>            fe;
  FE_Q<dim>            fe_lo;

  ConstraintMatrix     constraints;
  ConstraintMatrix     constraints_lo;

  SparsityPattern      sparsity_pattern;
  SparsityPattern      sparsity_pattern_lo;

  SparseMatrix<double> system_matrix;
  SparseMatrix<double> system_matrix_lo;

  Vector<double>       solution;
  Vector<double>       solution_lo;

  Vector<double>       system_rhs;
  Vector<double>       system_rhs_lo;
};




template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;
};



template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}



template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points,
          ExcDimensionMismatch (values.size(), n_points));

  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  for (unsigned int i=0; i<n_points; ++i)
    {
      if (points[i].square() < 0.5*0.5)
        values[i] = 20;
      else
        values[i] = 1;
    }
}




template <int dim>
Step6<dim>::Step6 ()
  :
  dof_handler (triangulation),
  dof_handler_lo (triangulation_lo),
  fe (2),
  fe_lo (2)
{}



template <int dim>
Step6<dim>::~Step6 ()
{
  dof_handler.clear ();
}



template <int dim>
void Step6<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  solution.reinit (dof_handler.n_dofs());
  solution_lo.reinit (dof_handler.n_dofs());

  system_rhs.reinit (dof_handler.n_dofs());
  system_rhs_lo.reinit (dof_handler.n_dofs());

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);


  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ConstantFunction<dim>(1.),
                                            constraints);


  constraints.close ();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  constraints_lo.clear ();
  constraints_lo.close ();

  DynamicSparsityPattern dsp_lo(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp_lo,
                                  constraints_lo,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern_lo.copy_from(dsp_lo);

  system_matrix_lo.reinit (sparsity_pattern_lo);
}



template <int dim>
void Step6<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(3);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values (n_q_points);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      coefficient.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (coefficient_values[q_index] *
                                   fe_values.shape_grad(i,q_index) *
                                   fe_values.shape_grad(j,q_index) *
                                   fe_values.JxW(q_index));

            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            1.0 *
                            fe_values.JxW(q_index));
          }

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
}

template <int dim>
void Step6<dim>::assemble_system_lo ()
{
  const QGauss<dim>  quadrature_formula(3);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values (n_q_points);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      coefficient.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) +=  (coefficient_values[q_index] *
                                    fe_values.shape_grad(i,q_index) *
                                    fe_values.shape_grad(j,q_index) *
                                    fe_values.JxW(q_index));

            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            1.0 *
                            fe_values.JxW(q_index));
          }

      cell->get_dof_indices (local_dof_indices);
      constraints_lo.distribute_local_to_global (cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix_lo,
                                                 system_rhs_lo);
    }

}


template <int dim>
void Step6<dim>::solve ()
{
  // constraints.condense(system_matrix, system_rhs);
  unsigned int n_iterations = 0;
  auto M = linear_operator(system_matrix);

  SolverControl      solver_control (10000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  auto M_inv = inverse_operator(  M,
                                  solver,
                                  preconditioner);



  M_inv.vmult(solution, system_rhs);
  n_iterations = solver_control.last_step();
  constraints.distribute (solution);
  // std::cout << n_iterations  << " iterations old style."
  //     << std::endl;
}


template <int dim>
void Step6<dim>::solve_lo ()
{
  unsigned int n_iterations = 0;

  auto new_system_rhs_lo = constrained_rhs< >(
                             constraints, system_matrix_lo, system_rhs_lo);;
  auto M = constrained_linear_operator< >( constraints, system_matrix_lo );

  SolverControl      solver_control (10000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2); // TODO: Riscrivere questa riga
  // usando i LinearOperator

  auto M_inv = inverse_operator(  M,
                                  solver,
                                  preconditioner);

  M_inv.vmult(solution_lo, new_system_rhs_lo);
  n_iterations = solver_control.last_step();

  constraints.distribute(solution_lo);

  // std::cout << n_iterations  << " iterations LinearOperator mode."
  //     << std::endl;

  // auto rhs_tmp = system_rhs;
  // new_system_rhs_lo.apply(rhs_tmp);
  // compare_solution(system_rhs, rhs_tmp, dof_handler.n_dofs(), 1e-6, false, false, "RHS");

}



template <int dim>
void Step6<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);

  triangulation.execute_coarsening_and_refinement ();
  // triangulation.refine_global ();
}





template <int dim>
void Step6<dim>::run ()
{
  // unsigned int rep = 20;

  std::cout << std::setw( 7 ) << "  "
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

  std::cout << std::setw( 7 ) << " dim "
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

  std::cout << std::setw( 7 ) << "-------"
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

  for (unsigned int cycle=0; cycle<40; ++cycle)
    {
      // std::cout << std::endl
      // << std::endl
      // << std::endl
      // <<  "Cycle " << cycle << ':' << std::endl << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_ball (triangulation);

          static const SphericalManifold<dim> boundary;
          triangulation.set_all_manifold_ids_on_boundary(0);
          triangulation.set_manifold (0, boundary);

          triangulation.refine_global (1);
        }
      else
        refine_grid ();


      // std::cout << "   Number of active cells:       "
      //           << triangulation.n_active_cells()
      //           << std::endl;

      setup_system ();

      // std::cout << "   Number of degrees of freedom: "
      //           << dof_handler.n_dofs()
      //           << std::endl;

      for (unsigned rep = 500; rep<1000; rep*=10)
        {
          Timer timer;
          timer.start();
          for (unsigned int i = 0; i<rep ; i++)
            {
              assemble_system ();
              solve ();
            }
          timer.stop();

          Timer timer_lo;
          timer_lo.start();
          for (unsigned int i = 0; i<rep ; i++)
            {
              assemble_system_lo ();
              solve_lo ();
            }
          timer_lo.stop();
          // compare_solution(solution, solution_lo, dof_handler.n_dofs(), 1e-6, false, false, "solution");

          std::cout << std::setw( 7 ) << dof_handler.n_dofs()
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
    }

  DataOutBase::EpsFlags vtk_flags;
  vtk_flags.z_scaling = 4;

  // New implementation
  DataOut<dim> data_out_lo;
  data_out_lo.set_flags (vtk_flags);

  data_out_lo.attach_dof_handler (dof_handler);
  data_out_lo.add_data_vector (solution_lo, "solution_lo");
  data_out_lo.build_patches ();

  std::ofstream output_lo ("final-solution_lo.vtk");
  data_out_lo.write_vtk (output_lo);

  // Old implementation
  DataOut<dim> data_out;
  data_out.set_flags (vtk_flags);

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output ("final-solution.vtk");
  data_out.write_vtk (output);
}



int main ()
{

  try
    {
      deallog.depth_console (0);

      Step6<2> laplace_problem_2d;
      laplace_problem_2d.run ();
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
