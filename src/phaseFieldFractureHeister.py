"""
 * This file is a part of fenicsFEA, and is distributed under the
 * GNU General Public License v3.0.
 *
 * FEModel: Phase-field fracture
 * 
 * Usage: python3 phaseFieldFractureHeister.py <model_to_run>
 *        mpirun -np <no_of_procs> python3 phaseFieldFractureHeister.py <model_to_run>
 *
 * Features: 
 *           Energy split             - Amor (doi:10.1016/j.jmps.2009.04.011)
 *           Fracture irreversibility - Penalization (doi:10.1016/j.cma.2019.05.038)
 *           Solution strategy        - Monolithic
 *           Convexification          - Phase-field extrapolation (doi:10.1016/j.cma.2015.03.009)
 * 
 * Author:   Ritukesh Bharali, ritukesh.bharali@chalmers.se
 *           Chalmers University of Technology
 *
 * Date:     Fri 25 Feb 15:50:11 CET 2022
 *
 * Updates (what, who, when):
 *
 *
"""

# ---------------------------------------------------------------#
# Import standard libraries and utility classes
# ---------------------------------------------------------------#

# (legacy) fenics
from dolfin import *
import ufl

# MPI and linear solver
from mpi4py import MPI
from petsc4py import PETSc

# Python
import os
import sys
import time
import json
import sympy
import shutil
import numpy as np
import matplotlib.pyplot as plt



# ---------------------------------------------------------------#
# Setup fenics parameters
# ---------------------------------------------------------------#

#set_log_level(40)  #ERROR=40, WARNING=30
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"



# ---------------------------------------------------------------#
# Setup MPI
# ---------------------------------------------------------------#

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parameters["std_out_all_processes"]  = False;           # Terminal output on only rank 0 
parameters["mesh_partitioner"]       = "SCOTCH"         # options: ParMETIS, SCOTCH
parameters["linear_algebra_backend"] = "PETSc"          # options: uBLAS, Tpetra, PETSc
parameters["ghost_mode"]             = "shared_facet"     # options: none, shared_vertex, shared_facet



# ---------------------------------------------------------------#
# Model to run
# ---------------------------------------------------------------#

nargs = len(sys.argv)

if nargs != 2:
    if rank == 0:
        print("Running the default model: sent!")
    inputCase = "sent"
else:
    inputCase = sys.argv[1] 



# ---------------------------------------------------------------#
# Load problem input data
# ---------------------------------------------------------------#

# Full path + part of file name
file_prefix  = "../input_data/"+inputCase+"/"

# Load mesh (parallel)
mesh         = Mesh(comm,file_prefix+"mesh.xml.gz")
ndim         = mesh.topology().dim()

# Load model input file
problem_data = json.load(open(file_prefix+"params.json","r"))

# Material parameters
E            = problem_data.get("material").get("E")
nu           = problem_data.get("material").get("nu")
Gc           = problem_data.get("material").get("Gc")
l0           = problem_data.get("material").get("l0")
pen_fact     = problem_data.get("material").get("penalty_factor")

# Derived quantities
mu    = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu)*(1.0 - 2.0 * nu))
K     = lmbda+2/3*mu

# Constraints
fix_edge     = problem_data.get("constraints").get("fix_edge")
fix_dofs     = problem_data.get("constraints").get("fix_dofs")
pres_edge    = problem_data.get("constraints").get("pres_edge")
pres_dofs    = problem_data.get("constraints").get("pres_dofs")

# Stepping
dt           = problem_data.get("stepping").get("dt")
dt2          = problem_data.get("stepping").get("dt2")
t_switch     = problem_data.get("stepping").get("t_switch")
t_fact       = problem_data.get("stepping").get("t_fact")
t_stop_ratio = problem_data.get("stepping").get("t_stop_ratio")

old_timestep = dt

# Nonlinear solver
nl_type      = problem_data.get("nl_solver").get("nl_type")
nl_tol       = problem_data.get("nl_solver").get("nl_tol")
max_iter     = problem_data.get("nl_solver").get("max_iter")

# Post-processing
lodi         = problem_data.get("post_process").get("lodi")
lodi_edge    = problem_data.get("post_process").get("lodi_edge")
lodi_ldof    = problem_data.get("post_process").get("lodi_ldof")



# ---------------------------------------------------------------#
# Some utility functions
# ---------------------------------------------------------------#


def w(d):
    """Dissipated energy function as a function of the phase-field """
    return d*d

def dw(d):
    """Derivate of dissipated energy function as a function of the phase-field """
    return 2.0*d    

def a(d, k_ell=1.e-12):
    """Stiffness modulation as a function of the damage """
    return (1 - d) ** 2 + k_ell

def da(d, k_ell=1.e-12):
    """Derivative of stiffness modulation as a function of the damage """
    return -2.0 * (1 - d)

def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))    

def sigma_p(u):
    """Positive stress tensor based on Amor split as a function of the displacement"""
    return 2.0 * mu * ufl.dev(eps(u)) + K * (0.5*(ufl.tr(eps(u))+abs(ufl.tr(eps(u))))) * ufl.Identity(ndim)

def sigma_n(u):
    """Negative stress tensor based on Amor split as a function of the displacement"""
    return K * (0.5*(ufl.tr(eps(u))-abs(ufl.tr(eps(u))))) * ufl.Identity(ndim)

def sigma(u):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(extra_pf(d0,d00)) * sigma_p(u) + sigma_n(u)

def pen(d,d0):
    """ Penalty term for fracture irreversibility"""
    return pen_fact * Gc / l0 * conditional(lt(d,d0),(d0-d),0.0)

def extra_pf(d0,d00):
    """Extrapolated phase-field"""
    return d00 + (d0-d00)*dt/old_timestep

# Compute the phase-field normalization constant
z     = sympy.Symbol("z")
c_w   = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1)) 



# ---------------------------------------------------------------#
# Setting up function spaces and functions
# ---------------------------------------------------------------#

# Reference element
element_u = ufl.VectorElement("Lagrange",mesh.ufl_cell(),degree=1,dim=ndim)
element_d = ufl.FiniteElement("Lagrange",mesh.ufl_cell(),degree=1)

# Function space
V         = FunctionSpace(mesh,element_u*element_d)

# Trial function
dv_       = TrialFunction(V)

# Test function
(u_, d_)  = TestFunctions(V)

# Current solution
v         = Function(V)
(u, d)    = split(v)  

# Old step solution
v0        = Function(V)
(u0, d0)  = split(v0)

# Old Old step solution
v00       = Function(V)
(u00,d00) = split(v00)



# ---------------------------------------------------------------#
# Setting up boundary conditions
# ---------------------------------------------------------------#

u_pres  = Expression("t", t = 0.0, degree=1)
count   = 0
edge    = [None] * (len(fix_dofs) + len(pres_dofs))
bcs     = [None] * (len(fix_dofs) + len(pres_dofs))

# Dirichlet constraints (homogeneous)
for i in range(len(fix_dofs)):
    edge[count]  = CompiledSubDomain(fix_edge[i], tol=1e-4)
    dof          = fix_dofs[i]
    if dof == "dx":
        bcs[count] = DirichletBC(V.sub(0).sub(0), Constant((0.0)), edge[count])
    elif dof == "dy":
        bcs[count] = DirichletBC(V.sub(0).sub(1), Constant((0.0)), edge[count])
    elif dof == "dz":
        bcs[count] = DirichletBC(V.sub(0).sub(2), Constant((0.0)), edge[count])
    elif dof == "dd":
        bcs[count] = DirichletBC(V.sub(1), Constant((0.0)), edge[count])
    else:
        raise Exception("Choose dx, dy, dz for displacement dofs!!") 
    count += 1    

# Dirichlet constraints (inhomogeneous)
for i in range(len(pres_dofs)):
    edge[count]  = CompiledSubDomain(pres_edge[i], tol=1e-4)
    dof          = pres_dofs[i]
    if dof == "dx":
        bcs[count] = DirichletBC(V.sub(0).sub(0), u_pres, edge[count])
    elif dof == "dy":
        bcs[count] = DirichletBC(V.sub(0).sub(1), u_pres, edge[count])
    elif dof == "dz":
        bcs[count] = DirichletBC(V.sub(0).sub(2), u_pres, edge[count])
    elif dof == "dd":
        bcs[count] = DirichletBC(V.sub(1), Constant((0.0)), edge[count])
    else:
        raise Exception("Choose dx, dy, dz for displacement dofs!!") 
    count += 1  

# Post-processing boundaries
lodi_ds = CompiledSubDomain(lodi_edge, tol=1e-4)
if lodi_ldof == "dx":
    lodi_ldof = 0
elif lodi_ldof == "dy":
    lodi_ldof = 1
elif lodi_ldof == "dz":
    lodi_ldof = 2
else:
    raise Exception("Choose dx, dy, dz for displacement dofs!!") 
boundaries = MeshFunction("size_t", mesh, ndim - 1)
lodi_ds.mark(boundaries,1)

# Integral measures and normal
dx         = ufl.Measure("dx",domain=mesh)
ds         = ufl.Measure("ds",domain=mesh)(subdomain_data=boundaries)
n          = FacetNormal(mesh)



# ---------------------------------------------------------------#
# Setting up the variational problem
# ---------------------------------------------------------------#

# Momentum balance equation
F1 = ufl.inner(sigma(u),eps(u_)) * dx

# Phase-field evolution
F2 = da(d) * 0.5*ufl.inner(sigma_p(u), eps(u)) * d_ * dx + \
     Gc / float(c_w) * (dw(d) * d_  / l0 + 
                    l0 * ufl.dot(ufl.grad(d), ufl.grad(d_))) * dx

# Penalty
F3 = - pen(d,d0) * d_ * dx

# Combined weak form
F = F1 + F2 + F3

# Jacobian
J = derivative(F,v,dv_) 

# Problem
problem = NonlinearVariationalProblem(F,v,bcs,J)     



# ---------------------------------------------------------------#
# Setting up solvers
# ---------------------------------------------------------------#

# Displacement and phase-field solvers
solver          = NonlinearVariationalSolver(problem)

# Solver configuration
if nl_type == "snes":

    # PETSc SNES
    solver.parameters['nonlinear_solver']                       = 'snes'
    solver.parameters["snes_solver"]["linear_solver"]           = "gmres"
    solver.parameters["snes_solver"]["preconditioner"]          = "petsc_amg"
    solver.parameters['snes_solver']['line_search']             = 'bt' 
    solver.parameters["snes_solver"]["maximum_iterations"]      = 30
    solver.parameters["snes_solver"]["report"]                  = True
    solver.parameters["snes_solver"]["error_on_nonconvergence"] = True
    solver.parameters["snes_solver"]["relative_tolerance"]      = 1e-6
    solver.parameters["snes_solver"]["absolute_tolerance"]      = 1e-4

else:

    # Newton-Raphson
    solver.parameters['newton_solver']['convergence_criterion']   = 'incremental'
    solver.parameters["newton_solver"]["linear_solver"]           = "gmres"
    solver.parameters["newton_solver"]["preconditioner"]          = "petsc_amg"
    solver.parameters["newton_solver"]["maximum_iterations"]      = 30
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    solver.parameters["newton_solver"]["relative_tolerance"]      = 1e-6
    solver.parameters["newton_solver"]["absolute_tolerance"]      = 1e-4



# ---------------------------------------------------------------#
# Setting up post-processing options
# ---------------------------------------------------------------#

# Delete existing output folder and create a new one
out_dir  = "../output/"+inputCase
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
    print("Deleted existing folder!")
os.makedirs(out_dir, exist_ok=False)

# Load-displacement data on root process
if rank == 0:
    lodi_out = open(out_dir+"/lodi.txt", 'w')

# VTK output
vtk_out = File(out_dir+"/output.pvd")

# VTK mesh partition
File(out_dir+"/meshPartition.pvd").write(
        MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.mpi_comm().rank))



# ---------------------------------------------------------------#
# Begin time-stepping
# ---------------------------------------------------------------#

# Start timer
tic = time.perf_counter()

# Some parameters
step         = 0
t            = 0.0
max_load     = 0.0
old_timestep = dt

# Start stepping
while True:

    step  += 1

    if t <= t_switch:
        t     += dt
        tstep  = dt
    else:
        t     += dt2
        tstep  = dt2

    u_pres.t = t

    if rank == 0:
        print("==============================")
        print("  Step:",step," Time:", "%.7g" % (t) )
        print("------------------------------")
        sys.stdout.flush()


    # Solve a step
    (iter,conv) = solver.solve()

    # Compute traction for load-displacement plot
    trac         = dot(sigma(u),n)
    local_trac   = assemble(trac[lodi_ldof]*ds(lodi_ldof))
    glob_trac    = MPI.COMM_WORLD.allreduce(local_trac, op=MPI.SUM)/size

    if rank == 0:
        print("------------------------------")
        print("Converged in", iter, " iterations!")
        print("Load:", "%.7g" % (glob_trac), "\n")
        sys.stdout.flush()

    # VTK Output
    u, d = v.split(deepcopy=True)
    vtk_out << (u,step)
    vtk_out << (d,step)


    # Update old step data
    v00.assign(v0)
    v0.assign(v)
    old_timestep = tstep

    # Compute ratio of current traction to max in loading history
    max_load    = max(abs(glob_trac),abs(max_load))
    ratio_load  = abs(glob_trac/max_load)

    # Criteria to break the time-stepping loop
    if ratio_load < t_stop_ratio:
        break

toc = time.perf_counter()

if rank == 0:
    print ('Simulation completed in', "%.4g" % (toc-tic), " seconds" ) 


# Show the final phase-field only if run on single process
if size == 1:
    p=plot(d)
    p.set_cmap("viridis")
    p.set_clim(0.0,1.0)
    plt.colorbar(p);
    plt.show()

# ---------------------------------------------------------------#
# End time-stepping
# ---------------------------------------------------------------#