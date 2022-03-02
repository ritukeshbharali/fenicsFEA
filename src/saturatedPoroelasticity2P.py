"""
 * This file is a part of fenicsFEA, and is distributed under the
 * GNU General Public License v3.0.
 *
 * FEModel: Fully saturated porous media with solid and fluid phases
 * 
 * Usage: python3 saturatedPoroelasticity2P.py <model_to_run>
 *        mpirun -np <no_of_procs> python3 saturatedPoroelasticity2P.py <model_to_run>
 *
 * Features: 
 *           Solver                   - Coupled, PETSc LU 
 * 
 * Author:   Ritukesh Bharali, ritukesh.bharali@chalmers.se
 *           Chalmers University of Technology
 *
 * Date:     Mon 22 Feb 19:38:00 CET 2022
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

set_log_level(40)  #ERROR=40, WARNING=30
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
parameters["linear_algebra_backend"] = "PETSc"          # options: uBLAS, Epetra, PETSc
parameters["ghost_mode"]             = "shared_facet"   # options: none, shared_vertex, shared_facet



# ---------------------------------------------------------------#
# Model to run
# ---------------------------------------------------------------#

nargs = len(sys.argv)

if nargs != 2:
    if rank == 0:
        print("Running the default model: terzaghi!")
    inputCase = "terzaghi"
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
n            = FacetNormal(mesh)

# Load model input file
problem_data = json.load(open(file_prefix+"params.json","r"))

# Element type
elem_type    = problem_data.get("elem_type")

# Material parameters
E            = problem_data.get("material").get("E")
nu           = problem_data.get("material").get("nu")
K            = problem_data.get("material").get("K")
Ks           = problem_data.get("material").get("Ks")
Kf           = problem_data.get("material").get("Kf")
phi          = problem_data.get("material").get("phi")
alpha        = problem_data.get("material").get("alpha")

# Derived quantities
G     = Constant(E / (2.0 * (1.0 + nu)))
lmbda = Constant(E * nu / ((1.0 + nu)*(1.0 - 2.0 * nu)))
M     = Constant(1./(phi/Kf + (alpha - phi)/Ks))

# Dirichlet constraints
fix_edge     = problem_data.get("constraints").get("fix_edge")
fix_dofs     = problem_data.get("constraints").get("fix_dofs")
pres_edge    = problem_data.get("constraints").get("pres_edge")
pres_dofs    = problem_data.get("constraints").get("pres_dofs")

# Neumann constraints
neumann_edge = problem_data.get("constraints").get("neumann_edge")
neumann_load = problem_data.get("constraints").get("neumann_load")
neumann_dof  = problem_data.get("constraints").get("neumann_dof")

# Stepping
dt           = problem_data.get("stepping").get("dt")
t_max_steps  = problem_data.get("stepping").get("t_max_steps")

# Nonlinear solver
snes         = problem_data.get("nl_solver").get("snes")
nl_tol       = problem_data.get("nl_solver").get("nl_tol")
max_iter     = problem_data.get("nl_solver").get("max_iter")



# ---------------------------------------------------------------#
# Some utility functions
# ---------------------------------------------------------------#

def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))  

def sigma(u):
    """Stress tensor as a function of the displacement"""
    return 2.0 * G * eps(u) + lmbda * ufl.div(u) * ufl.Identity(ndim)

# Initial Condition
class InitialConditions(UserExpression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0           # x displacement
        values[1] = 0           # y displacement
        values[2] = 0           # pressure

    def value_shape(self):
        return (3,)



# ---------------------------------------------------------------#
# Setting up function spaces and functions
# ---------------------------------------------------------------#

# Reference element
if elem_type == "Linear":
    element_u = ufl.VectorElement("Lagrange",mesh.ufl_cell(),degree=1,dim=ndim)
    element_p = ufl.FiniteElement("Lagrange",mesh.ufl_cell(),degree=1)
elif elem_type == "TaylorHood":
    element_u = ufl.VectorElement("Lagrange",mesh.ufl_cell(),degree=2,dim=ndim)
    element_p = ufl.FiniteElement("Lagrange",mesh.ufl_cell(),degree=1)
else:
    if rank == 0:
        print("Choosing TaylorHood elements are the default mixed element!")
    element_u = ufl.VectorElement("Lagrange",mesh.ufl_cell(),degree=2,dim=ndim)
    element_p = ufl.FiniteElement("Lagrange",mesh.ufl_cell(),degree=1)        

# Function space
W         = FunctionSpace(mesh,element_u*element_p)

# Trial function
dw_       = TrialFunction(W)

# Test function
(u_, p_)  = TestFunctions(W)

# Current solution
w         = Function(W)
(u, p)    = split(w)  

# Old step solution
w0        = Function(W)
(u0, p0)  = split(w0)  

# Create intial conditions and interpolate
w_init = InitialConditions(degree=1)
w0.interpolate(w_init)
w.interpolate(w_init)



# ---------------------------------------------------------------#
# Setting up boundary conditions
# ---------------------------------------------------------------#

count   = 0
edge    = [None] * len(fix_dofs)
bcs     = [None] * len(fix_dofs)

for i in range(len(fix_dofs)):
    edge[count]  = CompiledSubDomain(fix_edge[i], tol=1e-4)
    dof          = fix_dofs[i]
    if dof == "dx":
        bcs[count] = DirichletBC(W.sub(0).sub(0), Constant((0.0)), edge[count])
    elif dof == "dy":
        bcs[count] = DirichletBC(W.sub(0).sub(1), Constant((0.0)), edge[count])
    elif dof == "dz":
        bcs[count] = DirichletBC(W.sub(0).sub(2), Constant((0.0)), edge[count])
    elif dof == "dp":
        bcs[count] = DirichletBC(W.sub(1), Constant((0.0)), edge[count])
    else:
        raise Exception("Choose dx, dy, dz for displacement dofs!!") 
    count += 1    

# Boundaries
neumann_boundary = CompiledSubDomain(neumann_edge, tol=1e-4)
boundaries = MeshFunction("size_t", mesh, ndim - 1)
neumann_boundary.mark(boundaries,1)
ds  = ufl.Measure("ds",domain=mesh)(subdomain_data=boundaries)

if neumann_dof == 'dx':
    nbc_dof = 0
    t = Constant((neumann_load, 0.0))
elif neumann_dof == 'dy':
    nbc_dof = 1
    t = Constant((0.0, neumann_load))



# ---------------------------------------------------------------#
# Setting up the variational problem
# ---------------------------------------------------------------#

# Elastic weak form
F0 = ufl.inner(sigma(u),eps(u_)) * dx - alpha * p * div(u_) * dx \
   - inner(t,u_) * ds(nbc_dof)

# Fluid mass conservation weak form
F1 = (p/M + alpha*div(u)) * p_ * dx + dt * inner(K * grad(p), grad(p_)) * dx \
   - (p0/M + alpha * div(u0)) * p_ * dx

# Combined weak form   
F  = F0 + F1

# Jacobian
J  = derivative(F,w, dw_)

# Problem
problem = NonlinearVariationalProblem(F,w,bcs,J)



# ---------------------------------------------------------------#
# Setting up the variational solver
# ---------------------------------------------------------------#

solver  = NonlinearVariationalSolver(problem)

# Solver configuration
solver.parameters['newton_solver']['convergence_criterion']   = 'incremental'
solver.parameters["newton_solver"]["linear_solver"]           = "petsc"
solver.parameters["newton_solver"]["maximum_iterations"]      = 20
solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
solver.parameters["newton_solver"]["relative_tolerance"]      = 1e-4



# ---------------------------------------------------------------#
# Setting up post-processing options
# ---------------------------------------------------------------#

# Delete existing output folder and create a new one
out_dir  = "../output/"+inputCase
if rank == 0:
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        print("Deleted existing folder!")
        os.makedirs(out_dir, exist_ok=False)

# VTK output
vtk_out = File(out_dir+"/output.pvd")

# VTK mesh partition
File(out_dir+"/meshPartition.pvd").write(
        MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.mpi_comm().rank))



# ---------------------------------------------------------------#
# Begin time-stepping
# ---------------------------------------------------------------#

as_backend_type(w.vector()).update_ghost_values()

tic = time.perf_counter()

t    = 0.0
step = 0

while step < t_max_steps:

    t += dt
    step += 1

    print("Step:", step)

    (iter,conv) = solver.solve()

    if rank == 0:
        print("Converged in",iter,"iterations!\n")

    u, p = w.split(deepcopy=True)
    vtk_out << (u,step)
    vtk_out << (p,step)

    w0.assign(w)

toc = time.perf_counter()

if rank == 0:
    print ('Simulation completed in', "%.4g" % (toc-tic), " seconds" )

# ---------------------------------------------------------------#
# End time-stepping
# ---------------------------------------------------------------#        