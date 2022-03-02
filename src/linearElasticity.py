"""
 * This file is a part of fenicsFEA, and is distributed under the
 * GNU General Public License v3.0.
 *
 * FEModel: Isotropic linear elasticity
 * 
 * Usage: python3 linearElasticity.py <model_to_run>
 *        mpirun -np <no_of_procs> python3 linearElasticity.py <model_to_run>
 *
 * Features: 
 *           
 * 
 * Author:   Ritukesh Bharali, ritukesh.bharali@chalmers.se
 *           Chalmers University of Technology
 *
 * Date:     Mon 24 Feb 16:10:07 CET 2022
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
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs" # quadrature
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


# ---------------------------------------------------------------#
# Setup MPI
# ---------------------------------------------------------------#

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parameters["std_out_all_processes"]  = False;             # Terminal output on only rank 0 
parameters["mesh_partitioner"]       = "SCOTCH"           # options: ParMETIS, SCOTCH
parameters["linear_algebra_backend"] = "PETSc"            # options: uBLAS, Tpetra, PETSc
parameters["ghost_mode"]             = "shared_facet"     # options: none, shared_vertex, shared_facet



# ---------------------------------------------------------------#
# Get model to run and print some info
# ---------------------------------------------------------------#

nargs = len(sys.argv)

if nargs != 2:
    if rank == 0:
        print("Running the default model: beam!")
    inputCase = "beam"
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

# Derived quantities
mu    = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu)*(1.0 - 2.0 * nu))


# Constraints
fix_edge     = problem_data.get("constraints").get("fix_edge")
fix_dofs     = problem_data.get("constraints").get("fix_dofs")
pres_edge    = problem_data.get("constraints").get("pres_edge")
pres_dofs    = problem_data.get("constraints").get("pres_dofs")

# Stepping
dt           = problem_data.get("stepping").get("dt")
t_max_steps  = problem_data.get("stepping").get("t_max_steps")

# Nonlinear solver
nl_tol       = problem_data.get("nl_solver").get("nl_tol")
max_iter     = problem_data.get("nl_solver").get("max_iter")

# Post-processing
lodi         = problem_data.get("post_process").get("lodi")
lodi_edge    = problem_data.get("post_process").get("lodi_edge")
lodi_ldof    = problem_data.get("post_process").get("lodi_ldof")



# ---------------------------------------------------------------#
# Some utility functions
# ---------------------------------------------------------------#

def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))

def sigma(u):
    """Stress tensor of the elastic material as a function of the displacement"""
    return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)



# ---------------------------------------------------------------#
# Setting up function spaces and functions
# ---------------------------------------------------------------#

# Reference element
element_u = ufl.VectorElement("Lagrange",mesh.ufl_cell(),degree=1,dim=ndim)

# Function space
V_u       = FunctionSpace(mesh,element_u)

# Test function
u_        = TestFunction(V_u)

# Trial function
du_       = TrialFunction(V_u)

# Solution
u         = Function(V_u)



# ---------------------------------------------------------------#
# Setting up boundary conditions
# ---------------------------------------------------------------#

u_pres  = Expression("t", t = 0.0, degree=1)
count   = 0
edge    = [None] * (len(fix_dofs) + len(pres_dofs))
bcs_u   = [None] * (len(fix_dofs) + len(pres_dofs))

# Dirichlet constraints (homogeneous)
for i in range(len(fix_dofs)):
    edge[count]  = CompiledSubDomain(fix_edge[i], tol=1e-4)
    dof          = fix_dofs[i]
    if dof == "dx":
        bcs_u[count] = DirichletBC(V_u.sub(0), Constant((0.0)), edge[count])
    elif dof == "dy":
        bcs_u[count] = DirichletBC(V_u.sub(1), Constant((0.0)), edge[count])
    elif dof == "dz":
        bcs_u[count] = DirichletBC(V_u.sub(2), Constant((0.0)), edge[count])    
    else:
        raise Exception("Choose dx, dy, dz for displacement dofs!!") 
    count += 1    

# Dirichlet constraints (inhomogeneous)
for i in range(len(pres_dofs)):
    edge[count]  = CompiledSubDomain(pres_edge[i], tol=1e-4)
    dof          = pres_dofs[i]
    if dof == "dx":
        bcs_u[count] = DirichletBC(V_u.sub(0), u_pres, edge[count])
    elif dof == "dy":
        bcs_u[count] = DirichletBC(V_u.sub(1), u_pres, edge[count])
    elif dof == "dz":
        bcs_u[count] = DirichletBC(V_u.sub(2), u_pres, edge[count])    
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

# Elastic energy
total_energy      = 0.5 * ufl.inner(sigma(u), eps(u)) * dx 

# Extract the displacement sub-problem (Residual and Jacobian)
F_u               = ufl.derivative(total_energy,u,u_)
J_u               = ufl.derivative(F_u, u, du_)
problem_u         = NonlinearVariationalProblem(F_u, u,bcs_u,J_u)



# ---------------------------------------------------------------#
# Setting up solvers
# ---------------------------------------------------------------#

# Displacement and phase-field solvers
solver_u          = NonlinearVariationalSolver(problem_u)

# Newton-Raphson
solver_u.parameters['newton_solver']['convergence_criterion']   = 'incremental'
solver_u.parameters["newton_solver"]["linear_solver"]           = "gmres"
solver_u.parameters["newton_solver"]["preconditioner"]          = "petsc_amg"
solver_u.parameters["newton_solver"]["maximum_iterations"]      = max_iter
solver_u.parameters["newton_solver"]["error_on_nonconvergence"] = True



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

# Load-displacement data on root process
if rank == 0:
    lodi_out = open(out_dir+"/lodi.txt", 'w')

# XDMF output
xdmf_out = XDMFFile (mesh.mpi_comm(), out_dir+"/output.xdmf")

# VTK mesh partition
File(out_dir+"/meshPartition.pvd").write(
        MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.mpi_comm().rank))



# ---------------------------------------------------------------#
# Begin time-stepping
# ---------------------------------------------------------------#

tic = time.perf_counter()

# Staggered scheme
step     = 0
t        = 0.0

while True:

    step    += 1
    t       += dt
    u_pres.t = t

    if rank == 0:
        print("==============================")
        print("  Step:",step," Displacement:", "%.7g" % (t) )
        print("------------------------------")
        print("  Iteration     Error")
        sys.stdout.flush()
    
    (iter,conv) = solver_u.solve()

    traction   = dot(sigma(u),n)
    local_fy   = assemble(traction[lodi_ldof]*ds(lodi_ldof))
    fy         = MPI.COMM_WORLD.allreduce(local_fy, op=MPI.SUM)/size

    if rank == 0:
        print("------------------------------")
        print("Converged in", iter, " iterations!")
        print("Load:", "%.7g" % (fy), "\n")
        sys.stdout.flush()

        lodi_out.write(str(t)+"\t")
        lodi_out.write(str(fy)+"\n")

    xdmf_out.rename("u","u")
    xdmf_out.write(u,step)

    if step == t_max_steps:
        break

toc = time.perf_counter()

if rank == 0:
    print ('Simulation completed in', "%.4g" % (toc-tic), " seconds" ) 


# Show the final displacement field in loading direction
# only if run on single process
if size == 1:
    p=plot(u[lodi_ldof])
    plt.colorbar(p);
    plt.show(block=False)
    plt.pause(5)
    plt.close()

# ---------------------------------------------------------------#
# End time-stepping
# ---------------------------------------------------------------#