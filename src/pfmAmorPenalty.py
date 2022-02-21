"""
 * This file is a part of fenicsFEA, and is distributed under the
 * GNU General Public License v3.0.
 *
 * FEModel: Phase-field fracture
 * 
 * Usage: python3 pfmAmorPenalty.py <model_to_run>
 *        mpirun -np <no_of_procs> python3 pfmAmorPenalty.py <model_to_run>
 *
 * Features: 
 *           Energy split             - Amor (doi:10.1016/j.jmps.2009.04.011)
 *           Fracture irreversibility - Penalization (doi:10.1016/j.cma.2019.05.038)
 *           Solver                   - Alternative minization (staggered)
 * 
 * Author:   Ritukesh Bharali, ritukesh.bharali@chalmers.se
 *           Chalmers University of Technology
 *
 * Date:     Mon 21 Feb 19:38:00 CET 2022
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
parameters["linear_algebra_backend"] = "PETSc"          # options: uBLAS, Tpetra, PETSc
parameters["ghost_mode"]             = "shared_facet"     # options: none, shared_vertex, shared_facet



# ---------------------------------------------------------------#
# Model to run
# ---------------------------------------------------------------#

nargs = len(sys.argv)

if nargs != 2:
    if rank == 0:
        print("Running the default model: SENT!")
    problem = "SENT"
else:
    problem = sys.argv[1]  



# ---------------------------------------------------------------#
# Load problem input data
# ---------------------------------------------------------------#

# Full path + part of file name
file_prefix  = "../input_data/"+problem+"/"+problem  

# Load mesh (parallel)
mesh         = Mesh(comm,file_prefix+"_mesh.xml.gz")
ndim         = mesh.topology().dim()

# Load model input file
problem_data = json.load(open(file_prefix+"_input.json","r"))

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

print(E,nu,lmbda,mu,K)


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

# Nonlinear solver
snes         = problem_data.get("nl_solver").get("snes")
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

def a(d, k_ell=1.e-12):
    """Stiffness modulation as a function of the damage """
    return (1 - d) ** 2 + k_ell

def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))    

def sigma_p(u):
    """Positive stress tensor based on Amor split as a function of the displacement"""
    return 2.0 * mu * ufl.dev(eps(u)) + K * (0.5*(ufl.tr(eps(u))+abs(ufl.tr(eps(u))))) * ufl.Identity(ndim)

def sigma_n(u):
    """Negative stress tensor based on Amor split as a function of the displacement"""
    return K * (0.5*(ufl.tr(eps(u))-abs(ufl.tr(eps(u))))) * ufl.Identity(ndim)

def sigma(u,d):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(d) * sigma_p(u) + sigma_n(u)

def pen(d,d0):
    """ Penalty term for fracture irreversibility"""
    return pen_fact * Gc / l0 * conditional(lt(d,d0),(d0-d)*(d0-d),0.0)

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
V_u       = FunctionSpace(mesh,element_u)
V_d       = FunctionSpace(mesh,element_d)

# Test function
u_        = TestFunction(V_u)
d_        = TestFunction(V_d)

# Trial function
du_       = TrialFunction(V_u)
dd_       = TrialFunction(V_d)

# Solution
u         = Function(V_u)
d         = Function(V_d)

# Old iteration phase-field
dOld      = Function(V_d)

# Old step phase-field
d0        = Function(V_d)



# ---------------------------------------------------------------#
# Setting up boundary conditions
# ---------------------------------------------------------------#

u_pres  = Expression("t", t = 0.0, degree=1)
count   = 0
edge    = [None] * (len(fix_dofs) + len(pres_dofs))
bcs_u   = [None] * (len(fix_dofs) + len(pres_dofs))

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

# Prescribed boundaries
for idx in pres_edge:
    for dof in pres_dofs:
        edge[count]  = CompiledSubDomain(idx, tol=1e-4)
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
elastic_energy    = 0.5 * ufl.inner(sigma(u,d), eps(u)) * dx 

# Phase-field energy (dissipated energy)
dissipated_energy = Gc / float(c_w) * (w(d) / l0 + 
                    l0 * ufl.dot(ufl.grad(d), ufl.grad(d))) * dx

# Penalty energy for fracture irreversibility
penalty_term      = 0.5 * pen(d,d0) * dx

# Total energy
total_energy      = elastic_energy + dissipated_energy + penalty_term

# Extract the displacement sub-problem (Residual and Jacobian)
F_u               = ufl.derivative(total_energy,u,u_)
J_u               = ufl.derivative(F_u, u, du_)
problem_u         = NonlinearVariationalProblem(F_u, u,bcs_u,J_u)

# Extract the phase-field sub-problem (Residual and Jacobian)
F_d               = ufl.derivative(total_energy,d,d_)
J_d               = ufl.derivative(F_d, d, dd_)
problem_d         = NonlinearVariationalProblem(F_d,d,[],J_d)



# ---------------------------------------------------------------#
# Setting up solvers
# ---------------------------------------------------------------#

# Displacement and phase-field solvers
solver_u          = NonlinearVariationalSolver(problem_u)
solver_d          = NonlinearVariationalSolver(problem_d)

# Solver configuration
if snes:

    # PETSc SNES
    solver_u.parameters['nonlinear_solver']                       = 'snes'
    solver_u.parameters["snes_solver"]["linear_solver"]           = "cg"
    solver_u.parameters["snes_solver"]["preconditioner"]          = "petsc_amg"
    solver_u.parameters["snes_solver"]["maximum_iterations"]      = 1
    solver_u.parameters["snes_solver"]["report"]                  = False
    solver_u.parameters["snes_solver"]["error_on_nonconvergence"] = False

    solver_d.parameters['nonlinear_solver']                       = 'snes'
    solver_d.parameters["snes_solver"]["linear_solver"]           = "cg"
    solver_d.parameters["snes_solver"]["preconditioner"]          = "petsc_amg"
    solver_d.parameters["snes_solver"]["maximum_iterations"]      = 1
    solver_d.parameters["snes_solver"]["report"]                  = False
    solver_d.parameters["snes_solver"]["error_on_nonconvergence"] = False

else:

    # Newton-Raphson
    solver_u.parameters['newton_solver']['convergence_criterion']   = 'incremental'
    solver_u.parameters["newton_solver"]["linear_solver"]           = "gmres"
    solver_u.parameters["newton_solver"]["preconditioner"]          = "petsc_amg"
    solver_u.parameters["newton_solver"]["maximum_iterations"]      = 1
    solver_u.parameters["newton_solver"]["error_on_nonconvergence"] = False

    solver_d.parameters['newton_solver']['convergence_criterion']   = 'incremental'
    solver_d.parameters["newton_solver"]["linear_solver"]           = "gmres"
    solver_d.parameters["newton_solver"]["preconditioner"]          = "petsc_amg"
    solver_d.parameters["newton_solver"]["maximum_iterations"]      = 1
    solver_d.parameters["newton_solver"]["error_on_nonconvergence"] = False



# ---------------------------------------------------------------#
# Setting up post-processing options
# ---------------------------------------------------------------#

out_dir  = "../output/"+problem
os.makedirs(out_dir, exist_ok=True)

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

as_backend_type(u.vector()).update_ghost_values()
as_backend_type(d.vector()).update_ghost_values()

tic = time.perf_counter()

# Staggered scheme
step     = 0
t        = 0.0
max_load = 0.0

while True:

    step  += 1

    if t <= t_switch:
        t     += dt
    else:
        t     += dt2

    u_pres.t = t
    iter   = 0

    if rank == 0:
        print("==============================")
        print("  Step:",step," Time:", "%.7g" % (t) )
        print("------------------------------")
        print("  Iteration     Error")
        sys.stdout.flush()

    while True:

        iter += 1
        dOld.assign(d)
        solver_u.solve()
        solver_d.solve()

        local_err = errornorm(d,dOld,norm_type='l2',mesh = mesh)
        err       = MPI.COMM_WORLD.allreduce(local_err, op=MPI.SUM)/size

        if rank == 0:
            print("\t",iter,"\t", "%.4g" % (err) )
            sys.stdout.flush()
        
        if err < nl_tol:

            d0.assign(d)

            trac         = dot(sigma(u,d),n)
            local_trac   = assemble(trac[lodi_ldof]*ds(lodi_ldof))
            glob_trac    = MPI.COMM_WORLD.allreduce(local_trac, op=MPI.SUM)/size

            if rank == 0:
                print("------------------------------")
                print("Converged in", iter, " iterations!")
                print("Load:", "%.7g" % (glob_trac), "\n")
                sys.stdout.flush()

                lodi_out.write(str(t)+"\t")
                lodi_out.write(str(glob_trac)+"\n")

            xdmf_out.rename("u","u")
            xdmf_out.rename("d","d")

            xdmf_out.write(u,step)
            xdmf_out.write(d,step)

            break

    max_load    = max(abs(glob_trac),abs(max_load))
    ratio_load  = abs(glob_trac/max_load)

    if ratio_load < t_stop_ratio:
        break

toc = time.perf_counter()

if rank == 0:
    print ('Simulation completed in', "%.4g" % (toc-tic), " seconds" ) 


# Show the final phase-field only if run on single process
if size == 0:
    p=plot(d)
    p.set_cmap("viridis")
    p.set_clim(0.0,1.0)
    plt.colorbar(p);
    plt.show()

# ---------------------------------------------------------------#
# End time-stepping
# ---------------------------------------------------------------#