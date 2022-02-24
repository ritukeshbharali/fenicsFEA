# ===================================================================================== #
#                Creates the input files for Terzaghi consolidation                     #
# ===================================================================================== #


# ---------------------------------------------------------------#
# Import standard libraries
# ---------------------------------------------------------------#

# Fenics
import sys, getopt, json
from dolfin import *
from mshr import *

# Python
import matplotlib.pyplot as plt



# ---------------------------------------------------------------#
# User Data
# ---------------------------------------------------------------#

vis_mesh  = False

# Geometry
L          = 1.
H          = 10.
Nx         = 20
Ny         = 200

# Element type
elem_type  = "Linear"     # Linear, TaylorHood


# Material parameters
E          = 1e8          # Youngs' modulus
nu         = 0.35         # Poisson ratio
K          = 1e-16        # (Intrinsic permeability/Dynamic viscosity)
Kf         = 2e09         # Fluid bulk stiffness
Ks         = 1e10         # Solid grain stiffness
phi        = 0.375        # Porosity
alpha      = 1.0          # Biot's coefficient

# Dirichlet constraints
fix_edge   = ["near(x[0], 0.0) || near(x[0], 1.0)",
              "near(x[1], 0.0)",
              "near(x[1], 10.0)"]
fix_dofs   = ["dx",
              "dy",
              "dp"]

# Neumann constraints
neumann_edge = ["near(x[1],10.0) && on_boundary"]
neumann_load = -1e4
neumann_dof  = "dy"

# Stepping
dt           = 0.005 * 86400
t_max_steps  = 2

# Nonlinear solver
snes       = True
nl_tol     = 1e-3
max_iter   = 1200

# Linear solver
lin_solver = "PETSc"



# ---------------------------------------------------------------#
# Generate mesh and carry out refinement as required
# ---------------------------------------------------------------#

mesh          = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny)



# ---------------------------------------------------------------#
# Save mesh
# ---------------------------------------------------------------#

nargs = len(sys.argv)

if nargs != 2:
    File("mesh.xml.gz") << mesh
else:
    File("mesh."+sys.argv[1]) << mesh



# ---------------------------------------------------------------#
# Save input data to json file
# ---------------------------------------------------------------#

Data = {
    "elem_type": elem_type,    

    "material": {
        "E": E,
        "nu": nu,
        "K": K,
        "Kf": Kf,
        "Ks": Ks,
        "phi": phi,
        "alpha": alpha
    },

    "constraints": {
        "fix_edge": fix_edge,
        "fix_dofs": fix_dofs,
        "neumann_edge": neumann_edge,
        "neumann_load": neumann_load,
        "neumann_dof": neumann_dof
    },

    "stepping": {
        "dt": dt,
        "t_max_steps":t_max_steps
    },

    "nl_solver": {
        "snes": snes,
        "nl_tol": nl_tol,
        "max_iter": max_iter
    },

    "lin_solver": {
        "type": lin_solver
    }
    
}

json.dump(Data,open("./params.json","w"), indent=4, sort_keys=False)

print("Input files succesfully prepared! \n")


# ---------------------------------------------------------------#
# If you wish to visualize the mesh!
# ---------------------------------------------------------------#

if vis_mesh:
    plt.figure(1)
    plt.subplot(211)
    plot(mesh)
    plt.subplot(212)
    plot(cellRefMark)
    plt.show(block=True)