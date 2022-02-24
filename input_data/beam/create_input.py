# ===================================================================================== #
#                Creates the input files for an elastic beam bending                    #
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
L          = 25.
H          = 1.
Nx         = 125
Ny         = 5

# Material parameters
E          = 1e5
nu         = 0.3
hmax       = 0.1

# Constraints
fix_edge   = ["near(x[0],0.0) && on_boundary"]
fix_dofs   = ["dx","dy"]
pres_edge  = ["near(x[0],25.0) && on_boundary"]
pres_dofs  = ["dy"]

# Stepping
dt           = -1e-2
t_max_steps  = 10

# Nonlinear solver
nl_type    = "snes"
nl_tol     = 1e-3
max_iter   = 1200

# Linear solver
lin_solver = "PETSc"

# Post-processing
lodi      = True
lodi_edge = "near(x[0],0.0) && on_boundary"
lodi_ldof = "dy"



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

    "material": {
        "E": E,
        "nu": nu
    },

    "constraints": {
        "fix_edge": fix_edge,
        "fix_dofs": fix_dofs,
        "pres_edge": pres_edge,
        "pres_dofs": pres_dofs,
    },

    "stepping": {
        "dt": dt,
        "t_max_steps":t_max_steps
    },

    "nl_solver": {
        "nl_type": nl_type,
        "nl_tol": nl_tol,
        "max_iter": max_iter
    },

    "lin_solver": {
        "type": lin_solver
    },

    "post_process": {
        "lodi": lodi,
        "lodi_edge": lodi_edge,
        "lodi_ldof": lodi_ldof
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