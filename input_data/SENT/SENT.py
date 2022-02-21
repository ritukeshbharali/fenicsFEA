# ===================================================================================== #
#      Creates the input files for Single Edge Notched specimen under Tension (SENT)
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

# Material parameters
E          = 210e3
nu         = 0.3
Gc         = 2.7
l0         = 0.015
hmax       = l0/2
pen_factor = 7500

# Constraints
fix_edge   = ["near(x[1],0.0) && on_boundary",
              "near(x[1],0.0) && on_boundary"]
fix_dofs   = ["dx",
              "dy"]
pres_edge  = ["near(x[1],1.0) && on_boundary"]
pres_dofs  = ["dy"]

# Stepping
dt           = 5e-4
dt2          = 5e-6
t_switch     = 0.0055
t_fact       = 100
t_stop_ratio = 0.05
t_max_steps  = 1200

# Nonlinear solver
snes       = True
nl_tol     = 1e-3
max_iter   = 1200

# Linear solver
lin_solver = "PETSc"

# Post-processing
lodi      = True
lodi_edge = "near(x[1],1.0) && on_boundary"
lodi_ldof = "dy"


# ---------------------------------------------------------------#
# Construct geometry
# ---------------------------------------------------------------#

S0 = Rectangle(Point(0, 0), Point(1, 1))
S1 = Rectangle(Point(0,0.5),Point(0.5,0.505))
S2 = Rectangle(Point(0.5,0.45),Point(1.0,0.55))

geometry  = (S0-S1+S2)



# ---------------------------------------------------------------#
# Generate mesh and carry out refinement as required
# ---------------------------------------------------------------#

mesh          = generate_mesh(geometry,25)
i            = 0
do_refine    = True

while do_refine:

    i            += 1
    subDomain     = CompiledSubDomain("x[0]>0.5 - DOLFIN_EPS && x[0]<1.0 + DOLFIN_EPS && x[1]<0.525 + DOLFIN_EPS && x[1]>0.475 - DOLFIN_EPS")
    subDomainMark = MeshFunction("size_t", mesh, mesh.topology().dim(),0)
    cellRefMark   = MeshFunction("bool", mesh, mesh.topology().dim(), False)
    
    subDomain.mark(subDomainMark,1)
    
    marked_cells = SubsetIterator(subDomainMark,1)
    d            = Circumradius(mesh)
    d_vector     = project(d,FunctionSpace(mesh,"DG",0)).vector()[:] * 2

    for c in marked_cells:
        if d_vector[c.index()] > hmax:
            cellRefMark[c] = True
        else:
            cellRefMark[c] = False

    numCells0 = mesh.num_cells()
    mesh      = refine(mesh, cellRefMark)
    numCells  = mesh.num_cells()

    if numCells0 == numCells:
        do_refine = False

print("Mesh is refined "+str(i)+" times! \n")



# ---------------------------------------------------------------#
# Save mesh
# ---------------------------------------------------------------#

nargs = len(sys.argv)

if nargs != 2:
    File("SENT_mesh.xml.gz") << mesh
else:
    File("SENT_mesh."+sys.argv[1]) << mesh



# ---------------------------------------------------------------#
# Save input data to json file
# ---------------------------------------------------------------#

SENT = {

    "material": {
        "E": E,
        "nu": nu,
        "Gc": Gc,
        "l0": l0,
        "penalty_factor":pen_factor
    },

    "constraints": {
        "fix_edge": fix_edge,
        "fix_dofs": fix_dofs,
        "pres_edge": pres_edge,
        "pres_dofs": pres_dofs,
    },

    "stepping": {
        "dt": dt,
        "dt2":dt2,
        "t_switch": t_switch,
        "t_fact": t_fact,
        "t_stop_ratio":t_stop_ratio,
        "t_max_steps":t_max_steps
    },

    "nl_solver": {
        "snes": snes,
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
json.dump(SENT,open("./SENT_input.json","w"), indent=4, sort_keys=False)

print("SENT input files succesfully prepared! \n")


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