#!/usr/bin/python

from dolfin import *
import mms_exp as mms
import numpy as np

set_log_level(PROGRESS)

#----------------------------------------------------------#

ele_disc = "DG"
ele_order = 1

#----------------------------------------------------------#

def run(nx):
    mesh = UnitSquareMesh(nx, nx)
    n = FacetNormal(mesh)
    h = CellSize(mesh)

    VFS = VectorFunctionSpace(mesh, "DG", 1)
    FS = FunctionSpace(mesh, "DG", 1)
    
    ic = project(Expression((mms.u(),mms.v()), degree=5), VFS)
    
    u = project(ic, VFS)

    f = project(Expression((mms.f_u(),mms.f_v()), degree=5), VFS)
    
    nu = project(Expression(mms.nu(), degree=5), FS)

    uFile = File("u.pvd")
    nuFile = File("nu.pvd")
    uFile << u
    nuFile << nu 

    i, j, k, l = indices(4)
    p, q, r, s = indices(4)

    v = TestFunction(VFS)

    # PARTIAL STRESS FORM
    # A = as_tensor( 
    #     [ [ [ [ 2*nu, 0.0 ], [0.0 , nu  ] ],
    #         [ [ 0.0 , nu  ], [0.0 , 0.0 ] ] ],
    #       [ [ [ 0.0 , 0.0 ], [nu  , 0.0 ] ],
    #         [ [ nu  , 0.0 ], [0.0 , 2*nu] ] ] ]
    #     )
    # TENSOR FORM
    A = as_tensor( 
        [ [ [ [ 1.0, 0.0 ], [0.0, 0.0] ],
            [ [ 0.0, 1.0 ], [0.0, 0.0] ] ],
          [ [ [ 0.0, 0.0 ], [1.0, 0.0] ],
            [ [ 0.0, 0.0 ], [0.0, 1.0] ] ] ]
        )

    un = jump(u)[i]*n("+")[j]
    un_ij = as_tensor(un, (i,j))
    vn = jump(v)[i]*n("+")[j]
    vn_ij = as_tensor(vn, (i,j))

    un_b = (u-ic)[i]*n[j]
    un_bij = as_tensor(un_b, (i,j))
    vn_b = v[i]*n[j]
    vn_bij = as_tensor(vn_b, (i,j))
    
    F_z = 0
    F_u = 0
    for i_ in range(2):
    
        F = (-
            ( 
                grad(u)[i_,j]*grad(v)[i_,j]*dx 
                )+ 
            (
                avg(grad(v)[i_,j])*un_ij[i_,j]*dS + 
                avg(grad(u)[i_,j])*vn_ij[i_,j]*dS + 
                grad(v)[i_,j]*un_bij[i_,j]*ds + 
                grad(u)[i_,j]*vn_bij[i_,j]*ds
                )+
            (
                un_ij[i_,j]*vn_ij[i_,j]*dS +
                un_bij[i_,j]*vn_bij[i_,j]*ds
                )-
            (
                v[i_]*u[i_]*dx 
                )-
            (
                f[i_]*v[i_]*dx
                )
            )
         
    print u.vector().array()   
    solve(F == 0, u)

    uFile << u
    
    Eu = errornorm(u, ic, norm_type="L2", degree_rise=3)
    print Eu

    return Eu
    
if __name__=='__main__':
    
    h = [] # element sizes
    E = [] # errors

    for i, nx in enumerate([4, 8, 16, 24]):
        h.append(1.0/nx)
        print 'h is: ', h[-1]
        E.append(run(nx))

    for i in range(1, len(E)):
        ru = np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E ru=%.2f Eu=%.2e"
                    % (h[i], ru, E[i]) )
