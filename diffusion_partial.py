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

    FS = FunctionSpace(mesh, "DG", 1)
    W = MixedFunctionSpace([FS, FS, FS, FS, FS, FS])
    w = Function(W)
    
    w_ic = project((Expression(
                (
                    mms.u(),
                    mms.v(),
                    '0.0',
                    '0.0',
                    '0.0',
                    '0.0'
                    )
                , W.ufl_element())), W)
    
    f = project((Expression(
                (
                    mms.f_u(),
                    mms.f_v(),
                    '0.0',
                    '0.0',
                    '0.0',
                    '0.0',
                    )
                , W.ufl_element())), W)
    
    test = TestFunction(W)
    trial = TrialFunction(W)
    L = 0; a = 0
    for i in range(len(w_ic)):
        a += inner(test[i], trial[i])*dx
        L += inner(test[i], w_ic[i])*dx
    solve(a == L, w)

    nu = project(Expression(mms.nu()), FS)

    u_0, u_1, z_00, z_01, z_10, z_11 = w.split()
    u0File = File("u0.pvd")
    u1File = File("u1.pvd")
    nuFile = File("nu.pvd")
    z00File = File("z00.pvd")
    z01File = File("z01.pvd")
    z10File = File("z10.pvd")
    z11File = File("z11.pvd")
    u0File << u_0
    u1File << u_1 
    nuFile << nu 
    z00File << z_00
    z01File << z_01
    z10File << z_10
    z11File << z_11

    i, j, k, l = indices(4)
    p, q, r, s = indices(4)

    (v, q) = (as_vector((test[0], test[1])), 
              as_tensor(((test[2], test[3]),(test[4], test[5]))))
    (u, z) = (as_vector((w[0], w[1])), 
              as_tensor(((w[2], w[3]),(w[4], w[5]))))

    # PARTIAL STRESS FORM
    A = as_tensor( 
        [ [ [ [ 2*nu, 0.0 ], [0.0 , nu  ] ],
            [ [ 0.0 , nu  ], [0.0 , 0.0 ] ] ],
          [ [ [ 0.0 , 0.0 ], [nu  , 0.0 ] ],
            [ [ nu  , 0.0 ], [0.0 , 2*nu] ] ] ]
        )
    # TENSOR FORM
    # A = as_tensor( 
    #     [ [ [ [ 1.0, 0.0 ], [0.0, 0.0] ],
    #         [ [ 0.0, 1.0 ], [0.0, 0.0] ] ],
    #       [ [ [ 0.0, 0.0 ], [1.0, 0.0] ],
    #         [ [ 0.0, 0.0 ], [0.0, 1.0] ] ] ]
    #     )

    u = as_vector((w_ic[0],w_ic[1]))

    un = jump(u)[i]*n("+")[j]
    un_ij = as_tensor(un, (i,j))
    vn = jump(v)[i]*n("+")[j]
    vn_ij = as_tensor(vn, (i,j))

    u_b = as_vector((w_ic[0],w_ic[1]))
    un_b = (u-u_b)[i]*n[j]
    un_bij = as_tensor(un_b, (i,j))
    vn_b = v[i]*n[j]
    vn_bij = as_tensor(vn_b, (i,j))
    
    F_z = 0
    F_u = 0
    for i_ in range(2):

        F_z += A[i_,j,r,s]*z[i_,j]*q[r,s]*dx - \
            q[i_,j]*grad(u)[i_,j]*dx + \
            un_ij[i_,j]*avg(q[i_,j])*dS + \
            un_bij[i_,j]*q[i_,j]*ds

        F_u = v[i_]*u[i_]*dx - v[i_]*u[i_]*dx

        # F_u += - z[i_,j]*grad(v)[i_,j]*dx + \
        #     vn_ij[i_,j]*avg(z[i_,j])*dS + \
        #     vn_bij[i_,j]*z[i_,j]*ds - \
        #     v[i_]*u[i_]*dx - \
        #     f[i_]*v[i_]*dx
    
    F = F_z + F_u

    solve(F == 0, w)
    
    u_0, u_1, z_00, z_01, z_10, z_11 = w.split()
    u0File << u_0
    u1File << u_1 
    z00File << z_00
    z01File << z_01
    z10File << z_10
    z11File << z_11
    
    FE = FunctionSpace(mesh, "DG", 3)
    S_u = project(Expression(mms.u(), degree=5), FE)
    S_v = project(Expression(mms.v(), degree=5), FE)

    Eu = errornorm(u_0, S_u, norm_type="L2", degree_rise=3)
    Ev = errornorm(u_1, S_v, norm_type="L2", degree_rise=3)

    return Eu, Ev
    
if __name__=='__main__':
    
    h = [] # element sizes
    E = [] # errors

    for i, nx in enumerate([4, 8, 16, 24]):
        h.append(1.0/nx)
        print 'h is: ', h[-1]
        E.append(run(nx))

    for i in range(1, len(E)):
        ru = np.log(E[i][0]/E[i-1][0])/np.log(h[i]/h[i-1])
        rv = np.log(E[i][1]/E[i-1][1])/np.log(h[i]/h[i-1])
        print ( "h=%10.2E ru=%.2f rv=%.2f Eu=%.2e Ev=%.2e"
                    % (h[i], ru, rv, E[i][0], E[i][1]) )
