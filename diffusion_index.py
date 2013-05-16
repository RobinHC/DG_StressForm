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

    u_0, u_1, z_00, z_01, z_10, z_11 = w.split()
    u0File = File("u0.pvd")
    u1File = File("u1.pvd")
    z00File = File("z00.pvd")
    z01File = File("z01.pvd")
    z10File = File("z10.pvd")
    z11File = File("z11.pvd")
    u0File << u_0
    u1File << u_1 
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

    # int0ij = z[:,j]*q[:,j]
    # int0 = as_tensor(int0ij, (i,j))

    # int1 = q[i,j]*grad(u)[i,j]
    # int1i = as_tensor(int1, (i,j))

    un = jump(u)[i]*n("+")[j]
    un_ij = as_tensor(un, (i,j))
    vn = jump(v)[i]*n("+")[j]
    vn_ij = as_tensor(vn, (i,j))

    # int_2 = int2_0ij[i,j]*avg(q)[i,j]
    # int2i = as_vector(int2, (i))
    
    F_z = 0
    F_u = 0
    for i_ in range(2):
        F_z += z[i_,j]*q[i_,j]*dx - \
            q[i_,j]*grad(u)[i_,j]*dx + \
            un_ij[i_,j]*avg(q)[i_,j]*dS
        F_u += - z[i_,j]*grad(v)[i_,j]*dx + \
            vn_ij[i_,j]*avg(z)[i_,j]*dS + \
            (u[i_]-w_ic[i_])*v[i_]*ds - \
            v[i_]*u[i_]*dx - \
            f[i_]*v[i_]*dx
    
    # def F_z(z, u, q):
    #     return (inner(z, q)*dx - 
    #             inner(q, grad(u))*dx +
    #             inner(jump(u)*n("+"), avg(q))*dS
    #             )
    
    # def F_u(z, u, u_b, f_, v):
    #     i,j = indices(2)

    #     if z[2] == 0:
    #         A0 = as_matrix([[2.0,0.0],[0.0,1.0]])
    #         A1 = as_matrix([[1.0,0.0],[0.0,0.0]])
    #     if z[2] == 1:
    #         A0 = as_matrix([[0.0,0.0],[0.0,1.0]])
    #         A1 = as_matrix([[1.0,0.0],[0.0,2.0]])
    
    #     Az0 = A0*z[0]
    #     # Az0 = as_vector(Az0i, i)
    #     Az1 = A1*z[1]
    #     # Az1 = as_vector(Az1i, i)

    #     return (- (inner(Az0,grad(v))*dx#  + 
    #                # inner(Az1,grad(v))*dx
    #                ) + 
    #               (inner(jump(v)*n("+"), avg(Az0))*dS#  +
    #                # inner(jump(v)*n("+"), avg(Az1))*dS
    #                ) +
    #               (u-u_b)*v*ds -
    #               v*u*dx -
    #               f_*v*dx
    #               )
    
    # Fz = F_z(z = z0, u = u0, q = q0) + \
    #     F_z(z = z1, u = u1, q = q1)
    # Fu = F_u(z = [z0,z1,0], u = u0, u_b = Expression(mms.u()), f_ = f[0], v = v0) + \
    #     F_u(z = [z0,z1,1], u = u1, u_b = Expression(mms.v()), f_ = f[3], v = v1) 
    
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
