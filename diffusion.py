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
    VFS = VectorFunctionSpace(mesh, "DG", 1)
    W = MixedFunctionSpace([FS, VFS, FS, VFS])
    w = Function(W)
    
    w_ic = project((Expression(
                (
                    mms.u(),
                    '0.0','0.0',
                    mms.v(),
                    '0.0','0.0'
                    )
                , W.ufl_element())), W)
    
    f = project((Expression(
                (
                    mms.f_u(),
                    '0.0','0.0',
                    mms.f_v(),
                    '0.0','0.0',
                    )
                , W.ufl_element())), W)
    
    test = TestFunction(W)
    trial = TrialFunction(W)
    L = 0; a = 0
    for i in range(len(w_ic)):
        a += inner(test[i], trial[i])*dx
        L += inner(test[i], w_ic[i])*dx
    solve(a == L, w)

    u_0, z_0, u_1, z_1 = w.split()
    u0File = File("u0.pvd")
    z0File = File("z0.pvd")
    u1File = File("u1.pvd")
    z1File = File("z1.pvd")
    u0File << u_0
    z0File << z_0
    u1File << u_1 
    z1File << z_1

    (v0, q0, v1, q1) = TestFunctions(W)
    (u0, z0, u1, z1) = (w[0], as_vector((w[1], w[2])), w[3], as_vector((w[4], w[5])))
    
    def F_z(z, u, q):
        return (inner(z, q)*dx - 
                inner(q, grad(u))*dx +
                inner(jump(u)*n("+"), avg(q))*dS
                )
    
    def F_u(z, u, u_b, f_, v):
        return (- inner(z, grad(v))*dx + 
                  inner(jump(v)*n("+"), avg(z))*dS +
                  (u-u_b)*v*ds -
                  v*u*dx -
                  f_*v*dx
                  )
    
    Fz = F_z(z = z0, u = u0, q = q0) + \
        F_z(z = z1, u = u1, q = q1)
    Fu = F_u(z = z0, u = u0, u_b = Expression(mms.u()), f_ = f[0], v = v0) + \
        F_u(z = z1, u = u1, u_b = Expression(mms.v()), f_ = f[3], v = v1) 
    F = Fz + Fu

    solve(F == 0, w)
    
    u_0, z_0, u_1, z_1 = w.split()
    u0File << u_0 
    z0File << z_0
    u1File << u_1 
    z1File << z_1
    
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
