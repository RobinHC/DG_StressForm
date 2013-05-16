y = var('y')

pi_ = 3.14159265359

f = 1.0
u = cos(f*y*pi_)
u = sin(f*x*pi_) + cos(f*y*pi_)
v = cos(f*x*pi_)
v = integral(-diff(u,x),y)

# nu = 2.0+cos(f*y*x*pi_)
nu = 1.0

tau_xx = 2*nu*diff(u,x)            
tau_xy = nu*(diff(u,y) + diff(v,x))
tau_yy = 2*nu*diff(v,y)            
tau_yx = nu*(diff(u,y) + diff(v,x))  

f_u = (diff(tau_xx, x) + diff(tau_xy, y)) - u
f_v = (diff(tau_yx, x) + diff(tau_yy, y)) - v

print "def u():"
print "    return '", str(u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def v():"
print "    return '", str(v).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def f_u():"
print "    return '", str(f_u.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def f_v():"
print "    return '", str(f_v.simplify()).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
print "def nu():"
print "    return '", str(nu).replace("000000000000", "").replace("x", "x[0]").replace("y", "x[1]"), "'"
