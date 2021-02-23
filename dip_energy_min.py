import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from mpl_toolkits.mplot3d import Axes3D
# from numpy import linalg as LA
from numba import jit, njit, prange

from scipy import constants
u0 = constants.mu_0
ec = constants.e
hb = constants.hbar
kB = constants.Boltzmann
muB = constants.physical_constants['Bohr magneton'][0]

preFac = u0 / 4 / np.pi

# Maths Utilities
@jit(nopython=True, fastmath=True, cache=True)
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


@jit(nopython=True, fastmath=True, cache=True)
def sph2cart(r, az, el):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


# Rotate points about the z-axis
def rot_points(points, theta):
    theta = np.deg2rad(theta)
    v3d = np.zeros([3, 3])
    v3d = np.array([[np.cos(theta), - np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    out = np.dot(points, v3d.T)
    return(out)

# Generate magnet dipoles positions
# @jit()


def generate_pos(r, zz, step, NN):
    rdip = np.array([[0, r, 0],
                     [-r * np.sin(np.deg2rad(120)), r *
                      np.cos(np.deg2rad(120)), 0],
                     [r * np.sin(np.deg2rad(120)),
                     r * np.cos(np.deg2rad(120)), 0]])

    p3 = rdip
    for k in np.arange(1, NN):
        p2 = rot_points(rdip, k * step)
        p2[:, 2] = k * zz
        p3 = np.vstack((p3, p2))
    return p3


@jit(nopython=True, fastmath=True, cache=True)
def delete_row(m, KK):
    N1 = len(m)
    mloc = np.zeros((N1 - 1, 3))
    ii = np.int(0)
    for k in range(0, N1):
        if (k != KK):
            mloc[ii, 0] = m[k, 0]
            mloc[ii, 1] = m[k, 1]
            mloc[ii, 2] = m[k, 2]
            ii += 1
    return mloc


@jit(nopython=True, fastmath=True, cache=True)
def norm_dist(rloc):
    N1 = len(rloc)
    dloc = np.zeros(N1)
    for k in prange(len(rloc)):
        dloc[k] = np.sqrt(rloc[k, 0]**2 + rloc[k, 1]**2 + rloc[k, 2]**2)
    return(dloc)


@jit(nopython=True, fastmath=True, cache=True)
def dotprod(a, b):
    N1 = len(a)
    val = np.zeros(N1)
    for j in range(N1):
        for k in range(3):
            val[j] += a[j, k] * b[j, k]
    return val


@jit(nopython=True, fastmath=True)
def mult_array(a, b):
    N1 = len(a)
    val = np.zeros((N1, 3))
    for j in range(N1):
        for k in range(3):
            val[j, k] += a[j] * b[j, k]
    return val


@jit(nopython=True, fastmath=True, cache=True)
def mdotB(a, b, Bext):
    N1 = len(b)
    val = np.zeros(1)
    val2 = np.zeros(3)
    for k in range(3):
        for j in range(N1):
            val2[k] += b[j, k]
        val2[k] += Bext[k]
        val += a[k] * val2[k]
    return val, val2


# Magnetic Moment Routines
def m_conv(m_spher, m_atom):
    m = np.asarray(
        sph2cart(m_spher[:, 0], m_spher[:, 1], m_spher[:, 2]),
        dtype=np.float64).T
    # m[np.abs(m) < 1e-16] = 0
    m *= muB * m_atom
    return(m)

# @jit(nopython=True,fastmath=True,cache=True)


def gen_m(p, az, el, m_atom):
    m_spher = np.zeros_like(p)
    m_spher[:, 0] = 1
    m_spher[:, 1] = np.deg2rad(az)
    m_spher[:, 2] = np.deg2rad(el)
    m = m_conv(m_spher, m_atom)
    return m

# @jit(nopython=True,fastmath=True,cache=True)


def init_magn(p, m_atom, az, el):
    m_spher = np.zeros_like(p)
    m_spher[:, 0] = 1
    m_spher[:, 1] = np.deg2rad(az)
    m_spher[:, 2] = np.deg2rad(el)
    m = m_conv(m_spher, m_atom)
    return(m)


@jit(nopython=True, fastmath=True)
def curie_moment(m_atom, B, T):
    out = muB * m_atom**2 * B / 3 / kB / T
    return out

# Main
# @jit(parallel=True)


def calcU(p, m, preFac, Bext):
    U = np.zeros(len(m))
    Bt = np.zeros_like(m)

    for KK in prange(len(m)):
        #     Calculate dipolar fields at one site due to the rest
        rloc = delete_row(p - p[KK], KK)
        dloc = norm_dist(rloc)
        mloc = delete_row(m, KK)
        m_dotR = 3 * dotprod(mloc, rloc) / dloc**5
        Bloc = preFac * (mult_array(m_dotR, rloc) -
                         mult_array(1 / dloc**3, mloc))
        # Return total magnetic energy and local magnetic field due
        # to all other atoms
        U[KK], Bt[KK] = mdotB(m[KK], Bloc, Bext)
    out = - np.sum(U)
    return out, Bt


# Quiver Plot
def plot_res(p, vec):
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    x = p[:, 0] * 1e9
    y = p[:, 1] * 1e9
    z = p[:, 2] * 1e9

    # ax3D.scatter(x,y,z,s=30,c=z,cmap='viridis')
    ax3D.scatter(x, y, z, s=30, c='k')
    ax3D.set_xlabel('$x$ (nm)')
    ax3D.set_ylabel('$y$ (nm)')
    ax3D.set_zlabel('$z$ (nm)')

    u = vec[:, 0]
    v = vec[:, 1]
    w = vec[:, 2]
    M = np.sqrt(u**2 + v**2 + w**2)
    u /= M
    v /= M
    w /= M

    c = plt.cm.bwr(w)
    qq = ax3D.quiver(x, y, z, u, v, w,
                     colors=c, cmap=plt.cm.bwr, clim=[-1, 1])
    plt.colorbar(qq)
    plt.show()


##
# Define Monomer parameters - C3 symmetry
r = 2.25e-9  # nm radius
zz = 0.35e-9  # nm stacking distance
# r = 2e-9
# zz = 1e-9
step = 10  # chirality -  degrees roatation
NN = 200  # number of monomers


p = generate_pos(r, zz, step, NN)


m_atom = 10.6
Bext = np.array([0, 0, 1])

m_curie = curie_moment(m_atom, Bext[2], 250)


az = 0.0
el = 0.0

m = init_magn(p, m_curie, az, el)


U = np.zeros(len(m))
Bt = np.zeros_like(m)

U, Bt = calcU(p, m, preFac, Bext)
print('{0:0d} monomers, L = {1:0.1f} nm, U/kT = {2:0.3e}'.format(NN,
                                                                 NN * zz * 1e9,
                                                                 U / kB / 300))

plt.close('all')
# plot_res(p,m/muB)

# Macroscpin Minimisation
r = 2.25e-9  # nm radius
zz = 0.35e-9  # nm stacking distance
# r = 2e-9
# zz = 1e-9
step = 20  # chirality -  degrees roatation
NN = 20  # number of monomers
p = generate_pos(r, zz, step, NN)

m_atom = 1
az = 45.0
el = 45.0

m = init_magn(p, m_atom, az, el)
U = 0.0
Bt = np.zeros_like(m)
Bext = np.array([0, 1, 0])


def f(param, Bext, p):
    m_atom = 1
    # Bext = np.array([0,0,0])
    az = param[0]
    el = param[1]
    m = init_magn(p, m_atom, az, el)
    U = 0.0
    Bt = np.zeros_like(m)
    U, Bt = calcU(p, m, preFac, Bext)
    return U / muB


# method = "Powell"
# method="BFGS"
# method="Newton-CG"
method = "L-BFGS-B"


initial_guess = [az, el]
result = optimize.minimize(f, initial_guess, args=(
    Bext, p), method=method, options={'disp': True})
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)


# minimizer_kwargs = {"method": "BFGS", "args":(Bext,p)}
#
# result_hop = optimize.basinhopping(f, initial_guess,
#     minimizer_kwargs=minimizer_kwargs,niter=200)
# fitted_params = result_hop.x
# print(result_hop.x)


az = result.x[0]
el = result.x[1]

m = init_magn(p, m_atom, az, el)
U, Bt = calcU(p, m, preFac, Bext)
print(U)
print(U / kB / 300)
plt.close('all')
plot_res(p, m / muB)
print('{0:0d} monomers, L = {1:0.1f} nm, U/kT = {2:0.3e}'.format(NN,
                                                                 NN * zz * 1e9,
                                                                 U / kB / 300))


##

r = 2.25e-9  # nm radius
zz = 0.35e-9  # nm stacking distance
step = 10  # chirality -  degrees roatation
NN = 10  # number of monomers

m_atom = 10

p = generate_pos(r, zz, step, NN)
N1 = len(p)

Bext = np.array([0, 0, 0])

az = np.ones(N1) * 25
el = np.ones(N1) * 90

initial_guess = np.append(az, el)

m = gen_m(p, az, el, m_atom)

U = 0.0
Bt = np.zeros_like(m)
U, Bt = calcU(p, m, preFac, Bext)

# Full Minimisation


def f2(param, Bext, p):
    m_atom = 1
    N1 = len(p)
    az = param[0:N1]
    el = param[N1:]
    m = gen_m(p, az, el, m_atom)
    U = 0.0
    Bt = np.zeros_like(m)
    U, Bt = calcU(p, m, preFac, Bext)
    return U / muB


Bext = np.array([0, 0, 0])


method = "BFGS"
# method = "Powell"
method = "L-BFGS-B"

minimizer_kwargs = {"method": method, "args": (Bext, p)}

result = optimize.basinhopping(f2, initial_guess,
                               minimizer_kwargs=minimizer_kwargs, niter=200)

print(result)

##

# m_atom = 10
m_max = gen_m(p, np.ones(N1) * 90, np.ones(N1) * 90, m_atom)
m_max = np.sum(m_max[:, 2])

az = result.x[0:N1] % 360
el = result.x[N1:] % 360

m = init_magn(p, m_atom, az, el)
m_norm = (np.sum(m[:, 2]) / m_max)
print(m_norm)
print('Length = ')

U, Bt = calcU(p, m, preFac, Bext)
print(U)
print(U / kB / 300)
plt.close('all')
plot_res(p, m / muB)
plot_res(p, Bt)


print('{0:0d} monomers, Mz = {1:0.3f}, U = {2:0.3e}/kT'.format(NN, m_norm,
                                                                    U/kB/300))
