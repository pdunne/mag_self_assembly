"""Old optimisation and global optimisation modules

NOTE: See self_consistent.py for documentation. This module is left here for
reference and convenience.
"""
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, prange
import scipy.optimize as optimize

from scipy import constants

u0 = constants.mu_0
ec = constants.e
hb = constants.hbar
kB = constants.Boltzmann
muB = constants.physical_constants["Bohr magneton"][0]
Na = constants.Avogadro

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
    v3d = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    out = np.dot(points, v3d.T)
    return out


# Generate magnet dipoles positions
def generate_pos(r, zz, step, NN):
    rdip = np.array(
        [
            [0, r, 0],
            [-r * np.sin(np.deg2rad(120)), r * np.cos(np.deg2rad(120)), 0],
            [r * np.sin(np.deg2rad(120)), r * np.cos(np.deg2rad(120)), 0],
        ]
    )

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
        if k != KK:
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
        dloc[k] = np.sqrt(rloc[k, 0] ** 2 + rloc[k, 1] ** 2 + rloc[k, 2] ** 2)
    return dloc


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
    """Converts moment from spherical to cartesian co-ordinates"""
    m = np.asarray(
        sph2cart(m_spher[:, 0], m_spher[:, 1], m_spher[:, 2]), dtype=np.float64
    ).T
    # m[np.abs(m) < 1e-16] = 0
    m *= muB * m_atom
    return m


# @jit(nopython=True, fastmath=True, cache=True)
def gen_m(p, az, el, m_atom):
    """Generates magnetic moment array based on azimuth and polar angles

    gen_m(p,az,el,m_atom)

    Args:
        p: atom positions [N,3] array
        az: azimuthal angle for all moments
        el: polar angle for all moments
        m_atom: magnetic moment of individual atoms (Am^2)
    """
    m_spher = np.zeros_like(p)
    m_spher[:, 0] = 1
    m_spher[:, 1] = np.deg2rad(az)
    m_spher[:, 2] = np.deg2rad(el)
    m = m_conv(m_spher, m_atom)
    return m


# @jit(nopython=True,fastmath=True,cache=True)


def init_magn(p, m_atom, az, el):
    """Initialise cartesian magnetic moment array

    `init_magn(p,m_atom,az,el)`

    Args:
        p: atom positions [N,3] array
        m_atom: magnetic moment of individual atoms (Am^2)
        az: azimuthal angle for all moments
        el: polar angle for all moments
    """
    m_spher = np.zeros_like(p)
    m_spher[:, 0] = 1
    m_spher[:, 1] = np.deg2rad(az)
    m_spher[:, 2] = np.deg2rad(el)
    m = m_conv(m_spher, m_atom)
    return m


@jit(nopython=True, fastmath=True)
def curie_moment(m_atom, B, T):
    """Calculates Curie-Law moment

    Args:
        m_atom ([type]): [description]
        B ([type]): Magnetic field in T
        T ([type]): Temperature in K

    Returns:
        [float/ndarray]:
    """
    out = muB * m_atom ** 2 * B / 3 / kB / T
    return out


# Main
# @jit(parallel=True)


def calcU(p, m, preFac, Bext):
    """Calculates total magnetic energy

    Args:
        p ([type]): [description]
        m ([type]): [description]
        preFac ([type]): [description]
        Bext ([type]): [description]

    Returns:
        [type]: [description]
    """
    """ Calculates total magnetic energy
        Return Utot[0], and Blocal[dipoles]
    """
    U = np.zeros(len(m))
    Bt = np.zeros_like(m)

    for KK in prange(len(m)):
        #     Calculate dipolar fields at one site due to the rest
        rloc = delete_row(p - p[KK], KK)
        dloc = norm_dist(rloc)
        mloc = delete_row(m, KK)
        m_dotR = 3 * dotprod(mloc, rloc) / dloc ** 5
        Bloc = preFac * (mult_array(m_dotR, rloc) - mult_array(1 / dloc ** 3, mloc))
        # Return total magnetic energy and local magnetic field due to all
        # other atoms
        U[KK], Bt[KK] = mdotB(m[KK], Bloc, Bext)
    out = -np.sum(U)
    return out, Bt


def macro_calc(param):
    """Calculates Energies and minimise using a macrospin approximation

    Args:
        param (dictionary): contains dictionary of parameters

    Returns:
    tuple: U, Bt, p
    """
    p = generate_pos(param["r"], param["d"], param["step"], param["NN"])
    m = init_magn(p, param["m_curie"], param["az"], param["el"])
    U = 0.0
    Bt = np.zeros_like(m)
    if param["min"] == 1:
        method = "L-BFGS-B"
        initial_guess = [param["az"], param["el"]]
        result = optimize.minimize(
            macrospin_min,
            initial_guess,
            args=(param["m_curie"], param["Bext"], p),
            method=method,
            options={"disp": False},
        )
        if result.success:
            # print(result.message)
            param["result"] = result
            param["az"] = result.x[0]
            param["el"] = result.x[1]
        else:
            print(result.message)
    U, Bt = calcU(p, m, preFac, param["Bext"])
    return (U, Bt, p)


def macrospin_min(angles, m_atom, Bext, p):
    """Returns total energy for minimiser function

    The routine minmises over one az and el pair for all moments
    i.e. macrospin approach.

    Args:
        angles ([type]): [description]
        m_atom ([type]): [description]
        Bext ([type]): [description]
        p ([type]): [description]

    Returns:
        [float]: Total energy used for minimiser function
    """
    az = angles[0]
    el = angles[1]
    m = init_magn(p, m_atom, az, el)
    U = 0.0
    Bt = np.zeros_like(m)
    U, Bt = calcU(p, m, preFac, Bext)
    return U / muB


def global_calc(param):
    p = generate_pos(param["r"], param["d"], param["step"], param["NN"])
    N1 = len(p)
    az = np.ones(N1) * param["az"]
    el = np.ones(N1) * param["el"]
    # az = np.random.uniform(0, 180, [N1])
    # el = np.random.uniform(0, 180, [N1])
    initial_guess = np.append(az, el)
    m = gen_m(p, az, el, param["m_curie"])

    U = 0.0
    Bt = np.zeros_like(m)
    if param["min"] == 1:
        method = "L-BFGS-B"
        minimizer_kwargs = {
            "method": method,
            "args": (param["m_curie"], param["Bext"], p),
        }
        result = optimize.basinhopping(
            global_min, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=200
        )
        # if result.success:
        param["result"] = result
        param["az"] = result.x[0:N1] % 360
        param["el"] = result.x[N1:] % 360
        m = init_magn(p, param["m_curie"], param["az"], param["el"])
        # else:
        #   print(result.message)
        #   raise ValueError(result.message)
    U, Bt = calcU(p, m, preFac, param["Bext"])
    return (U, Bt, p)


def global_min(angles, m_atom, Bext, p):
    """Returns total energy for minimiser function
    minmises each individual az and el pair for all
    moments
    """
    # m_atom = 1
    N1 = len(p)
    az = angles[0:N1]
    el = angles[N1:]
    m = gen_m(p, az, el, m_atom)
    U = 0.0
    Bt = np.zeros_like(m)
    U, Bt = calcU(p, m, preFac, Bext)
    return U / muB


# Quiver Plot
def plot_res(p, vec, filenm, SAV=0):
    """3D plot of vector field at each atomic point

    Args:
        p (numpy array  (N_atoms, 3): atom co-ordinates
        vec (numpy array, (N atoms, 3)): vectors field at each atomic site
        filenm (string): file name
        SAV (boolean, optional): Save to file?. Defaults to 0.
    """
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection="3d")
    x = p[:, 0] * 1e9
    y = p[:, 1] * 1e9
    z = p[:, 2] * 1e9

    ax3D.scatter(x, y, z, s=30, c="k")
    ax3D.set_xlabel("$x$ (nm)")
    ax3D.set_ylabel("$y$ (nm)")
    ax3D.set_zlabel("$z$ (nm)")
    ax3D.view_init(elev=28, azim=57)
    ax3D.auto_scale_xyz

    u = vec[:, 0]
    v = vec[:, 1]
    w = vec[:, 2]
    M = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    u /= M
    v /= M
    w /= M

    c = plt.cm.bwr(w)
    qq = ax3D.quiver(x, y, z, u, v, w, colors=c, cmap=plt.cm.bwr, clim=[-1, 1])
    plt.colorbar(qq)
    plt.show()

    if SAV == 1:
        plt.savefig(filenm + ".pdf", dpi=300, bbox_inches="tight")
        print("Saving to " + filenm + ".pdf")
