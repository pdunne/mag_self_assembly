"""Self-consitent Calculations

Contains functions needed to perform magnetic self-consistent calculations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, prange
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
    """Converts cartesian coordinates to spherical

    Args:
        x (ndarray): x position
        y (ndarray): y position
        z (ndarray): z position

    Returns:
        tuple: converted coordinates az, el, r
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


@jit(nopython=True, fastmath=True, cache=True)
def sph2cart(r, az, el):
    """Converts spherical coordinates to cartesian

    Args:
        r (ndarray): distance from origin
        az (ndarray): azimuth
        el (ndarray): elevation

    Returns:
        tuple: converted coordinates x, y, z
    """
    rsin_theta = r * np.sin(el)
    x = rsin_theta * np.cos(az)
    y = rsin_theta * np.sin(az)
    z = r * np.cos(el)
    return x, y, z


# Rotate points about the z-axis
def rot_points(points, theta):
    """Rotates points about the z-axis

    Args:
        points (ndarray): positions
        theta (float): roatation angle

    Returns:
        ndarray: rotated points
    """
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


def generate_pos(r, zz, step, NN):
    """Generates atomic positions of caged ions in the chiral DOTA-BTA
    assembles

    Args:
        r (float): radius
        zz (float): inter-layer spacing
        step ([type]): [description]
        NN ([type]): [description]

    Returns:
        [type]: [description]
    """
    theta = np.deg2rad(120)
    rdip = np.array(
        [
            [0, r, 0],
            [-r * np.sin(theta), r * np.cos(theta), 0],
            [r * np.sin(theta), r * np.cos(theta), 0],
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
    """Calculates normalised distance

    Args:
        rloc (ndarray): point coordinates

    Returns:
        ndarray: Normalised distance between each point
    """
    N1 = len(rloc)
    dloc = np.zeros(N1)
    for k in prange(len(rloc)):
        dloc[k] = np.sqrt(
            np.power(rloc[k, 0], 2) + np.power(rloc[k, 1], 2) + np.power(rloc[k, 2], 2)
        )
    return dloc


@jit(nopython=True, fastmath=True, cache=True)
def dotprod(a, b):
    """Dot product

    Args:
        a ([array]): [description]
        b ([array]): [description]

    Returns:
        [type]: [description]
    """
    N1 = len(a)
    val = np.zeros(N1)
    for j in range(N1):
        for k in range(3):
            val[j] += a[j, k] * b[j, k]
    return val


@jit(nopython=True, fastmath=True)
def mult_array(a, b):
    """Multiply array a by b

    Args:
        a ([ndarray]): array of shape (j,3)
        b ([ndarray]): array of shape (j,3)

    Returns:
        [ndarry]: array of shape (j, k)
    """
    N1 = len(a)
    val = np.zeros((N1, 3))
    for j in range(N1):
        for k in range(3):
            val[j, k] += a[j] * b[j, k]
    return val


@jit(nopython=True, fastmath=True, cache=True)
def mdotB(moment, Bloc, Bext):
    """Calculates total magnetic energy and total magnetic fields at one atomic
    position due to all other dipolar fields.

    Args:
        moment (ndarray): magnetic moment, shape (3)
        Bloc (ndarray): Local magnetic field, shape (N, 3)
        Bext (array): External field, [Bx, By, Bz]

    Returns:
        tuple: (energy, Btotal) magnetic energy at atomic position and array of total magnetic field.
    """
    N1 = len(Bloc)
    energy = np.zeros(1)
    Btot = np.zeros(3)
    for k in range(3):
        for j in range(N1):
            Btot[k] += Bloc[j, k]
        Btot[k] += Bext[k]
        energy += moment[k] * Btot[k]
    return energy, Btot


# Magnetic Moment Routines
def m_conv(m_spher, m_atom):
    """Convert moment from spherical to cartesian co-ordinates"""
    m = np.asarray(
        sph2cart(m_spher[:, 0], m_spher[:, 1], m_spher[:, 2]), dtype=np.float64
    ).T
    # m[np.abs(m) < 1e-16] = 0
    m *= muB * m_atom
    return m


def curie_prefac(m_atom, T=298):
    """
    Calculate Curie-Law prefactor for linear moment

    Args:
        m_atom [float]: p_eff
        T [float]: Temperature in K, defaults to 298 K

    Returns:
        [float]:  prefactor
    """
    return muB ** 2 * m_atom ** 2 / 3 / kB / T


def calc_Um(m_eff, B):
    """Calculates magnetic energy

    Args:
        m_eff (float/ndarray): effective mangetic moment in µB
        B (float/ndarray): magnetic field in T
    """
    return -3 * m_eff * B * muB / 2


def calcB(p, m, preFac, Bext):
    """Calculates total magnetic energy

    Args:
        p (ndarray): atomic positions  (N,3) array
        m (ndarray): magnetic moments (N,3) array
        preFac (float): [description]
        Bext (array): External magnetic field in T (1, 3) array

    Returns:
        [float]: total energy
        [ndarray]: local dipole fields
    """
    U = np.zeros(len(m))
    Bt = np.zeros_like(m)

    for KK in prange(len(m)):
        # Calculate dipolar fields at one site due to the rest
        rloc = delete_row(p - p[KK], KK)
        dloc = norm_dist(rloc)
        mloc = delete_row(m, KK)
        m_dotR = 3 * dotprod(mloc, rloc) / dloc ** 5
        Bloc = preFac * (
            mult_array(m_dotR, rloc) - mult_array(1 / np.power(dloc, 3), mloc)
        )
        # Return total magnetic energy and local magnetic field due to all
        # other atoms
        U[KK], Bt[KK] = mdotB(m[KK], Bloc, Bext)
    Ut = -np.sum(U) / 2
    return Ut, Bt


def test_array_nan(array):
    """Replaces any NaNs in an array with 0.0

    Args:
        array (ndarray): any numpy array

    Returns:
        ndarray: processed array
    """
    if np.isnan(array).any():
        array[np.isnan(array)] = 0.0
        return array
    else:
        return array


def g_fac(J, L, S):
    """Landé g-factor

    Args:
        J (float): Total angular momentum
        L (float): Orbital angular momentum
        S (float): Spin angular momentum

    Returns:
        float: Landé g-factor
    """
    return 1 + (J * (J + 1) - L * (L + 1) + S * (S + 1)) / (2 * J * (J + 1))


def brill_x(g, J, B, T=298.0):
    """Calculates Brillouin parameter `x`

    Also known as the Zeeman energy ratio.

    Args:
        g (float): g-factor
        J (float): Total angular momentum
        B (float): Magnetic field in T
        T (float, optional): Temperature in K. Defaults to 298.

    Returns:
       float: Zeeman energy ratio
    """
    """
    Brillouin parameter x -- the ratio of the Zeeman energy for
    a magnetic moment versus kT, used to calculate the
    paramagnetic moment using a Brillouin function.

    Arguments:
    g -- G-factor
    J -- Total angular momentum
    B -- Magnetic field in T
    T -- Temperature in K

    Returns:
    x  -- Zeeman energy ratio
    """
    return g * muB * J * B / kB / T


def brill_func(L, S, m_atom, B, T=298.0):
    """Brillouin function

    Args:
        L (float): Orbital angular momentum
        S (float): Spin angular momentum
        m_atom (float): magnetic moment in Bohr magneton
        B (float): Magnetic field in T
        T (float, optional): Temperature in K. Defaults to 298.0.

    Returns:
        float: Brillouin function value
    """
    J = L + S
    j1 = (2 * J + 1) / 2 / J
    j2 = 1 / 2 / J
    g_calc = g_fac(J, L, S)
    xB = brill_x(g_calc, J, B, T)
    out = muB * m_atom * ((j1 / np.tanh(j1 * xB)) - (j2 / np.tanh(j2 * xB)))
    out = test_array_nan(out)
    return out


def k_sh(M, N):
    """Shape anisotropy energy density, K_sh

    Args:
        M (float): Saturation magnetisation in A/m
        N (float): Demagnetisation factor

    Returns:
        float: Shape anisotropy K_sh energy density in J/m3
    """
    out = (u0 * M ** 2) * (1 - 3 * N) / 4
    return out


def main_calc(
    Bmag=1.0, theta=0.0, phi=0.0, NN=10, T=298, print_res=False, mode="linear"
):
    """Computres the dipole fields and induced magnetic moment for an assembly
    of monomers.

    The maximum number of iterations is hard coded to 50.

    Args:
        Bmag (float, optional): Magnitude of the external field in T. Defaults to 1.0.
        theta (float, optional): elevation angle w.r.t. z-axis in degrees. Defaults to 0.0.
        phi (float, optional): azimutal angle w.r.t. x-axis in degrees. Defaults to 0.0.
        NN (int, optional): Number of monomers. Defaults to 10.
        T (float, optional): Temeprature in K. Defaults to 298.
        print_res (bool, optional): Boolean flat to print simulation results or not. Defaults to False.
        mode (str, optional): magnetic moment model, `linear` or `brillouin`. Defaults to "linear".

    Returns:
        p: matrix of atomic positions (NN, 3) units: m
        m1: induced dipole moments (NN, 3) units: Am2
        Bdip: local dipolar fields (NN, 3) units: T
        Mag: Assembly magnetisation (1,3) units: A/m
        Ut: Total energy of the assembly, units: J
        vol: Assembly volume, units: m3
    """
    r = 2.25e-9  # nm radius
    d = 0.35e-9  # nm stacking distance
    step = 10  # chirality -  degrees rotation

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    Bext = np.asarray(sph2cart(Bmag, phi, theta))
    if print_res:
        print(f"Applied Field:\n [x\t y\t z] T \n {Bext}")
    m_atom = 8.9  # atomic moment in µB

    p = generate_pos(r, d, step, NN)

    m0 = np.zeros_like(p)
    Bt = np.zeros_like(p)

    Bt += Bext
    if mode.lower() == "brillouin":
        m0 = brill_func(0, 7 / 2, m_atom, Bt, T)
    else:
        pf1 = curie_prefac(m_atom, T)
        m0 = pf1 * Bt
    num_iterations = 50
    costs = []

    # Self-consistent loop.
    # At each step m1 is calculated and compared to m0, if the difference is
    # greater than the threshold, m0 is updated and the loop repeated
    for i in range(num_iterations):
        Ut, Bt = calcB(p, m0, preFac, Bext)
        if mode.lower() == "brillouin":
            m1 = brill_func(0, 7 / 2, m_atom, Bt, T)
        else:
            m1 = pf1 * Bt
        dM = (m1 - m0) / muB
        dM = np.linalg.norm(dM)
        m0 = m1
        costs.append(dM)
        if dM < 1e-18:
            break

    Bdip = Bt - Bext
    Bn = np.linalg.norm(Bdip, ord=2, axis=1)
    vol = np.pi * r ** 2 * NN * d
    Mag = np.sum(m1, axis=0, keepdims=True) / vol
    if print_res:
        print(f"### Self-consistent calculation completed in {i} loops ###")
        print(f"dM error is {dM:0.2e}")
        print(
            f"M = {np.linalg.norm(Mag):0.1f} (A/m) \t <Bdip> = "
            f"{np.mean(Bn)*1e3:0.2f} (mT)"
        )
        print(
            f"{Ut*Na:0.2f} J/mol/assem {Ut*Na/NN:0.2f} "
            f"J/mol/mono {NN:0d} monomers {NN*d*1e9:0.2f} nm \n"
        )
    return p, m1, Bdip, Mag, Ut, vol


# Quiver Plot
def plot_res(p, vec, filenm="quiver_plot", save_png=False):
    """3D Quiver plot of magnetic moments

    Args:
        p (array): atomic positions
        vec (array): vector field to plot
        filenm (string): Filename. Defaults to quiver_plot.
        save_png (bool, optional): Save to png file flag. Defaults to False.
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
    dX = np.max(x) - np.min(x)

    dY = np.max(y) - np.min(y)
    dZ = np.max(z) - np.min(z)
    axis_range = np.max([dX, dY, dZ]) / 2

    cX = (np.max(x) + np.min(x)) / 2
    cY = (np.max(y) + np.min(y)) / 2
    cZ = (np.max(z) + np.min(z)) / 2

    ax3D.set_xlim3d(cX - axis_range, cX + axis_range)
    ax3D.set_ylim3d(cY - axis_range, cY + axis_range)
    ax3D.set_zlim3d(cZ - axis_range, cZ + axis_range)

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

    if save_png:
        plt.savefig(filenm + ".png", dpi=150, bbox_inches="tight")
        print("Saving to " + filenm + ".png")


def test_moment_functions():
    """Computes and plots the magnetic moment for Gd ions at T = 4 K and
    T = 298 K, for -7 ≤ B ≤ 7 T using the Curie-Law and Brillouin functions
    """
    S = 7 / 2
    L = 0
    m_atom = 8.9
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    T = 4
    B = np.linspace(-7, 7)
    mB = brill_func(L, S, m_atom, B, T)
    meff = curie_prefac(m_atom, T) * B

    ax1.plot(B, mB / muB, label=r"Brillouin")
    ax1.plot(B, meff / muB, label=r"Linear")
    ax1.set_xlabel("$B$ (T)")
    ax1.set_ylabel(r"$m$ ($\mu_B$)")
    ax1.set_title(f"T = {T}K")
    ax1.legend(loc="best")
    ax1.set_aspect("auto", "box")

    T = 298
    mB = brill_func(L, S, m_atom, B, T)
    meff = curie_prefac(m_atom, T) * B

    ax2.plot(B, mB / muB, label=r"Brillouin")
    ax2.plot(B, meff / muB, label=r"Linear")
    ax2.set_xlabel("$B$ (T)")
    ax2.set_ylabel(r"$m$ ($\mu_B$)")
    ax2.set_title(f"T = {T}K")
    ax2.legend(loc="best")
    ax2.set_aspect("auto", "box")

    plt.show()


def plot_dipolar_field_distn(
    Bn, Bmag, T, filenm, file_type="png", num_bins=10, save_to_file=False
):
    """Plots the distribution of dipolar fields

    Args:
        Bn (ndarray): [description]
        Bmag ([type]): [description]
        T ([type]): [description]
        filenm ([type]): [description]
        file_type (str, optional): image type, accepts matplotlib defaults such as png, pdf, etc. Defaults to "png".
        save_to_file (bool, optional): [description]. Defaults to False.
    """

    NN = Bn.shape[0] / 3
    weights = np.ones_like(Bn) / NN
    _, _ = plt.subplots()

    plt.hist(
        Bn * 1e3,
        bins=num_bins,
        weights=weights,
        facecolor="blue",
        edgecolor="black",
        alpha=0.5,
    )
    plt.xlabel(r"$|B|$ (mT)")
    plt.ylabel("Probablility per monomer")
    plt.title(rf"$B$ = {Bmag} T, $T$ = {T} K")
    plt.show()
    if save_to_file:
        plt.savefig(f"{filenm}.{file_type}", dpi=150, bbox_inches="tight")
        print(f"Saving to {filenm}.{file_type}")


def plot_angular_contour(
    array,
    theta,
    phi,
    cb_label,
    filenm="angular_plot",
    file_type="png",
    save_to_file=False,
):
    """Contour plots

    Args:
        array (ndarray): Scalar to plot
        theta (ndarray): Magnetic field angle w.r.t. z axis
        phi (ndarray): Magnetic field angle w.r.t. x axis
        cb_label (string): Colorbar label
        filenm (str, optional): File name. Defaults to "angular_plot".
        file_type (str, optional): File type. Defaults to "png".
        save_to_file (bool, optional): Save to file boolean flag. Defaults to False.
    """
    fig, ax = plt.subplots()
    CS = plt.contour(theta, phi, array, linewidths=1.0, colors="k")
    CS = plt.contourf(theta, phi, array, cmap=plt.get_cmap("viridis"))
    CB = plt.colorbar(CS)
    CB.ax.get_yaxis().labelpad = 15
    CB.ax.set_ylabel(cb_label, rotation=270)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"$\theta$ ˚")
    plt.ylabel(r"$\phi$ ˚")
    fig.tight_layout()
    plt.show()
    if save_to_file:
        plt.savefig(f"{filenm}.{file_type}", dpi=150, bbox_inches="tight")
        print(f"Saving to {filenm}.{file_type}")
