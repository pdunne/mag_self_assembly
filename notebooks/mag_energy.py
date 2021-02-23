import numpy as np
from scipy import constants

u0 = constants.mu_0
kB = constants.Boltzmann
muB = constants.physical_constants["Bohr magneton"][0]
Na = constants.Avogadro


def curie_moment(m_atom, B, T):
    """Calculates Curie-Law moment

    Args:
        m_atom (float/ndarray): magnetic moment in Bohr magneton µB
        B (float/ndarray): magnetic field in T
        T (float/ndarray): temperature in K

    Returns:
        [float/ndarray]: effective moment in Bohr magnetons µB
    """
    out = muB * m_atom ** 2 * B / 3 / kB / T
    return out


def calc_Um(m_eff, B):
    """Calculates magnetic energy

    Args:
        m_eff (float/ndarray): effective mangetic moment in µB
        B (float/ndarray): magnetic field in T
    """
    return -3 * m_eff * B * muB / 2


def calc_dG(Um, Vol):
    """Calculates the change in Gibbs Free Energy

    Args:
        Um (float/ndarray): Magnetic energy density J/m3
        Vol (float): object volume in m3

    Returns:
        float/ndarray: Gibbs Free Energy in J
    """

    return Um / 2 / Vol


def calc_Lc(Um, d, B, T):
    """Calculates the critical polymer length

    Args:
        Um ([float]): Magnetic energy of a monomer
        d ([float]): inter-layer spacing in nm
        B ([float]): magnetic field in T
        T ([float]): [description]

    Returns:
        [float]: critical length in nm
    """
    return -d * kB * T / Um


class Monomer:
    """Monomer class

    Assumes a monomer radius of 3.1 nm for volume calculation.

    Attributes:
        instances (list): List of instantiated monomers
    """

    instances = []

    def __init__(self, name, d, m, B, T):
        """[summary]

        Args:
            name (string): name
            d (float): interlayer spacing in nm
            m (float): magnetic moment in µB
            B (float): Magnetic field in T
            T (float): Temperature in K
        """
        self.name = name
        self.m = m
        self.B = B
        self.T = T
        self.d = d
        self.vol = d * np.pi * (3.1) ** 2
        self.m_eff = curie_moment(self.m, self.B, self.T)
        self.Um = calc_Um(self.m_eff, self.B)
        self.Lc = calc_Lc(self.Um, self.d, self.B, self.T)
        self.Nc = np.int(np.ceil(self.Lc / self.d))
        self.Uc = self.Um * self.Nc
        self.dG = self.Um / 2
        self.dGc = self.dG * self.Nc
        Monomer.instances.append(self)

    @classmethod
    def reset(cls):
        """Clears instance register and deletes monomer instantiated"""
        for mono in cls.instances:
            del mono
        cls.instances = []

    def __repr__(self):
        return "Monomer: m:%.2f Um:%.2f J/mol/mono Lc:%.2f nm Nc: %.d" % (
            self.m,
            self.Um * Na,
            self.Lc,
            self.Nc,
        )

    def __str__(self):
        return "Monomer: m:%.2f Um:%.2f J/mol/mono Lc:%.2f nm Nc: %.d" % (
            self.m,
            self.Um * Na,
            self.Lc,
            self.Nc,
        )
