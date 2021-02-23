# mag_self_assembly

Calculate the energy of a supramolecular assembly under applied magnetic fields.

This repository consists of magnetic energy and field calculations used for a
paper on [Magnetic Control over the Topology of Supramolecular Rod Networks](https://doi.org/10.26434/chemrxiv.12762269.v1)
, based on the change of length and structure of Gd(III)-DOTA-BTA
supramolecular assemblies.

The first notebook [Magnetic_Energy_Assemblies](notebooks/Magnetic_Energy_Assemblies.ipynb),
provides a Monomer class for analytically calculating energies and critical
lengths for supramolecular assemblies with different caged paramagnetic ions.

The second notebook [Self-Consistent](notebooks/Self-Consistent.ipynb),
provides a self-consistent approach to summing up the magnetic interactions of
each dipole moment under an external field, recalculating the induced magnetic
moments, and thus calculating the total energy of the assembly.

Two additional notebooks demonstrate inefficient methods to achieve the same
goal, based on the rotation of magnetic moments. The first,
[Macrospin_Approximation](notebooks/Macrospin_Approximation.ipynb), assumes a
single macrospin, with all moments aligned, and minimises the energy by
rotating a single ($\theta$, $\phi$) tuple. The second,
[Global_Optimisation](notebooks/Global_Optimisation.ipynb), performs a global
optimisation by freely rotating the all moments.

## 1. DOTA-BTA Structure

The DOTA-BTA monomers stack in a chiral structure, with

<img src="img/chiral_struct.png" alt="chiral_struct" width="400"/>

[1] P. Besenius, G. Portale, P. H. H. Bomans, H. M. Janssen, A. R. A. Palmans, and E. W. Meijer, _Controlling the Growth and Shape of Chiral Supramolecular Polymers in Water_, PNAS **107**, 17888 (2010).

## 2. Magnetic Energy Calculations

The energy of an induced paramagnetic magnetic moment in an external field is

$U_{m} = - \frac{1}{2} \mathbf{m}\cdot\mathbf{B}$

At room temperature, local dipolar fields, and exchange interactions from neighbouring rare earth atoms in a DOTA-BTA assembly are negligible. The former is confirmed with detailed calculations elaborated below, and the latter from SQUID magnetometry, which showed an exchange interaction of -0.7 K (see [main text](https://doi.org/10.26434/chemrxiv.12762269.v1)). Furthermore, we can assume a linear Curie law dependence of these atomic moments for an applied magnetic field $\mathbf{B}$, giving an effective atomic moment, $m_{eff}$, in units of Bohr magneton, $\mathrm{\mu_B}$

$$m_{eff} =   \frac{\mathrm{\mu_B}^2 m_\mathrm{atom}^2 B}{3 \mathrm{k_B} T}$$

where $m_\mathrm{atom}$ is the 0 K atomic moment also in units of $\mathrm{\mu_B}$. In this scheme, $\mathbf{m}$ is always collinear with $\mathbf{B}$, and the magnetic energy of a monomer, with three rare earth atoms is thus

$$ U_{mono} =   - \frac{\mathrm{\mu_B}^2 m_\mathrm{atom}^2 B^2}{ 2\mathrm{k_B} T} $$

We define a critical length $L_c$, as the minimum length where the magnitude of the magnetic energy of an assembly is sufficient to equal or overcome thermal energy:

$$ L_c = \frac{d \, \mathrm{k_B} T}{\left|U_{mono}\right|} $$

where $d$ is the inter-layer spacing, and the critical monomer number  is $ N_c = \left \lceil L_c/ d \right \rceil $. Alternatively, the critical length can be written as

$$L_c =  2d\left(\frac{ \mathrm{k_B} T}{\mu_B m_\mathrm{atom} B}  \right)^2$$

To generate the table below, we take $d$ = 0.35 nm, $B$ = 2 T, and $T$ = 298 K. The table units are: m: $\mu_B$, m_eff: %, Um: J mol$^{-1}$, Lc: nm, Nc: none.

| Ion                              |  $m$ | $m_{eff}$ | $U_m$ |  $dG$ | $L_c$ | $N_c$ |
| :------------------------------- | ---: | --------: | ----: | ----: | ----: | ----: |
| Ti<sup>3+</sup>, V<sup>4+</sup>  |  1.7 |      0.26 | -0.07 | -0.04 | 11918 | 34052 |
| Ti<sup>2+</sup>, V<sup>3+</sup>  |  2.8 |      0.42 |  -0.2 |  -0.1 |  4393 | 12553 |
| V<sup>2+</sup>, Cr<sup>3+</sup>  |  3.8 |      0.57 | -0.36 | -0.18 |  2385 |  6816 |
| Cr<sup>2+</sup>, Mn<sup>3+</sup> |  4.9 |      0.74 |  -0.6 |  -0.3 |  1435 |  4099 |
| Mn<sup>2+</sup>, Fe<sup>3+</sup> |  5.9 |      0.89 | -0.88 | -0.44 |   989 |  2828 |
| Fe<sup>2+</sup>, Co<sup>3+</sup> |  5.4 |      0.81 | -0.73 | -0.37 |  1181 |  3375 |
| Co<sup>2+</sup>, Ni<sup>3+</sup> |  4.8 |      0.72 | -0.58 | -0.29 |  1495 |  4272 |
| Ni<sup>2+</sup>                  |  3.2 |      0.48 | -0.26 | -0.13 |  3364 |  9611 |
| Cu<sup>2+</sup>                  |  1.9 |      0.29 | -0.09 | -0.05 |  9541 | 27261 |
| Ce<sup>3+</sup>                  |  2.5 |      0.38 | -0.16 | -0.08 |  5511 | 15746 |
| Pr<sup>3+</sup>                  |  3.5 |      0.53 | -0.31 | -0.15 |  2812 |  8034 |
| Nd<sup>3+</sup>                  |  3.4 |      0.51 | -0.29 | -0.15 |  2980 |  8513 |
| Sm<sup>3+</sup>                  |  1.7 |      0.26 | -0.07 | -0.04 | 11918 | 34052 |
| Eu<sup>3+</sup>                  |  3.4 |      0.51 | -0.29 | -0.15 |  2980 |  8513 |
| Gd<sup>3+</sup>                  |  8.9 |      1.34 | -1.99 |    -1 |   435 |  1243 |
| Tb<sup>3+</sup>                  |  9.8 |      1.47 | -2.42 | -1.21 |   359 |  1025 |
| Dy<sup>3+</sup>                  | 10.6 |      1.59 | -2.83 | -1.41 |   307 |   876 |
| Ho<sup>3+</sup>                  | 10.4 |      1.56 | -2.72 | -1.36 |   318 |   910 |
| Er<sup>3+</sup>                  |  9.5 |      1.43 | -2.27 | -1.14 |   382 |  1091 |
| Tm<sup>3+</sup>                  |  7.6 |      1.14 | -1.45 | -0.73 |   596 |  1704 |
| Yb<sup>3+</sup>                  |  4.5 |      0.68 | -0.51 | -0.25 |  1701 |  4860 |

## 3. Magnetic Dipole Calculations

### Self-Consistent Method

To ensure that, at room temperature, the local dipolar fields play no role in
the arrangement of the magnetic moments, we calculate the magnetic ground state
for a magnetic assembly of length N_m in an external magnetic field using a
lattice sum approach:

1. Impose an external field
$\mathbf{B} = B_x \mathbf{\hat{x}} + B_y \mathbf{\hat{y}} + B_z \mathbf{\hat{z}}$.
2. Generate the positions of each rare earth atom in the BTA assemblies:
knowing that in a single monomer those positions are $(x, y, z) = (0,r,0)$,
$\left(\frac{\sqrt{3}r}{2}, \frac{r}{\sqrt{2}}, 0\right)$, and
$\left(-\frac{\sqrt{3}r}{2}, \frac{r}{\sqrt{2}}, 0\right)$ where $r = 2.25$ nm,
that the stacking distance between monomers is $d$ = 0.35 nm, and that there is
a helicity of 10˚ rotation per layer, which for the i<sup>th</sup> atom of the
j<sup>th</sup> layer is:
$$ \begin{bmatrix}
    x' \\
    y' \\
    z'
\end{bmatrix}_{i,j}
=
\begin{bmatrix}
    \cos j\theta  & -\sin j\theta & 1  \\
    \sin j\theta  & \cos j\theta & 1  \\
    0             & 0 & 1 &
\end{bmatrix}
\begin{bmatrix}
    x  \\
    y \\
    z + j d
\end{bmatrix}_{i,j} $$

3. Calculate each local dipole moment using either a Curie-Law or Brillouin
function (at T = 298 K the difference is negligible) giving and array of
$\mathbf{m} = m_x \mathbf{\hat{x}} + m_y \mathbf{\hat{y}} + m_z \mathbf{\hat{z}}$.
4. At each atomic site, calculate the dipolar field due to all other moments in
the assembly using
$$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi} \left[\frac{3 \mathbf{m}
    (\mathbf{m} \cdot \mathbf{\hat{r}} ) } {r^5} - \frac{\mathbf{m}} {r^3}\right] $$
which is the magnetic field at a point in space due to a magnetic dipole,
resulting in a matrix
$\mathbf{B_{dip}} = B_{dip, x} \mathbf{\hat{x}} + B_{dip, y} \mathbf{\hat{y}} + B_{dip, y} \mathbf{\hat{z}}$

5. At each atomic site sum the external and dipole fields $B_{total} =  B_{ext} + B_{dip}$
6. Calculate a new moment m_new at each site using the local $B_{total}$
7. Calculate the mean L2 norm for the difference between the updated and old moment vectors at each site
$$ \lVert m \rVert_2 = \frac{1}{N_{atoms}} \sum (m_{new} - m_{old})^2 \rightarrow 0 $$

8. Repeat 3 – 7 with updated total field and moments until $ \lVert m \rVert_2 \leq threshold $

Other minimisation approaches are possible, such as by fixing the magnitude of
the magnetic moments, and minimising the energy cost function through a free
rotation of each moment. A first, simpler cost function, takes a macro-spin
approach, assigning one polar ($\theta$) – azimuthal ($\phi$) angle pair, with all atomic
moments rotating coherently, which is minimised using the L-BFGS-B algorithm.
The second allows the free rotation of each moment using basin hopping global
optimisation, with L-BFGS-B as the local optimiser. However, both approaches
are slower than the L2 norm method presented above, particularly the global
optimization method.

### TODO

1. Change functions to accept dictionaries or unwrap \*\*kwargs instead.
