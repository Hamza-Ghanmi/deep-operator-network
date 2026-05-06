# Lamé's Thick-Walled Hollow Sphere

A 3-D linear elasticity problem with an exact closed-form solution, used here as a
benchmark for operator learning with **DeepONet3D**.

---

## 1. Physical Setup

A hollow sphere made of isotropic linear-elastic material is subjected to uniform
internal pressure on its inner surface and uniform external pressure on its outer
surface.

| Quantity | Symbol | Fixed value |
|---|---|---|
| Inner radius | a | **0.4 m** |
| Outer radius | b | **1.0 m** |
| Wall thickness | b − a | 0.6 m |

The domain Ω is the spherical shell:

```
Ω = { (x, y, z) ∈ ℝ³  :  a ≤ √(x²+y²+z²) ≤ b }
```

### Material model

Isotropic linear elasticity (Hooke's law):

```
σ_ij = λ ε_kk δ_ij + 2μ ε_ij
```

where the Lamé constants relate to engineering constants as:

```
λ = E ν / [(1+ν)(1−2ν)]
μ = E / [2(1+ν)]
```

### Loading

| Surface | Condition |
|---|---|
| Inner sphere r = a | Uniform internal pressure p_i (directed outward = +r) |
| Outer sphere r = b | Uniform external pressure p_e (directed inward = −r) |

The loading is **spherically symmetric**: both pressures are uniform over their
respective surfaces, so the solution has no angular dependence.

---

## 2. Governing Equations

### Navier–Cauchy equilibrium (no body forces)

```
(λ + 2μ) ∇(∇·u) − μ ∇×(∇×u) = 0
```

In spherical coordinates (r, θ, φ) with spherical symmetry (u = u_r(r) r̂, no
angular dependence), the vector equation reduces to a single scalar ODE for u_r(r):

```
d/dr [ (1/r²) d/dr(r² u_r) ] = 0
```

Expanding:

```
d²u_r/dr² + (2/r) du_r/dr − (2/r²) u_r = 0
```

This is an Euler–Cauchy ODE with general solution:

```
u_r(r) = C·r + D/r²
```

### Strain–displacement relations (spherical)

```
ε_r  = du_r/dr
ε_θ  = ε_φ  = u_r / r
```

### Stress–strain relations (spherical, isotropic)

```
σ_r = (λ + 2μ) ε_r + 2λ ε_θ
σ_θ = σ_φ = λ ε_r + (λ + 2μ + λ) ε_θ = λ ε_r + 2(λ + μ) ε_θ
```

---

## 3. Analytical Solution (Lamé, 1852)

### Applying boundary conditions

The boundary conditions on the stress are:

```
σ_r(a) = −p_i      (compressive → negative convention)
σ_r(b) = −p_e
```

Substituting u_r = C·r + D/r² into the stress expressions and applying the two BCs
gives a 2×2 linear system. The solution yields the Lamé constants **C1** and **C2**:

```
C1 = (p_i·a³ − p_e·b³) / (b³ − a³)
C2 = (p_i − p_e)·a³·b³ / (b³ − a³)
```

With the fixed geometry (a = 0.4, b = 1.0):

```
a³ = 0.064,  b³ = 1.0,  b³ − a³ = 0.936
```

### Displacement field

**Radial displacement magnitude (m):**

```
u_r(r) = (1/E) · [ C1·(1 − 2ν)·r  +  C2·(1 + ν) / (2r²) ]
```

**Full 3-D displacement vector:**

```
u_x(x,y,z) = u_r(r) · x/r
u_y(x,y,z) = u_r(r) · y/r
u_z(x,y,z) = u_r(r) · z/r
```

where r = √(x²+y²+z²).

### Stress field

**Radial stress (Pa):**

```
σ_r(r) = C1 − C2 / r³
```

At the boundaries: σ_r(a) = C1 − C2/a³ = −p_i  ✓  and  σ_r(b) = C1 − C2/b³ = −p_e  ✓

**Hoop/circumferential stress (Pa):**  (equal in both θ and φ by symmetry)

```
σ_θ(r) = σ_φ(r) = C1 + C2 / (2r³)
```

The hoop stress is always larger in magnitude at r = a than at r = b.

### Derived quantities

**Von Mises equivalent stress (Pa):**

With principal stresses σ₁ = σ_r, σ₂ = σ₃ = σ_θ:

```
σ_vm = √{ [(σ_r−σ_θ)² + (σ_θ−σ_θ)² + (σ_θ−σ_r)²] / 2 }
      = |σ_r − σ_θ|
      = |C2| · 3 / (2r³)
```

σ_vm is **monotonically decreasing** with r (highest at inner wall, lowest at outer wall),
which is the physically correct result: the inner wall is the critical location for
failure in a pressurised vessel.

**Cauchy stress tensor in Cartesian coordinates:**

```
σ_ij = σ_θ·δ_ij + (σ_r − σ_θ)·(x_i·x_j / r²)
```

Explicit components:

```
σ_xx = σ_θ + (σ_r − σ_θ)·(x/r)²
σ_yy = σ_θ + (σ_r − σ_θ)·(y/r)²
σ_zz = σ_θ + (σ_r − σ_θ)·(z/r)²
σ_xy = (σ_r − σ_θ)·(xy/r²)
σ_xz = (σ_r − σ_θ)·(xz/r²)
σ_yz = (σ_r − σ_θ)·(yz/r²)
```

---

## 4. Parameters

### Fixed (geometry)

| Symbol | Value | Description |
|---|---|---|
| a | 0.4 m | Inner radius |
| b | 1.0 m | Outer radius |

### Variable (sampled per dataset case)

| Symbol | Range | Units | Sampling |
|---|---|---|---|
| p_i | [1, 20] | MPa | Uniform |
| p_e | [0, min(5, p_i − 0.1)] | MPa | Conditional uniform |
| E | [50, 250] | GPa | **Log-uniform** |
| ν | [0.20, 0.45] | — | Uniform |

Log-uniform sampling for E matches the physical intuition that a factor-of-2 change
in stiffness is equally significant whether E goes from 50→100 GPa or 100→200 GPa.

Constraint p_e < p_i ensures a non-trivial deviatoric term (C2 ≠ 0). If p_i = p_e,
then C2 = 0 and σ_vm = 0 everywhere — a degenerate (purely hydrostatic) case.

---

## 5. Geometry for CAD

### Dimensions

| Feature | Value |
|---|---|
| Outer sphere radius | **1000 mm** (1.0 m) |
| Inner sphere radius | **400 mm** (0.4 m) |
| Center | Origin (0, 0, 0) |
| Material volume | Shell between the two surfaces |

### Construction steps

1. Create a solid sphere of radius 1000 mm centred at the origin.
2. Create a solid sphere of radius 400 mm centred at the origin.
3. Subtract the inner sphere from the outer sphere (**Boolean subtraction**).
   The result is a hollow spherical shell.

### Sector option (recommended for meshing efficiency)

Because the problem is fully spherically symmetric, only a **1/8 octant sector**
(x ≥ 0, y ≥ 0, z ≥ 0) needs to be modelled with symmetry boundary conditions on
the three flat faces.

Intersect the hollow sphere with a cube [0, 1200 mm]³ to extract the positive octant.

### Mesh guidance

| Parameter | Recommendation |
|---|---|
| Radial element size | 50 mm (~12 elements across wall) |
| Refinement | Increase density near r = 400 mm (inner surface — highest stress) |
| Element type | Quadratic hexahedral (C3D20) or tetrahedral (C3D10) |
| Mesh type | Structured hex aligned with spherical coordinates gives cleanest results |

The trained DeepONet3D accepts arbitrary `(x, y, z)` query points inside the shell,
so any mesh node set can be used for inference — not just the training grid.

---

## 6. Boundary Conditions for FEM Validation

Use these to validate the analytical solution against a FEM solver (Abaqus, FEniCS,
OpenFOAM, etc.):

| Surface | Type | Value |
|---|---|---|
| Inner surface (r = 0.4 m) | Pressure | p_i [Pa], positive = inward normal |
| Outer surface (r = 1.0 m) | Pressure | p_e [Pa], positive = inward normal |
| Symmetry face x = 0 (if sector) | Symmetry (u_x = 0) | — |
| Symmetry face y = 0 (if sector) | Symmetry (u_y = 0) | — |
| Symmetry face z = 0 (if sector) | Symmetry (u_z = 0) | — |

**Material inputs:** E (Young's modulus, Pa), ν (Poisson's ratio, dimensionless).

---

## 7. Operator Learning Formulation

### Operator

```
G : (p_i, p_e, E, ν)  →  σ_vm(x, y, z)  ∈ ℝ^{N_PTS}
```

- **Input (branch net):** 4 scalar physics parameters (normalised to [0, 1])
- **Query (trunk net):** 3-D Cartesian coordinates (x, y, z) of N_PTS = 16,000 points
- **Output:** Von Mises stress at each query point

The operator G is an infinite-dimensional map: given the parameters, it predicts the
stress field at *any* point in the shell — including points not seen during training.

### Why DeepONet is suitable

| Property | Reason it matters |
|---|---|
| Fixed geometry, variable parameters | Branch encodes "which case"; trunk encodes "where in the domain" |
| Smooth, analytic field | No discontinuities; MLP approximation works well |
| Radially symmetric field | σ_vm depends only on r = ‖x‖; the trunk must learn this from data |
| 3-D query coordinates | Extends naturally from 2-D (heat) to 3-D (elasticity) |

### Architecture

```
DeepONet3D
├── branch: MLP(4 → 256 → 256 → 256 → 128)   — maps params to ℝ^128
├── trunk:  MLP(3 → 256 → 256 → 256 → 128)   — maps (x,y,z) to ℝ^128, tanh on output
└── output: einsum('bp,np→bn') + bias          — (batch, N_PTS)
```

Total trainable parameters: ~397,000.

---

## 8. Dataset Statistics

| Quantity | Value |
|---|---|
| Total cases | 5,000 |
| Train / val / test | 3,500 / 750 / 750 |
| Query points per case | 16,000 (NR=20 × NΘ=20 × NΦ=40, spherical grid) |
| Records in parquet | 5,000 (1 per case — static problem, no time dimension) |
| Fields stored per record | σ_vm, u_r, σ_r, σ_θ  (each 16,000 float32 values) |
| Estimated file size | ~320 MB total (Snappy-compressed) |

### Query grid

Points are placed on a structured spherical grid in the shell:

- **Radial (r):** 20 linearly spaced values from a = 0.4 to b = 1.0 m
- **Polar (θ):** 20 values in (0, π), excluding exact poles to avoid degeneracy
- **Azimuthal (φ):** 40 values in [0, 2π), excluding the duplicate endpoint

Converted to Cartesian: `x = r sin θ cos φ`, `y = r sin θ sin φ`, `z = r cos θ`.

### Stress ranges (approximate, for normalisation)

| Field | Typical range | Normalisation scale |
|---|---|---|
| σ_vm | 0 – ~24 MPa | 100 MPa (σ_vm_scale = 1×10⁸ Pa) |
| σ_r | −20 – 0 MPa | — |
| σ_θ | 0 – 30 MPa | — |
| u_r | 0 – ~0.1 mm | — |
