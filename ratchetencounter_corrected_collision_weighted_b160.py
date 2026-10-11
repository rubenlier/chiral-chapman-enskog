#!/usr/bin/env python
# coding: utf-8

# # Ratchet collision experiment — corrected collision weighting and local contact refinement
# 
# This version keeps the logic of the original notebook, but makes the following corrections/improvements:
# 
# 1. **Collision-weighted relative speed:** samples
#    \[
#    P_{\rm coll}(g)\propto g^2\exp[-m g^2/(4k_BT)],
#    \]
#    appropriate when fitting a velocity-independent effective collision rule for the Boltzmann kernel.
# 2. **No angular-velocity rejection cutoff:** rare large \(|\omega/g|\) events are handled with an adaptive timestep instead of changing the thermal distribution.
# 3. **Much finer contact resolution without a global fine grid:** the code only constructs a grid inside the intersection of the two radius-\(R_{\max}\) bounding boxes. The default is equivalent to a global **512×512** spatial resolution near contact, but is substantially faster than evaluating the original global 128×128 masks at every step.
# 4. **Contact-time refinement:** once overlap is detected, the first contact time inside that timestep is refined by bisection.
# 5. **No integer truncation of the overlap centroid:** the physical coordinates of the refined overlap samples are averaged directly.
# 6. **Analytic mass and moment of inertia:** these no longer depend on raster resolution.
# 7. **Impact-parameter midpoint rule:** exactly `N_b` intervals cover \([-2R_{\max},2R_{\max}]\), avoiding the old 41-points-with-40-interval-weights bookkeeping issue.
# 
# The fitted effective-radius model is otherwise kept the same so that the result remains directly comparable to the original calculation.
# 

# In[1]:


import numpy as np

# -----------------------------------------------------------------------------
# Numerical accuracy controls
# -----------------------------------------------------------------------------

Lx, Ly = 6.0, 6.0

# This is an *effective* global resolution only: no global array of this size is
# created.  A fine grid is made only in the small rectangle where the two
# R_max bounding boxes intersect.
effective_grid_N = 512
contact_spacing = min(Lx, Ly) / (effective_grid_N - 1)

# Refine first-contact time only on timesteps where an overlap was detected.
contact_bisection_steps = 5

# Adaptive timestep: a ratchet rotates by at most this angle in one step.
max_rotation_per_step = 0.08


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def chiral_radius(phi, t, n_teeth, R_min, R_max, omega, phase0):
    """Boundary radius of the rotating sawblade."""
    psi = n_teeth * (phi - omega * t - phase0)
    u = np.mod(psi, 2*np.pi) / (2*np.pi)
    return R_min + (R_max - R_min) * u


def particle_occupancy(X, Y, cx, cy, t,
                       n_teeth, R_min, R_max, omega, phase0):
    """Boolean occupancy mask evaluated only on the supplied local grid."""
    xr = X - cx
    yr = Y - cy
    r = np.hypot(xr, yr)
    phi = np.arctan2(yr, xr)
    r_bound = chiral_radius(phi, t, n_teeth, R_min, R_max, omega, phase0)
    return r <= r_bound


def sawblade_mass_inertia(R_min, R_max, density=1.0):
    """
    Exact area (=mass for density 1) and polar moment of inertia.

    For the repeated linear sawtooth ramp, the full-angle integrals are
    independent of n_teeth:
        M = rho/2 ∫ R(phi)^2 dphi
        I = rho/4 ∫ R(phi)^4 dphi.
    """
    if R_max <= R_min:
        raise ValueError("Require R_max > R_min for the sawblade ramp.")

    dR = R_max - R_min
    int_R2 = 2*np.pi * (R_max**3 - R_min**3) / (3*dR)
    int_R4 = 2*np.pi * (R_max**5 - R_min**5) / (5*dR)
    mass = density * 0.5 * int_R2
    inertia = density * 0.25 * int_R4
    return mass, inertia


def local_overlap_info(cx1, cy1, cx2, cy2, t,
                       n_teeth, R_min, R_max,
                       omega1, phase1, omega2, phase2,
                       spacing=contact_spacing):
    """
    Return (x_contact, y_contact) if the discretized shapes overlap, else None.

    Crucially, only the intersection of the two radius-R_max bounding boxes is
    sampled.  This gives high local resolution without an expensive global grid.
    """
    xmin = max(cx1 - R_max, cx2 - R_max)
    xmax = min(cx1 + R_max, cx2 + R_max)
    ymin = max(cy1 - R_max, cy2 - R_max)
    ymax = min(cy1 + R_max, cy2 + R_max)

    # Cheap broad-phase rejection.
    if xmax < xmin or ymax < ymin:
        return None

    # Make the actual spacing <= requested spacing.
    nx = max(2, int(np.ceil((xmax - xmin) / spacing)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / spacing)) + 1)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    Xloc, Yloc = np.meshgrid(xs, ys, indexing='xy')

    occ1 = particle_occupancy(Xloc, Yloc, cx1, cy1, t,
                              n_teeth, R_min, R_max, omega1, phase1)
    occ2 = particle_occupancy(Xloc, Yloc, cx2, cy2, t,
                              n_teeth, R_min, R_max, omega2, phase2)
    overlap = occ1 & occ2

    if not np.any(overlap):
        return None

    # Use the physical coordinates directly.  The old notebook first averaged
    # integer indices and then truncated them, which introduced a directional bias.
    x_coll = float(np.mean(Xloc[overlap]))
    y_coll = float(np.mean(Yloc[overlap]))
    return x_coll, y_coll


def get_surface_normal(x_col, y_col, cx, cy, t,
                       n_teeth, R_min, R_max, omega, phase0):
    """Outward unit normal, including both spiral-ramp and radial-drop sides."""
    dx = x_col - cx
    dy = y_col - cy
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)

    psi = n_teeth * (phi - omega*t - phase0)
    psi_mod = np.mod(psi, 2*np.pi)
    slope = (R_max - R_min) * n_teeth / (2*np.pi)

    r_hat = np.array([np.cos(phi), np.sin(phi)])
    phi_hat = np.array([-np.sin(phi), np.cos(phi)])

    # Spiral/ramp side: n ∝ r_hat - (R'/R) phi_hat.
    ramp_normal = r_hat - (slope / max(r, 1e-14)) * phi_hat
    u = psi_mod / (2*np.pi)
    r_bound = R_min + (R_max - R_min) * u
    dist_ramp = abs(r - r_bound)

    # Radial drop.  For the increasing sawtooth convention, +phi_hat is outward.
    dist_phase_to_wall = min(psi_mod, 2*np.pi - psi_mod)
    dist_flat = r * dist_phase_to_wall / n_teeth

    normal = phi_hat if dist_flat < dist_ramp else ramp_normal
    normal /= np.linalg.norm(normal)
    return float(normal[0]), float(normal[1])


# -----------------------------------------------------------------------------
# Thermal sampling
# -----------------------------------------------------------------------------

def sample_relative_speed_for_collisions(mass, kT=1.0, rng=None):
    r"""
    Sample the incoming relative speed from the collision ensemble

        P_coll(g) ∝ g^2 exp[-m g^2/(4 kT)].

    Equivalently s=g^2 follows Gamma(shape=3/2, scale=4 kT/m).
    """
    if rng is None:
        rng = np.random.default_rng()
    return np.sqrt(rng.gamma(shape=1.5, scale=4.0*kT/mass))


# -----------------------------------------------------------------------------
# One scattering experiment
# -----------------------------------------------------------------------------

def simulate_one_trial(y1, y2, rotation_frequency_bias, v_x=1.0,
                       n_teeth=3, R_min=1.0, R_max=2.0,
                       t_max=6.0, dt=0.02, kT=1.0,
                       max_rotation_per_step=max_rotation_per_step,
                       spacing=contact_spacing,
                       bisection_steps=contact_bisection_steps,
                       rng=None):
    """
    Simulate one two-ratchet encounter.

    The physical relative speed g is sampled from the collision-weighted thermal
    ensemble.  The geometric simulation is rescaled to relative speed v_x
    (default 1), so the simulated angular velocities are omega * v_x/g.
    """
    if rng is None:
        rng = np.random.default_rng()

    mass, inertia = sawblade_mass_inertia(R_min, R_max)

    # Initial centers and translational velocities in simulation units.
    cx1, cy1 = 0.75, y1
    cx2, cy2 = 5.25, y2
    vx1, vy1 = +v_x/2.0, 0.0
    vx2, vy2 = -v_x/2.0, 0.0

    # Biased Maxwell-Boltzmann angular velocity distribution.
    omega_scale = np.sqrt(2.0*kT/inertia)
    phase1 = rng.uniform(0, 2*np.pi)
    phase2 = rng.uniform(0, 2*np.pi)

    g_init = sample_relative_speed_for_collisions(mass, kT=kT, rng=rng)
    omega1_init = rng.normal(
        loc=rotation_frequency_bias*omega_scale,
        scale=np.sqrt(kT/inertia),
    )
    omega2_init = rng.normal(
        loc=rotation_frequency_bias*omega_scale,
        scale=np.sqrt(kT/inertia),
    )

    # Only the ratio omega/g controls the relative amount of rotation during an
    # encounter.  No rejection/truncation of the thermal distribution is used.
    omega1 = omega1_init * v_x / g_init
    omega2 = omega2_init * v_x / g_init

    t = 0.0
    if_collision = 0.0

    while t < t_max:
        # Adaptive dt replaces the old high-|omega/g| rejection cutoff.
        max_omega = max(abs(omega1), abs(omega2))
        dt_rotation = np.inf if max_omega == 0 else max_rotation_per_step/max_omega
        dt_step = min(dt, dt_rotation, t_max - t)
        if dt_step <= 0:
            break

        sx1, sy1 = cx1, cy1
        sx2, sy2 = cx2, cy2
        t0 = t
        t1 = t + dt_step

        # Trial move to the end of the step.
        ex1, ey1 = sx1 + vx1*dt_step, sy1 + vy1*dt_step
        ex2, ey2 = sx2 + vx2*dt_step, sy2 + vy2*dt_step

        hit_end = local_overlap_info(
            ex1, ey1, ex2, ey2, t1,
            n_teeth, R_min, R_max,
            omega1, phase1, omega2, phase2,
            spacing=spacing,
        )

        if hit_end is None:
            cx1, cy1, cx2, cy2 = ex1, ey1, ex2, ey2
            t = t1
            continue

        # Refine first-overlap time within this timestep.  This extra work is
        # only performed on collision timesteps.
        hit_start = local_overlap_info(
            sx1, sy1, sx2, sy2, t0,
            n_teeth, R_min, R_max,
            omega1, phase1, omega2, phase2,
            spacing=spacing,
        )

        if hit_start is None:
            lo, hi = 0.0, dt_step
            hit = hit_end
            for _ in range(bisection_steps):
                mid = 0.5*(lo + hi)
                mx1, my1 = sx1 + vx1*mid, sy1 + vy1*mid
                mx2, my2 = sx2 + vx2*mid, sy2 + vy2*mid
                hit_mid = local_overlap_info(
                    mx1, my1, mx2, my2, t0 + mid,
                    n_teeth, R_min, R_max,
                    omega1, phase1, omega2, phase2,
                    spacing=spacing,
                )
                if hit_mid is None:
                    lo = mid
                else:
                    hi = mid
                    hit = hit_mid
            tau = hi
        else:
            # A small discretized overlap can persist for a step or two while
            # already separating after an impulse.
            tau = 0.0
            hit = hit_start

        t_coll = t0 + tau
        cx1, cy1 = sx1 + vx1*tau, sy1 + vy1*tau
        cx2, cy2 = sx2 + vx2*tau, sy2 + vy2*tau
        x_coll, y_coll = hit

        # Choose the surface whose contact point is farther from its pointy tip.
        r1_sq = (x_coll-cx1)**2 + (y_coll-cy1)**2
        r2_sq = (x_coll-cx2)**2 + (y_coll-cy2)**2

        if r1_sq >= r2_sq:
            nx2, ny2 = get_surface_normal(
                x_coll, y_coll, cx2, cy2, t_coll,
                n_teeth, R_min, R_max, omega2, phase2,
            )
            normal_x, normal_y = -nx2, -ny2
        else:
            normal_x, normal_y = get_surface_normal(
                x_coll, y_coll, cx1, cy1, t_coll,
                n_teeth, R_min, R_max, omega1, phase1,
            )

        lever_arm1 = normal_x*(y_coll-cy1) - normal_y*(x_coll-cx1)
        lever_arm2 = -normal_x*(y_coll-cy2) + normal_y*(x_coll-cx2)

        v_rel_normal = (
            (vx1-vx2)*normal_x + (vy1-vy2)*normal_y
            - omega1*lever_arm1 - omega2*lever_arm2
        )

        if v_rel_normal >= 0:
            if_collision = 1.0

            impulse = v_rel_normal / (
                1.0/mass
                + (lever_arm1**2 + lever_arm2**2)/(2.0*inertia)
            )

            vx1 -= impulse*normal_x/mass
            vy1 -= impulse*normal_y/mass
            vx2 += impulse*normal_x/mass
            vy2 += impulse*normal_y/mass

            omega1_new = omega1 + lever_arm1*impulse/inertia
            omega2_new = omega2 + lever_arm2*impulse/inertia

            # Keep the physical orientation continuous across the instantaneous
            # change in angular velocity.
            phase1 += (omega1 - omega1_new)*t_coll
            phase2 += (omega2 - omega2_new)*t_coll
            omega1, omega2 = omega1_new, omega2_new

        # Finish the remaining fraction of the timestep with post-collision velocities.
        remaining = dt_step - tau
        cx1 += vx1*remaining
        cy1 += vy1*remaining
        cx2 += vx2*remaining
        cy2 += vy2*remaining
        t = t1

    # Convert simulation-unit velocities back to the sampled physical speed scale.
    velocity_scale = g_init / v_x
    return {
        "collision": if_collision,
        "velocity1": (velocity_scale*vx1, velocity_scale*vy1),
        "velocity2": (velocity_scale*vx2, velocity_scale*vy2),
        "g_init": g_init,
        "omega1_init": omega1_init,
        "omega2_init": omega2_init,
        "omega1": velocity_scale*omega1,
        "omega2": velocity_scale*omega2,
        "mass": mass,
        "moment_of_inertia": inertia,
    }


# -----------------------------------------------------------------------------
# Differential scattering cross section at one impact parameter
# -----------------------------------------------------------------------------

def cross_section_total(y1, y2, init_frequency_bias,
                        n_trials=1000, rng=None, **sim_kwargs):
    if rng is None:
        rng = np.random.default_rng()

    angles = np.zeros(n_trials)
    weights = np.zeros(n_trials)

    for i in range(n_trials):
        result = simulate_one_trial(
            y1, y2,
            rotation_frequency_bias=init_frequency_bias,
            rng=rng,
            **sim_kwargs,
        )

        vx1, vy1 = result["velocity1"]
        vx2, vy2 = result["velocity2"]
        weight = result["collision"]
        weights[i] = weight

        if weight == 0:
            # Value is irrelevant because the histogram weight is zero.
            angles[i] = np.pi
        else:
            angles[i] = np.arctan2(vy2-vy1, vx2-vx1)

    return {
        "scattering_angles": angles,
        "if_collision": weights,
    }


mass_check, inertia_check = sawblade_mass_inertia(1.0, 2.0)
print(f"Local contact spacing = {contact_spacing:.5f} (effective global N={effective_grid_N})")
print(f"Analytic mass = {mass_check:.6f}, analytic moment of inertia = {inertia_check:.6f}")


# In[2]:


# -----------------------------------------------------------------------------
# Build the thermally/collision-weighted differential scattering cross section
# -----------------------------------------------------------------------------

R_min = 1.0
R_max = 2.0
n_teeth = 3

# Exactly this many impact-parameter intervals cover [-2 R_max, 2 R_max].
no_of_b_values = 160  # 4x finer impact-parameter grid
trials_per_b = 1000
no_of_bins = 50

# Bias in units of sqrt(2 kT / I), as in the paper.
rotation_bias = 0.4

# Set to an integer for reproducibility; use None for a fresh random run.
seed = 12345
rng = np.random.default_rng(seed)

b_min, b_max = -2*R_max, 2*R_max
Delta_b = (b_max - b_min) / no_of_b_values

# Midpoint rule: N intervals, N sample points, no endpoint over-counting.
b_values = b_min + (np.arange(no_of_b_values) + 0.5)*Delta_b

Delta_chi = 2*np.pi/no_of_bins
bins = np.linspace(-np.pi, np.pi, no_of_bins + 1)

print("{Impact parameter, number of collisions}")

angles_all = []
weights_all = []

for b in b_values:
    # Keep the center-of-mass y coordinate fixed at 3 while imposing y1-y2=b.
    y1 = 3.0 + b/2.0
    y2 = 3.0 - b/2.0

    result = cross_section_total(
        y1, y2,
        init_frequency_bias=rotation_bias,
        n_trials=trials_per_b,
        rng=rng,
        n_teeth=n_teeth,
        R_min=R_min,
        R_max=R_max,
    )

    angles_all.append(result["scattering_angles"])
    weights_all.append(result["if_collision"])
    print(f"{{{b:.3f}, {np.sum(result['if_collision']):.0f}}},")

angles_all = np.asarray(angles_all)
weights_all = np.asarray(weights_all)

# Each successful trial represents an impact-parameter width Delta_b/N_trials.
# Dividing by Delta_chi gives d sigma / d chi.
trial_cross_section_weight = Delta_b/(trials_per_b*Delta_chi)

hist, bin_edges = np.histogram(
    angles_all.ravel(),
    bins=bins,
    weights=(weights_all*trial_cross_section_weight).ravel(),
)

print("The differential cross section histogram:")
print(hist)


# In[3]:


# -----------------------------------------------------------------------------
# Fit the same two-effective-radii toy model as in the original notebook
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
bin_width = np.diff(bin_edges)


def fitfunc(angle_0_to_pi, radius):
    return radius*np.sin(angle_0_to_pi/2.0)


half = len(bin_centers)//2
angle_shifted = bin_centers + np.pi  # map [-pi,pi] -> [0,2pi]

r1_opt, r1_cov = curve_fit(
    fitfunc,
    angle_shifted[:half],
    hist[:half],
)
r2_opt, r2_cov = curve_fit(
    fitfunc,
    angle_shifted[half:],
    hist[half:],
)

r1 = r1_opt[0]
r2 = r2_opt[0]
r_eff = 0.5*(r1+r2)
epsilon = (r2-r1)/(2*r_eff)

print(f"Effective radius for one chirality: {r1:.4f}")
print(f"Effective radius for the other chirality: {r2:.4f}")
print(f"r = {r_eff:.4f}")
print(f"epsilon = {epsilon:.5f}")

# Smooth fitted curve.
theta_fit = np.linspace(0, 2*np.pi, 1000)
fit_curve = np.where(
    theta_fit < np.pi,
    r1*np.sin(theta_fit/2.0),
    r2*np.sin(theta_fit/2.0),
)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.bar(bin_centers, hist, width=bin_width, edgecolor="black", alpha=0.7)
ax.plot(theta_fit + np.pi, fit_curve, lw=2, label="Fitted effective-radius model")
ax.tick_params(axis="both", labelsize=14)
ax.set_theta_zero_location("W")
ax.set_theta_direction(-1)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))

plt.savefig("scattering_fit_corrected_b160.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# ## Suggested zero-bias check
# 
# For the specific question about residual chirality at zero bias, rerun the cross-section cell with
# 
# ```python
# rotation_bias = 0.0
# ```
# 
# and compare `epsilon` as you vary only the two numerical controls
# 
# ```python
# effective_grid_N = 384, 512, 768
# contact_bisection_steps = 4, 5, 6
# ```
# 
# Because the fine grid is local, increasing `effective_grid_N` is much cheaper than increasing a global `Nx × Ny` grid. The most important diagnostic is whether the fitted zero-bias `epsilon` trends toward zero under this refinement.
# 

# In[ ]:





# In[5]:


# -----------------------------------------------------------------------------
# Fit the same two-effective-radii toy model as in the original notebook
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
bin_width = np.diff(bin_edges)

def fitfunc(angle_0_to_pi, radius):
    return radius*np.sin(angle_0_to_pi/2.0)

half = len(bin_centers)//2
angle_shifted = bin_centers + np.pi  # map [-pi,pi] -> [0,2pi]

r1_opt, r1_cov = curve_fit(
    fitfunc,
    angle_shifted[:half],
    hist[:half],
)
r2_opt, r2_cov = curve_fit(
    fitfunc,
    angle_shifted[half:],
    hist[half:],
)

r1 = r1_opt[0]
r2 = r2_opt[0]
r_eff = 0.5*(r1 + r2)
epsilon = (r2 - r1)/(2*r_eff)

print(f"Effective radius for one chirality: {r1:.4f}")
print(f"Effective radius for the other chirality: {r2:.4f}")
print(f"r = {r_eff:.4f}")
print(f"epsilon = {epsilon:.5f}")

# Smooth fitted curve
theta_fit = np.linspace(0, 2*np.pi, 1000)
fit_curve = np.where(
    theta_fit < np.pi,
    r1*np.sin(theta_fit/2.0),
    r2*np.sin(theta_fit/2.0),
)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

# Histogram
ax.bar(
    bin_centers,
    hist,
    width=bin_width,
    edgecolor="black",
    alpha=0.7
)

# Fitted model: RED line
ax.plot(
    theta_fit + np.pi,
    fit_curve,
    color="red",
    lw=2.5,
    label="Fitted chiral hard-disk model"
)

# Polar formatting
ax.set_theta_zero_location("W")
ax.set_theta_direction(-1)

# --- Make radial labels cleaner ---
# Move radial tick labels to a quieter angular location
ax.set_rlabel_position(135)

# Use fewer radial ticks to avoid clutter
rmax = max(np.max(hist), np.max(fit_curve))
# rticks = np.linspace(0, rmax, 6)
# ax.set_rticks(rticks)

# # Smaller label size and a bit more padding
# ax.tick_params(axis="x", labelsize=16)
# ax.tick_params(axis="y", labelsize=11, pad=8)


# Move radial tick labels away from the crowded 0-degree direction
ax.set_rlabel_position(135)

ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=11, pad=8)

# Legend
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=12)

plt.savefig("scattering_fit_correctedred_b160.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# In[ ]:





# In[ ]:




