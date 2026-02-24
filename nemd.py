


#code developed by Ruben Lier and Pawel Matus



import numpy as np
import pickle
import os

X, Y = 0, 1



from datetime import datetime


class MDSimulation:
    def __init__(self, pos, vel, r1, m1,
                 eps_strainrate,  # |e_dot| per unit time
                 T_period,         # period in steps (square-wave half period = T_period/2)
                 fps,delt):
        #the goal here is to connect information about initial conditions which is put into the simulation to
        #the simulation data.... 
        self.pos = np.asarray(pos, dtype=float)      # shape (n,2), inside [0,1)
        self.vel = np.asarray(vel, dtype=float)      # peculiar velocities
        self.n = self.pos.shape[0]
        self.radius = r1
        self.mass = m1
        self.nsteps = 0
        self.chirality = 0
        # count the collisions !!
        self.ncounter =0 
        self.ncounternet =0 

        self.delt = delt






        # shear / SLLOD / LE parameters
        self.eps = float(eps_strainrate)   # magnitude of strain rate e_dot
        self.T = int(T_period)             # period in *steps* for the square wave
        self.dt = 1.0/fps                  # stored so we can use it inside
        self.gamma = 0.0                   # accumulated shear strain
        # accumulators for sign-weighted averages
        self.sum_sgn_sigma      = np.zeros((2,2))
        self.sum_sgn_sigma_kin  = np.zeros((2,2))
        self.sum_sgn_sigma_col = np.zeros((2,2))
        self.sum_sgn_edot = 0.0
        #this collects the irving kirkwood so eventually you divide this by T!!
        self.sum_sigma = np.zeros((2,2))
        self.n_accum = 0

        # per-step collisional impulse log (cleared each step)
        self.coll_impulses = []  # list of tuples (J_ij_vector, r_ij_vector) with minimum-image r_ij at collision
        #sums over x and y component and means over the particles
        self.kBT_target = self.mass * np.mean(np.sum(self.vel**2, axis=1)) / 2.0

    def _unwrap_min_image_shear(self, dr):
        #so dx is the direction along which the displacement takes place? 
        # so apparetnly dr is some object that has all sorts of components but only the last one matters, check that later !!
        dx, dy = dr[...,0], dr[...,1]
        # this means something happens if dy < -0.5 or dy > 0.5, but remember this thing is only meaningful if the particles are actually in range
        #which rarely happens precisely at the periodic boundary !!! 
        m2 = np.rint(dy).astype(int)
        dy = dy - m2
        dx = dx - self.gamma * m2
        m1 = np.rint(dx).astype(int)
        # this fixes a potential issue with gamma
        dx = dx - m1
        return np.stack((dx, dy), axis=-1)


    # this fixes the position, this is very important
    def _wrap_pos_le(self, r_new):
        # something happens when y is more than 1 or less than 0, that is how np.floor works!!!
        x, y = r_new
        y_shift = np.floor(y)
        if y_shift != 0.0:
            y -= y_shift
            x -= y_shift * self.gamma  # Lees–Edwards lateral shift
        #this is very important this makes sure there is never an issue!!! 
        x -= np.floor(x)
        return np.array([x, y])

    # this is the thing that is used for the thermostat, this is a single uniform constant which is multiplied with the corresponding v!!
    def _alpha_isokinetic(self, e_dot):
        # Gaussian multiplier that enforces d/dt (Σ ½ m v^2) = 0 under SLLOD
        vx, vy = self.vel[:, 0], self.vel[:, 1]
        num = e_dot * np.sum(vx * vy)              # m·γ̇ Σ v_x v_y
        den = np.sum(vx*vx + vy*vy) + 1e-300       # m Σ v^2  (eps avoids 0/0)
        return num / den

    def _chirality_sign(self, a, b):
        cross_z = a[0]*b[1] - a[1]*b[0]
        return 1 if cross_z > 0 else -1

    # ---------- stress measurement ----------
    def _compute_kinetic_stress(self):
        # σ^kin = -(1/A) sum_i m v_i v_i  (A=1 here)
        # this is all good!!! 
        # this None stuff is a way to do a kronecker product for the second component of self.vel[:, :, None] which is the spatial one!!!
        # the sum is axis =0 which is the particle dimension!!!
        return -self.mass * (self.vel[:, :, None] * self.vel[:, None, :]).sum(axis=0)

    def _compute_collisional_stress(self):
        # Irving–Kirkwood collisional part over the time interval Δt:
        # σ^coll = -(1/(A Δt)) sum_collisions J_ij ⊗ r_ij  (A=1)
        # in absence of oddity you should expect the stress to be fully symmetric !!!
        acc = np.zeros((2,2))
        for J, rij in self.coll_impulses:
            acc += np.outer(J, rij)
        return -(1.0 / self.dt) * acc

    # ---------- one time step ----------
    def advance(self, dt):
        self.dt = dt  # keep in sync
        #this // means rounded division !!
        sgn = 1.0 if ((self.nsteps // (self.T//2)) % 2 == 0) else -1.0

        self.nsteps += 1
        e_dot = sgn * self.eps

        #so gamma is like the total accumulated displacement
        self.gamma += e_dot * dt



        self.pos[:, X] += dt * (self.vel[:, X] + e_dot * self.pos[:, Y])
        self.pos[:, Y] += dt * self.vel[:, Y]

        for i in range(self.n):
            self.pos[i] = self._wrap_pos_le(self.pos[i])

      
        # Minimum image convention for collisions
        delta = self.pos[:, None, :] - self.pos[None, :, :]        
        # make sure it is the shortest distance within the periodic box
        delta = self._unwrap_min_image_shear(delta)
        #distance measured with inner product for final spatial dimension
        delta_sq = np.sum(delta * delta, axis=-1)        # same as (delta**2).sum(-1)

        two_r  = 2.0 * self.radius
        upper2 = (two_r + self.delt)**2       # loose shell (your prefilter)
        lower2 = (two_r - self.delt)**2       # stricter shell for favored sense

        iarr, jarr = np.where(delta_sq < upper2)
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        self.coll_impulses.clear()

        for i, j in zip(iarr, jarr):

            # so here you do it in the convention that makes sense to me, rij with the i being the plus
            # this already did the right wrapping
            rij = delta[i,j]
            r2 = np.dot(rij, rij)
            if r2 == 0.0: 
                                        continue
            nor = rij / np.sqrt(r2)

            # relative peculiar velocity
            vij_total = (self.vel[i] - self.vel[j]) 

            v_rel_n = np.dot(vij_total, nor)

            # check if they are separating
            if v_rel_n > 0:
                continue  # separating

            #here you need to work out the chirality thing
            sign = self._chirality_sign(vij_total, nor)
            if sign > 0:
                if r2 > lower2:
                    continue
            self.ncounter +=1  
            self.ncounternet += sign

            # elastic reflection along normal: change in peculiar velocities
            # impulse on i: J_ij = -m * Δv_i = -m * (-(v_rel_n) * nor) = m * v_rel_n * nor
            # but Δv_i = - v_rel_n * nor  (since we add -delta_v to i below)
            delta_v = v_rel_n * nor
            # this velocity is the PECULIAR velocity
            self.vel[i] -= delta_v
            self.vel[j] += delta_v

            # why this sign because it is a -v kick times the r !! 
            J_ij = self.mass * (-delta_v)  # mass * change in vel of i (positive into i) = m*(+delta_v)

            # log pair impulse for Irving–Kirkwood collisional stress over this dt
            # so once you have these ingredients 
            # you just store everything, next time step you clear the tuple
            self.coll_impulses.append((J_ij, rij))


        # Gaussian isokinetic thermostat (Evans–Morriss 6.44–6.45)
        alpha = self._alpha_isokinetic(e_dot)

        # Explicit Euler update for SLLOD + thermostat between collisions:
        # v_x' = v_x + dt*(alpha v_x - e_dot v_y)
        # v_y' = v_y + dt*(alpha v_y)
        self.vel[:, X] += dt * (alpha * self.vel[:, X] - e_dot * self.vel[:, Y])
        self.vel[:, Y] += dt * (alpha * self.vel[:, Y])



        #the brute force temperature rescaling
        self.vel -= np.mean(self.vel, axis=0)
        kBT_now = self.mass * np.mean(np.sum(self.vel**2, axis=1)) / 2.0
        scale = np.sqrt(self.kBT_target / kBT_now)
        self.vel *= scale





        #evans says you should only use the pecular velocity for the kinetic stress !! 
        sigma_kin = self._compute_kinetic_stress()
        sigma_col = self._compute_collisional_stress()
        sigma = sigma_kin + sigma_col

        # accumulators (plain and sign-weighted)
        self.sum_sigma += sigma
        self.sum_sgn_sigma +=     sgn * sigma  #this is really what you need!!! 
        self.sum_sgn_sigma_col += sgn * sigma_col
        self.sum_sgn_sigma_kin += sgn * sigma_kin  
        self.sum_sgn_edot += sgn * e_dot  # equals |e_dot| in time average
        self.n_accum += 1

        # chirality diagnostic retained from your code
        self.chirality += 0
        return sigma, sigma_kin, sigma_col, e_dot, sgn



import os, json, pickle, numpy as np

# If you use Y as the y-index later, define it once:
X, Y = 0, 1

# def slug(x):
#     """Make a short, filesystem-friendly label for a float."""
#     return f"{x:.3e}".replace("+", "").replace("-", "m").replace(".", "p")

def temp_from_vel(vel, m):
    v2_mean = np.mean(np.sum(vel**2, axis=1))
    return m * v2_mean / 2.0  # kB = 1

# ------------------------------
# GLOBAL / SHARED SETTINGS
# ------------------------------
n = 1000
# you want the filling fraction to be VERY low, probably 0.1%
# you need to make sure r1 is larger since it will now be much tinier for the - chirality collision
r1 = 25e-4
m1 = 1.0
n_runs = 1
# this cannot be too small because then there is only noise!!!!

base_seed = 20251013  # <- pick any constant


print("Packing fraction phi =", np.pi*(r1**2)*n)  # A = 1

for t_idx1 in range(4):
    t_idx = 0
    # ------------------------------
    # PARAMETERS THAT CHANGE WITH "TEMPERATURE"
    # ------------------------------     # originally this was of course (1 + thingmod2) with range range(2)
    thingmod = t_idx // 5
    thingmod2 = t_idx % 5
    sbar = 0.25 * np.sqrt(1/2 + thingmod2/2)
    FPS = (2**14)
    dt = 1 / FPS
    T_period_steps = int(5000*15*4)
    frames = T_period_steps*6
    eps_strainrate = 1/2+t_idx1/2
    delt = (0.5) * (2 * r1)
    now = datetime.now(); print(now.time())


    # NOTE: You sample speeds as s ~ U(0, s_max) with s_max = sbar*sqrt(3)
    kBT_init = m1 * (sbar**2) / 2.0
    sigma_diam = 2.0 * r1
    Tlabel = f"T_{t_idx}"

    # ------------------------------
    # MAIN SIMULATION LOOP
    # ------------------------------

    for run in range(n_runs):




        print(f"[{Tlabel}] Running simulation {run+1}/{n_runs}")
        rng = np.random.default_rng(base_seed + 1000*t_idx + run)

        # Random initial positions and velocities
        rng = np.random.default_rng(base_seed + 1000*t_idx + run)
        pos = rng.random((n, 2))
        theta = rng.random(n) * 2*np.pi
        s0 = rng.random(n) * sbar * np.sqrt(3.0)

        vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T  # true speeds initially

        #the brute force temperature rescaling
        vel -= np.mean(vel, axis=0)
        kBT_now = m1 * np.mean(np.sum(vel**2, axis=1)) / 2.0
        scale = np.sqrt(kBT_init/ kBT_now)
        vel *= scale



        # Peculiar = true - affine flow; at t=0 gamma=0 so same as true
        sim = MDSimulation(pos, vel, r1, m1, eps_strainrate, T_period_steps, FPS, delt)



        # run_velocities, run_positions, run_sigmas = [], [], []



        # --- burn-in over one full square-wave period (no I/O) ---
        # note: you are burning in with the shear turned on, why not I guess
        for i in range(T_period_steps):
            sigma, sigma_kin, sigma_col, e_dot, sgn = sim.advance(dt)




        # --- reset global accumulators so reports exclude burn-in ---
        sim.sum_sigma[:] = 0.0
        sim.sum_sgn_sigma[:] = 0.0
        sim.sum_sgn_sigma_col[:]   = 0.0   # ← add
        sim.sum_sgn_sigma_kin[:]   = 0.0   # ← add
        sim.sum_sgn_edot = 0.0
        sim.n_accum = 0
        sim.ncounter =0 
        sim.ncounternet =0
        #gamma and nsteps are the two things that build during the period. with nsteps corresponding to the sign and gamma being the strain, both must be reset so that 
        #you are really looking at a single period 
        sim.nsteps =0 
        sim.gamma = 0.0                   # accumulated shear strain




        # --- reset 100-step block accumulators ---
        acc_shear_sym = 0.0
        acc_odd_sym = 0.0
        acc_edot = 0.0
        acc_count = 0

        # the determination of the strain rate etc is all done in the advance thing
        for i in range(frames):
            sigma, sigma_kin, sigma_col, e_dot, sgn = sim.advance(dt)

            # accumulate over the current T_period_steps block
            shear_sym = 0.5 * (sigma[0,1] + sigma[1,0])
            odd_sym = 0.5 * (sigma[0,0] - sigma[1,1])
            acc_shear_sym += shear_sym*sgn
            acc_odd_sym += odd_sym*sgn
            acc_edot += e_dot*sgn
            # this averaging of e_dot is a bit silly but oh well it cant hurt
            acc_count += 1


            block_N = T_period_steps 
            # this is much smarter than how I did it before because it always skips the first
            if (i+1) % block_N == 0:
                # viscosity from this 100-step block:  <(σ_xy+σ_yx)/2> / <e_dot>
                # (no sign-weighting since your flip period is a multiple of 100)
                eta_block = acc_shear_sym / acc_edot
                eta_odd_block = acc_odd_sym / acc_edot

                start = i - block_N + 1 
                end = i + 1
                # this really prints the inermediate shear viscosity!!!
                print(f"  steps {start:>5d}-{end:<5d}: "
                    f"gamma={sim.gamma:.4e}, "
                    f"<e_dot>_blk={acc_edot/acc_count:.3e}, "
                    f"eta_100={eta_block:.3e}",
                    f"eta_o_100={eta_odd_block:.3e}"
                )
                now = datetime.now(); print(now.time())


                print("for this interval the temperature is given by",sim.mass * np.mean(np.sum(sim.vel**2, axis=1)) / 2.0)

                # reset accumulators for the next block
                acc_shear_sym = 0.0
                acc_odd_sym  =0 
                acc_edot = 0.0
                acc_count = 0



        print("the number of collisions is given by",sim.ncounter)
        print("the net number of chiral collisions is given by",sim.ncounternet)



        # ---- Reports (per run)
        avg_sgn_sigma = sim.sum_sgn_sigma / sim.n_accum
        avg_sgn_sigma_col = sim.sum_sgn_sigma_col / sim.n_accum
        avg_sgn_sigma_kin = sim.sum_sgn_sigma_kin / sim.n_accum
        avg_sigma = sim.sum_sigma / sim.n_accum
        avg_sgn_edot = sim.sum_sgn_edot / sim.n_accum



        shear_sym = 0.5*(avg_sgn_sigma[0,1] + avg_sgn_sigma[1,0])  # even viscosity numerator
        shear_sym_col = 0.5*(avg_sgn_sigma_col[0,1] + avg_sgn_sigma_col[1,0])  # even viscosity numerator
        shear_sym_kin = 0.5*(avg_sgn_sigma_kin[0,1] + avg_sgn_sigma_kin[1,0])  # even viscosity numerator


        R = shear_sym_kin / (shear_sym_col + 1e-30)
        print("shear_kin/shear_col =", R)


        odd_sym   = 0.5*(avg_sgn_sigma[0,0] - avg_sgn_sigma[1,1])  # (σ_xx - σ_yy)/2
        anti_sym   = 0.5*(avg_sgn_sigma[1,0] - avg_sgn_sigma[0,1])  # (σ_xy - σ_yx)/2


        eta_est          = shear_sym / avg_sgn_edot          # η
        eta_est_col     = shear_sym_col / avg_sgn_edot          # η
        eta_est_kin      = shear_sym_kin / avg_sgn_edot          # η

        odd_eta_est = odd_sym   / avg_sgn_edot          # η_o
        rot_eta_est = anti_sym   / avg_sgn_edot          # η_o


        print(f"Estimated eta   ~ {eta_est:.6e}")
        print(f"Estimated eta_kin   ~ {eta_est_kin:.6e}")
        print(f"Estimated eta_col   ~ {eta_est_col:.6e}")

        print(f"Estimated eta_o ~ {odd_eta_est:.6e}")
        print(f"Estimated eta_R ~ {rot_eta_est:.6e}")


        # Temperatures
        kbT_final_from_sim = temp_from_vel(sim.vel, m1)           # *peculiar* velocities
        print(f"Initial T (from sbar):            {kBT_init:.6e}")
        print(f"Final T (from sim.vel, peculiar): {kbT_final_from_sim:.6e}")
        print(f"eta divided by square root of temperature   ~ {eta_est/np.sqrt(kbT_final_from_sim):.6e}")


        # Optional: Chapman–Enskog hard-disk (dilute) estimate
        def eta_ce_hard_disks(m, kBT, sigma,delt):
            return (8.0 / (sigma)) * np.sqrt(m * kBT / np.pi)/(16+ delt**2/sigma**2)

        def eta_ce_odd_hard_disks(m, kBT, sigma,delt):
            return -(delt/sigma)*(2.0 / (sigma)) * np.sqrt(m * kBT / np.pi)/(16+ delt**2/sigma**2)


        eta_ce_init = eta_ce_hard_disks(m1, kBT_init, sigma_diam,sim.delt)
        eta_ce_odd_init = eta_ce_odd_hard_disks(m1, kBT_init, sigma_diam,sim.delt)
        print(f"Chapman–Enskog eta0 using initial T: {eta_ce_init:.6e}  [kBT_init={kBT_init:.6e}, sigma={sigma_diam:.3e}]")
        print(f"Chapman–Enskog etao0 using initial T: {eta_ce_odd_init:.6e}  [kBT_init={kBT_init:.6e}, sigma={sigma_diam:.3e}]")
        print("the ratio of numerics to analytics is",eta_est/eta_ce_init)
        print("the ratio of numerics to analytics for odd is",odd_eta_est/eta_ce_odd_init)


now = datetime.now(); print(now.time())


