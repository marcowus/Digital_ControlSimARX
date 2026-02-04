"""
Advanced Prescribed-Time Adaptive Control for 3rd-Order Strict-Feedback System.
Version 2: Refined Implementation based on feedback (Patches 1-5).

Key Improvements in V2:
1. Scaling: mu_dot=0 when t >= t_stop (prevents end-of-time spikes).
2. Koopman: Lift dimension fixed to 18, predict_drift uses previous applied input.
3. Adaptive: Discrete projection with hard clamping (prevents parameter explosion).
4. Logging: Logs actual clipped/applied inputs (eliminates fake spikes).
5. Stability: Safe inverse deadzone calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional, Dict

##############################################################################
#                               System & Utilities
##############################################################################

class ThirdOrderSystem:
    def __init__(self, disturbance_scale=1.0, extra_disturbance=False):
        self.kb1_base, self.kb2_base, self.kb3_base = 0.5, 0.4, 0.6
        self.disturbance_scale = disturbance_scale
        self.extra_disturbance = extra_disturbance

    def f1(self, x1):
        return np.sin(x1**2) * np.cos(x1)

    def f2(self, x2):
        return np.sin(x2) * (x2 + x2**2)

    def d1(self, t):
        return self.disturbance_scale * 0.01 * np.sin(t)

    def d2(self, t):
        val = self.disturbance_scale * 0.02 * np.sin(t)
        if self.extra_disturbance and t > 2.0:
            val += 0.5 # Step disturbance
        return val

    def get_constraints(self, t):
        return (0.1*np.sin(t) + self.kb1_base,
                0.1*np.sin(t) + self.kb2_base,
                0.3*np.sin(t) + self.kb3_base)

    def deadzone(self, v):
        # Asymmetric Dead-zone
        if v >= 0.75: return 1.2 * v - 0.9
        elif v < -0.5: return 0.8 * v + 0.4
        else: return 0.0

    def dynamics(self, t, x, v_cmd):
        x1, x2, x3 = x
        u = self.deadzone(v_cmd)
        dx1 = x2 + self.f1(x1) + self.d1(t)
        dx2 = x3 + self.f2(x2) + self.d2(t)
        dx3 = u
        return np.array([dx1, dx2, dx3])

class CommandFilter:
    def __init__(self, omega_n=40.0, zeta=0.8, enabled=True):
        self.omega_n = omega_n
        self.zeta = zeta
        self.enabled = enabled
        self.x1 = 0.0
        self.x2 = 0.0

    def step(self, u, dt):
        if not self.enabled:
            self.x1 = u
            self.x2 = 0.0
            return u, 0.0

        dx1 = self.x2
        dx2 = -2*self.zeta*self.omega_n*self.x2 - self.omega_n**2 * (self.x1 - u)

        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        return self.x1, self.x2

# Patch 1: StrictPrescribedTimeScaling with consistent cutoff
class StrictPrescribedTimeScaling:
    """
    Scaling function: mu(t) = (T / (T - t))^p.
    Consistent handling of t >= t_stop: mu remains constant, mu_dot becomes 0.
    """
    def __init__(self, T, p=2.0, t_stop_ratio=0.99, enabled=True, mu_max=None):
        self.T = T
        self.p = p
        self.t_stop = T * t_stop_ratio
        self.enabled = enabled
        self.mu_max = mu_max

    def _mu_at(self, t_eff: float):
        denom = max(self.T - t_eff, 1e-6)
        mu = (self.T / denom)**self.p
        mu_dot = self.p * (self.T**self.p) / (denom**(self.p + 1))
        return mu, mu_dot

    def get_mu(self, t):
        if not self.enabled:
            return 1.0, 0.0

        if t >= self.t_stop:
            mu_stop, _ = self._mu_at(self.t_stop)
            if self.mu_max is not None:
                mu_stop = min(mu_stop, self.mu_max)
            return mu_stop, 0.0  # KEY FIX: mu constant => mu_dot=0

        mu, mu_dot = self._mu_at(t)
        if self.mu_max is not None:
            if mu >= self.mu_max:
                mu = self.mu_max
                mu_dot = 0.0
        return mu, mu_dot

class Projection:
    @staticmethod
    def project(theta, d_theta, theta_min, theta_max):
        if theta > theta_max and d_theta > 0:
            return 0.0
        if theta < theta_min and d_theta < 0:
            return 0.0
        return d_theta

##############################################################################
#                      Enhanced Koopman (V2)
##############################################################################

# Patch 2: Koopman with correct dimensions and input-aware prediction
class KoopmanEstimator:
    def __init__(self, n_obs=18, lambda_reg=1e-2):
        self.n_obs = n_obs
        self.lambda_reg = lambda_reg
        self.A_aug = None

    def lift(self, x):
        x1, x2, x3 = x
        base = [
            x1, x2, x3,
            x1**2, x2**2, x3**2,
            x1*x2, x2*x3, x1*x3,
            np.sin(x1), np.cos(x1),
            np.sin(x2), np.cos(x2),
            np.sin(x3), np.cos(x3),
            np.maximum(0, x1), np.maximum(0, -x1),
            x1**3, x2**3, x3**3,  # padding to ensure enough features
        ]
        # ensure exactly n_obs
        if len(base) < self.n_obs:
            base = base + [0.0]*(self.n_obs - len(base))
        return np.array(base[:self.n_obs], dtype=float)

    def fit(self, X, Y, U):
        Psi_X = np.array([self.lift(xi) for xi in X])
        Psi_Y = np.array([self.lift(yi) for yi in Y])
        U_col = U.reshape(-1, 1)
        Phi = np.hstack([Psi_X, U_col])
        reg = self.lambda_reg * np.eye(Phi.shape[1])
        self.A_aug = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Psi_Y).T

    def predict_drift(self, x, u_in, dt):
        """Predict dx/dt using learned Koopman; u_in is the applied input."""
        if self.A_aug is None:
            return np.zeros(3)
        psi = self.lift(x)
        phi = np.hstack([psi, [float(u_in)]])
        psi_next = self.A_aug @ phi
        x_next = psi_next[:3]
        return (x_next - x) / dt

##############################################################################
#                  Advanced Controller Implementation (V2)
##############################################################################

@dataclass
class ControllerConfig:
    T_final: float = 5.0
    p_scaling: float = 2.0
    scaling_enabled: bool = True
    mu_max: Optional[float] = 1e5 # Optional cap

    cf_omega: float = 40.0
    cf_enabled: bool = True

    iblf_enabled: bool = True

    adaptive_enabled: bool = True
    deadzone_inverse_enabled: bool = True

    koopman_enabled: bool = True

    k_gains: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])

class AdvancedController:
    def __init__(self, config: ControllerConfig, koopman: Optional[KoopmanEstimator] = None):
        self.cfg = config
        self.koopman = koopman

        self.scaling = StrictPrescribedTimeScaling(
            self.cfg.T_final,
            p=self.cfg.p_scaling,
            t_stop_ratio=0.95, # Consistently use 0.95 to avoid extreme edge
            enabled=self.cfg.scaling_enabled,
            mu_max=self.cfg.mu_max
        )

        self.cf1 = CommandFilter(omega_n=self.cfg.cf_omega, enabled=self.cfg.cf_enabled)
        self.cf2 = CommandFilter(omega_n=self.cfg.cf_omega, enabled=self.cfg.cf_enabled)

        # Adaptive Parameters
        self.m_hat = 1.0
        self.b_hat = 0.0

        # Split logic removed for V2 simplicity (sticking to prompt's single estimator correction)
        # Re-reading prompt Patch 3: "self.m_hat += ...". It uses single parameter set.
        # I will stick to single parameter set as per specific Patch 3 instructions
        # to ensure apple-to-apple comparison on the "Fixes" requested.
        # (Though split estimation is better, Patch 3 specifically provided single param logic).

        self.last_alpha1 = 0.0
        self.last_alpha2 = 0.0
        self.first_step = True

        # Patch 4: Store last applied v
        self.last_v_applied = 0.0

    def iblf(self, x, k):
        if not self.cfg.iblf_enabled:
            return 0.0
        denom = (k - 0.01)**2 - x**2
        if denom < 1e-4: denom = 1e-4
        return x / denom

    def compute_control(self, system, x, t, dt):
        x1, x2, x3 = x
        mu, mu_dot = self.scaling.get_mu(t)
        mu_safe = max(mu, 1e-3)

        kb1, kb2, kb3 = system.get_constraints(t)

        yd = 0.3 * np.sin(t)
        yd_dot = 0.3 * np.cos(t)

        # --- Step 1 ---
        z1 = x1 - yd
        xi1 = mu * z1
        bar1 = self.iblf(x1, kb1)

        # Patch 4: Use last_v_applied for Koopman
        f_est = np.zeros(3)
        if self.cfg.koopman_enabled and self.koopman:
            f_est = self.koopman.predict_drift(x, self.last_v_applied, dt)

        f1_est = f_est[0] - x2

        alpha1_raw = (-self.cfg.k_gains[0]*xi1 - bar1)/mu_safe - (mu_dot/mu_safe)*z1 - f1_est + yd_dot
        if self.cfg.cf_enabled:
            alpha1, d_alpha1 = self.cf1.step(alpha1_raw, dt)
        else:
            alpha1 = alpha1_raw
            if self.first_step: d_alpha1 = 0.0
            else: d_alpha1 = (alpha1 - self.last_alpha1) / dt
            self.last_alpha1 = alpha1

        # --- Step 2 ---
        z2 = x2 - alpha1
        xi2 = mu * z2
        bar2 = self.iblf(x2, kb2)
        f2_est = f_est[1] - x3

        alpha2_raw = (-self.cfg.k_gains[1]*xi2 - bar2)/mu_safe - (mu_dot/mu_safe)*z2 - f2_est + d_alpha1
        if self.cfg.cf_enabled:
            alpha2, d_alpha2 = self.cf2.step(alpha2_raw, dt)
        else:
            alpha2 = alpha2_raw
            if self.first_step: d_alpha2 = 0.0
            else: d_alpha2 = (alpha2 - self.last_alpha2) / dt
            self.last_alpha2 = alpha2

        # --- Step 3 ---
        z3 = x3 - alpha2
        xi3 = mu * z3
        bar3 = self.iblf(x3, kb3)

        total_des = -self.cfg.k_gains[2]*xi3 - bar3 - mu_dot*z3 + mu*d_alpha2
        u_des = total_des / mu_safe

        # Patch 4: Safe inverse deadzone
        if self.cfg.deadzone_inverse_enabled:
            m_safe = max(float(self.m_hat), 0.5)
            v_cmd = (u_des - float(self.b_hat)) / m_safe
        else:
            v_cmd = u_des

        self.first_step = False

        debug = {
            't': t,
            'mu': mu, 'mu_dot': mu_dot,
            'z': (z1, z2, z3),
            'xi': (xi1, xi2, xi3),
            'u_des': u_des,
            # Placeholder, will be overwritten by run_simulation with APPLIED values
            'v_cmd': v_cmd,
            'u_act': 0.0,
            'm_hat': self.m_hat,
            'b_hat': self.b_hat
        }

        return v_cmd, debug

    # Patch 3: Update adaptation with hard clamping
    def update_adaptation(self, debug, v_applied, dt):
        if not self.cfg.adaptive_enabled:
            return

        gamma = 2.0
        xi3 = float(debug['xi'][2])

        dm = gamma * xi3 * float(v_applied)
        db = gamma * xi3 * 1.0

        dm_proj = Projection.project(self.m_hat, dm, 0.5, 2.0)
        db_proj = Projection.project(self.b_hat, db, -1.5, 1.5)

        self.m_hat += dm_proj * dt
        self.b_hat += db_proj * dt

        # KEY FIX: hard clamp
        self.m_hat = float(np.clip(self.m_hat, 0.5, 2.0))
        self.b_hat = float(np.clip(self.b_hat, -1.5, 1.5))


##############################################################################
#                                Simulation
##############################################################################

class DataGenerator:
    def __init__(self, system, dt=0.01):
        self.system = system
        self.dt = dt

    def generate(self, num_traj=50, steps=300):
        X, Y, U = [], [], []
        for _ in range(num_traj):
            x = np.random.uniform(-0.4, 0.4, 3)
            phase = np.random.uniform(0, 2*np.pi)
            valid = True
            traj_X, traj_Y, traj_U = [], [], []

            for step in range(steps):
                t = step * self.dt
                v = 2.0*np.sin(0.5*t+phase) + 1.5*np.sin(1.2*t) + np.random.randn()
                v = float(np.clip(v, -8, 8))
                dx = self.system.dynamics(t, x, v)
                x_next = x + dx * self.dt

                if np.any(np.abs(x_next) > 3.0):
                    valid = False
                    break
                traj_X.append(x.copy())
                traj_Y.append(x_next.copy())
                traj_U.append(v)
                x = x_next

            if valid:
                X.extend(traj_X)
                Y.extend(traj_Y)
                U.extend(traj_U)

        return np.array(X), np.array(Y), np.array(U)

def run_simulation(config: ControllerConfig, sys_args: Dict = None, x0: np.ndarray = None):
    if sys_args is None: sys_args = {}
    dt = 0.002
    times = np.arange(0, config.T_final, dt)

    sys = ThirdOrderSystem(**sys_args)

    est = None
    if config.koopman_enabled:
        gen_sys = ThirdOrderSystem()
        # Matched dt for correct drift scale
        gen = DataGenerator(gen_sys, dt=dt)
        X, Y, U = gen.generate(num_traj=30, steps=200)
        est = KoopmanEstimator()
        est.fit(X, Y, U)

    ctrl = AdvancedController(config, koopman=est)

    if x0 is None:
        x = np.array([0.1, 0.1, 0.1])
    else:
        x = np.array(x0)

    logs = {k: [] for k in ['t', 'x', 'mu', 'xi', 'z', 'u_des', 'v_cmd', 'u_act', 'm_hat', 'b_hat']}

    # Patch 5: Simulation Loop with correct logging
    for t in times:
        v_raw, debug = ctrl.compute_control(sys, x, t, dt)

        # applied command (what plant actually sees)
        v_app = float(np.clip(v_raw, -20, 20))
        u_app = float(sys.deadzone(v_app))

        # overwrite debug so plots reflect applied values, not raw
        debug['v_cmd'] = v_app
        debug['u_act'] = u_app

        # update adaptation using APPLIED v (not raw)
        ctrl.update_adaptation(debug, v_app, dt)

        # simulate plant with APPLIED v
        dx = sys.dynamics(t, x, v_app)
        x = x + dx * dt

        # update controller memory for next-step Koopman drift
        ctrl.last_v_applied = v_app

        # Store Data
        logs['t'].append(t)
        logs['x'].append(x.copy())
        logs['mu'].append(debug['mu'])
        logs['xi'].append(debug['xi'])
        logs['z'].append(debug['z'])
        logs['u_des'].append(debug['u_des'])
        logs['v_cmd'].append(v_app)
        logs['u_act'].append(u_app)
        logs['m_hat'].append(ctrl.m_hat)
        logs['b_hat'].append(ctrl.b_hat)

    for k in logs:
        logs[k] = np.array(logs[k])

    return logs
