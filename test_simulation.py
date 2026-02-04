
import unittest
import numpy as np
from sim_controlDigitalARX import run_simulation

class TestDigitalControl(unittest.TestCase):
    def test_simulation_math(self):
        # Run the simulation
        t, y, u, Ref = run_simulation()

        # Parameters (must match the script)
        Ts = 0.004
        Kp = -11.6083
        Ti = 0.1014

        # Calculate expected coefficients
        K0_expected = Kp + Kp * Ts / (2 * Ti)
        K1_expected = -Kp + Kp * Ts / (2 * Ti)

        # Check Step 0
        # y[0] should be 0 (initial state zero)
        self.assertAlmostEqual(y[0], 0.0, places=5)

        # Error[0] = 5 - 0 = 5
        error0 = 5.0
        # u[0] = 0 + K0 * error0 + K1 * 0
        u0_expected = K0_expected * error0

        # Check saturation logic manually
        Umax = 100.0
        Umin = -100.0
        if u0_expected > Umax: u0_expected = Umax
        elif u0_expected < Umin: u0_expected = Umin

        self.assertAlmostEqual(u[0], u0_expected, places=5)

        # Check Step 1
        # y[1] depends on previous y and u(k-4).
        # y[1] = 1.9366*y[0] ... - 0.001961*u[-3]. All zero.
        self.assertAlmostEqual(y[1], 0.0, places=5)

        error1 = 5.0 # Since y[1]=0
        # u[1] = u[0] + K0*error1 + K1*error0
        u1_expected = u[0] + K0_expected * error1 + K1_expected * error0

        if u1_expected > Umax: u1_expected = Umax
        elif u1_expected < Umin: u1_expected = Umin

        self.assertAlmostEqual(u[1], u1_expected, places=5)

        # Check Step 4 (First time u affects y)
        # y[k] = ... + b[3]*u[k-4]
        # y[4] = 1.9366*y[3] - ... + (-0.001961)*u[0]
        # Since y[0]..y[3] are 0, y[4] should be just b[3]*u[0]

        b3 = -0.001961
        y4_expected = b3 * u[0]
        self.assertAlmostEqual(y[4], y4_expected, places=5)

        print("Verification passed: Steps 0, 1, and 4 match theoretical calculations.")

if __name__ == '__main__':
    unittest.main()
