
import unittest
import numpy as np
from sim_advanced_control import ThirdOrderSystem, StrictPrescribedTimeScaling

class TestAdvancedComponents(unittest.TestCase):
    def test_deadzone(self):
        sys = ThirdOrderSystem()
        # Test Right side (v >= 0.75)
        # u = 1.2*v - 0.9
        self.assertAlmostEqual(sys.deadzone(1.0), 1.2*1.0 - 0.9)

        # Test Left side (v < -0.5)
        # u = 0.8*v + 0.4
        self.assertAlmostEqual(sys.deadzone(-1.0), 0.8*(-1.0) + 0.4)

        # Test Dead zone (-0.5 <= v < 0.75)
        self.assertEqual(sys.deadzone(0.0), 0.0)
        self.assertEqual(sys.deadzone(0.5), 0.0)
        self.assertEqual(sys.deadzone(-0.4), 0.0)

    def test_scaling_function(self):
        T = 5.0
        p = 2.0
        scaling = StrictPrescribedTimeScaling(T, p)

        # At t=0, mu = (T/T)^p = 1
        mu, mu_dot = scaling.get_mu(0.0)
        self.assertAlmostEqual(mu, 1.0)

        # mu_dot = p * T^p / (T-t)^(p+1)
        # At t=0, mu_dot = p / T
        self.assertAlmostEqual(mu_dot, p/T)

        # At t -> T, mu should be large
        mu_near, _ = scaling.get_mu(4.9)
        self.assertTrue(mu_near > 1.0)

if __name__ == '__main__':
    unittest.main()
