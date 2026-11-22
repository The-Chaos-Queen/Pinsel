import tempfile
import unittest
from pathlib import Path

import numpy as np

import pinsel as ps
from pinsel import pigments as pg


class CrayonTests(unittest.TestCase):
    def canvas(self):
        return ps.Canvas(112, 64, ground='rough', seed=7)

    def test_paper_contact_and_pressure(self):
        light, heavy = self.canvas(), self.canvas()
        path = [(10, 32), (102, 32)]
        stick = ps.OilCrayon(18, side=0.8).load(pg.ULTRAMARINE)
        for cv, pressure in [(light, 0.24), (heavy, 0.95)]:
            cv.stroke(stick, path, pressure=pressure)
        interior = np.s_[28:36, 16:96]
        peaks = light.tooth[interior] > 0.25
        pits = light.tooth[interior] < -0.25
        v = light.film_vol[interior]
        self.assertGreater(v[peaks].mean(), v[pits].mean())
        self.assertGreater(heavy.film_vol[interior].mean(), v.mean())
        self.assertLess((heavy.film_vol[interior] < 0.02).sum(), (v < 0.02).sum())

    def test_side_angle_masks_and_pressure_path(self):
        stick = ps.OilCrayon(12, side=1).load(pg.CADMIUM_RED)
        cv = self.canvas()
        horizontal = cv.stroke(stick, [(30, 22)], pressure=1, angle=0)
        vertical = cv.stroke(stick, [(80, 22)], pressure=1, angle=90)
        self.assertGreater(horizontal[2] - horizontal[0], 3*(horizontal[3] - horizontal[1]))
        self.assertGreater(vertical[3] - vertical[1], 3*(vertical[2] - vertical[0]))
        cv.mask_rect(50, 40, 65, 64)
        cv.stroke(stick, [(-10, 54, 0.0), (120, 54, 1.0)], angle=0)
        self.assertFalse(cv.film_vol[40:, 50:65].any())
        self.assertGreater(cv.film_vol[50:, 85:100].sum(), cv.film_vol[50:, :15].sum())
        self.assertIsNone(cv.stroke(stick, [(55, 48)], pressure=0))

    def test_checkpoint_continues_layering(self):
        cv = self.canvas()
        path = [(12, 32), (100, 32)]
        stick = ps.OilCrayon(14).load(pg.ULTRAMARINE)
        cv.stroke(stick, path)
        resumed = self.canvas()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'study.npz'
            cv.save_state(checkpoint)
            resumed.load_state(checkpoint)
        legacy = self.canvas()
        old_state = cv.get_state()
        del old_state['_strokes']
        legacy.set_state(old_state)
        stick.load(pg.YELLOW_OCHRE)
        for target in (cv, resumed, legacy):
            target.stroke(stick, path)
        for target in (resumed, legacy):
            self.assertEqual(target.strokes, 2)
            np.testing.assert_array_equal(target.base_K, cv.base_K)
            np.testing.assert_array_equal(target.film_K, cv.film_K)
            self.assertGreater(target.base_vol.sum(), 0)


if __name__ == '__main__':
    unittest.main()
