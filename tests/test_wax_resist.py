import tempfile
import unittest
from pathlib import Path

import numpy as np

import pinsel as ps
from pinsel import pigments as pg, watercolor as wc


class WaxResistTests(unittest.TestCase):
    def canvas(self):
        return ps.Canvas(88, 58, ground='cold_press', seed=17)

    def test_wash_enters_gaps_and_honors_mask(self):
        cv = self.canvas()
        stick = ps.OilCrayon(size=14).load(pg.TITANIUM_WHITE)
        cv.stroke(stick, [(4, 30), (83, 30)], pressure=[0.18, 1.0])
        cv.dry()
        cv.mask_rect(0, 0, 14, 58, value=0.5)
        control = self.canvas()
        control.set_mask(cv.mask.copy())
        for sheet in (cv, control):
            wc.wash(sheet, None, pg.ULTRAMARINE, water=0.6, variation=0, rough=0, seed=2)
        sealed = cv.wax == 1
        partial = (cv.wax > 0.1) & (cv.wax < 0.8)
        self.assertTrue(sealed.any() and partial.any())
        np.testing.assert_allclose(cv.wc.water, control.wc.water * (1 - cv.wax), atol=1e-7)
        np.testing.assert_allclose(cv.wc.m, control.wc.m * (1 - cv.wax), atol=1e-7)
        cv.wait(5)
        for field in (cv.wc.water, cv.wc.damp, cv.wc.m, cv.wc.dm):
            self.assertFalse(field[sealed].any())
        self.assertGreater(cv.wc.m[~sealed].sum() + cv.wc.dm[~sealed].sum(), 0)

    def test_fluid_ink_and_oil_cross_the_same_resist(self):
        cv = self.canvas()
        cv.stroke(ps.OilCrayon(16).load(pg.TITANIUM_WHITE),
                  [(43, 5), (43, 53)], pressure=1)
        cv.dry()
        sealed = cv.wax == 1
        path = [(-6, 29), (92, 29)]
        pen = ps.tools.Pen(width=4, feather=0.6, seed=13)
        pen.load(pg.ULTRAMARINE.as_medium('ink')).line(cv, path)
        self.assertTrue(cv.wc.dm.any())
        self.assertFalse(cv.wc.dm[sealed].any())
        self.assertFalse(cv.wc.m[sealed].any())
        pen.load(pg.BURNT_SIENNA).line(cv, path)
        self.assertGreater(cv.film_vol[sealed].sum(), 0)

    def test_checkpoint_and_tape_leave_wax_intact(self):
        cv = self.canvas()
        stick = ps.OilCrayon(18).load(pg.YELLOW_OCHRE)
        cv.mask_rect(0, 0, 28, 58)
        cv.stroke(stick, [(6, 30), (83, 30)], pressure=0.95)
        cv.clear_mask()
        self.assertFalse(cv.wax[:, :28].any())
        self.assertEqual(cv.wax.max(), 1)
        wc.drop(cv, 50, 22, pg.ULTRAMARINE, seed=4)
        resumed = self.canvas()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wet_study.npz'
            cv.save_state(path)
            resumed.load_state(path)
        np.testing.assert_array_equal(cv.wax, resumed.wax)
        self.assertEqual(cv.wc.bbox, resumed.wc.bbox)
        for sheet in (cv, resumed):
            sheet.wait(2)
        np.testing.assert_array_equal(cv.wc.water, resumed.wc.water)
        np.testing.assert_array_equal(cv.wc.m, resumed.wc.m)
        resumed.set_state(self.canvas().get_state())
        self.assertIsNone(resumed.wax)
        self.assertIs(resumed.watercolor_mask(), resumed.mask)


if __name__ == '__main__':
    unittest.main()
