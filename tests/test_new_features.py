import unittest
import warnings

import numpy as np

import pinsel as ps
from pinsel import path as P, pigments as pg


class NewFeaturesTests(unittest.TestCase):
    def test_linen_ground_has_no_divide_by_zero_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cv = ps.Canvas(128, 96, ground='linen', seed=42)
        div_warns = [w for w in caught if 'divide by zero' in str(w.message).lower()]
        self.assertEqual(div_warns, [])
        self.assertTrue(np.isfinite(cv.h_ground).all())

    def test_region_rendering_matches_full_render(self):
        cv = ps.Canvas(100, 80, ground='canvas', seed=5)
        b = ps.Brush('filbert', size=10, seed=3).load(pg.ULTRAMARINE)
        cv.stroke(b, P.line((15, 25), (85, 55)), pressure=0.8)
        reg = (16, 64, 20, 76)
        full_lin = cv.render_linear()
        sub_lin = cv.render_linear(region=reg)
        np.testing.assert_allclose(sub_lin, full_lin[16:64, 20:76], atol=1e-6)

        full_rgb = cv.render(lighting=True)
        sub_rgb = cv.render(lighting=True, region=reg)
        np.testing.assert_allclose(sub_rgb, full_rgb[16:64, 20:76], atol=1e-3)

    def test_checkpoint_undo_and_restore(self):
        cv = ps.Canvas(64, 48, ground='smooth', seed=1)
        b = ps.Brush('round', size=6, seed=2).load(pg.CADMIUM_RED)
        cv.checkpoint('blank')
        cv.stroke(b, P.line((10, 24), (54, 24)), pressure=0.8)
        cv.checkpoint('red_line')
        b.load(pg.ULTRAMARINE)
        cv.stroke(b, P.line((32, 6), (32, 42)), pressure=0.8)
        self.assertEqual(cv.checkpoints, ['blank', 'red_line'])
        self.assertEqual(cv.strokes, 2)

        lbl = cv.undo()
        self.assertEqual(lbl, 'red_line')
        self.assertEqual(cv.strokes, 1)
        self.assertEqual(cv.checkpoints, ['blank'])

        cv.restore('blank')
        self.assertEqual(cv.strokes, 0)
        self.assertFalse(cv.film_vol.any())

    def test_varnish_unifies_gloss_and_respects_mask(self):
        cv = ps.Canvas(64, 48, ground='canvas', seed=9)
        b = ps.Brush('flat', size=10, seed=4).load(pg.PRUSSIAN_BLUE)
        cv.stroke(b, P.line((8, 24), (56, 24)), pressure=0.9)
        cv.dry()
        before_R = cv.below_R.copy()
        cv.mask_rect(0, 0, 32, 48)
        cv.varnish(gloss=0.85, depth=0.15)
        # Masked left half unchanged; unmasked right half unified & deepened
        np.testing.assert_allclose(cv.below_R[:, :32], before_R[:, :32], atol=1e-6)
        self.assertAlmostEqual(float(cv.gloss_dry[24, 48]), 0.85, places=5)
        self.assertLess(float(cv.below_R[:, 32:].mean()), float(before_R[:, 32:].mean()))

    def test_paint_over_glaze_preview(self):
        glaze = pg.CADMIUM_ORANGE.thinned(0.7)
        over_white = glaze.over(pg.TITANIUM_WHITE, thickness=0.4)
        over_dark = glaze.over(pg.PRUSSIAN_BLUE, thickness=0.4)
        self.assertEqual(over_white.shape, (3,))
        self.assertGreater(float(over_white.mean()), float(over_dark.mean()))
        hex_str = glaze.over_hex('#ffffff', thickness=0.4)
        self.assertTrue(hex_str.startswith('#') and len(hex_str) == 7)

    def test_spectral_monotype_and_pull_onto_canvas(self):
        plate = ps.MonotypePlate(48, 36, seed=11)
        mask = np.zeros((36, 48), dtype=np.float32)
        mask[8:28, 6:18] = 1.0  # left side of plate -> right side when reversed!
        plate.ink(mask, pg.ULTRAMARINE, load=0.9)
        proof = plate.proof()
        self.assertEqual(proof.size, (48, 36))

        cv = ps.Canvas(64, 48, ground='cold_press', seed=3)
        cv.mask_rect(0, 0, 20, 48)
        plate.pull_onto(cv, offset=(8, 6), bake=False)
        # Reversed mark lands on right side of plate (cols 30..42 -> canvas x 38..50)
        self.assertGreater(float(cv.film_vol[14:34, 38:50].sum()), 0.0)
        self.assertFalse(cv.film_vol[:, :20].any())
        self.assertEqual(cv.strokes, 1)

        # Ghost pull still transfers residual ink
        ghost = plate.pull()
        self.assertEqual(ghost.size, (48, 36))

    def test_charcoal_eraser_and_fixative(self):
        cv = ps.Canvas(96, 64, ground='rough', seed=7)
        ch = ps.Charcoal(size=10, seed=5)
        cv.stroke(ch, P.line((12, 32), (84, 32)), pressure=0.85)
        vol_initial = float(cv.film_vol.sum())
        self.assertGreater(vol_initial, 0.0)

        # Kneaded eraser lifts unfixed charcoal
        er = ps.KneadedEraser(size=14, strength=0.8)
        er.stroke(cv, P.line((12, 32), (48, 32)), pressure=0.9)
        vol_after_erase = float(cv.film_vol.sum())
        self.assertLess(vol_after_erase, vol_initial * 0.85)

        # Fixative locks remaining charcoal into below_R so eraser can't lift it
        cv.fix(1.0)
        self.assertFalse(cv.film_vol.any())
        before_fixed = cv.below_R.copy()
        er.stroke(cv, P.line((48, 32), (84, 32)), pressure=1.0)
        np.testing.assert_array_equal(cv.below_R, before_fixed)


if __name__ == '__main__':
    unittest.main()
