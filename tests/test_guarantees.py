from dynaprod.core.guarantees import (coupling_bound, draw_error, hoeffding_halfwidth, min_calibration,
                                      online_miss_upper, sidak_level)


def test_guarantee_formulas():
    assert abs(100 * hoeffding_halfwidth(1024) - 4.24) < 0.01
    assert min_calibration(0.10) == 9 and min_calibration(0.05) == 19
    assert draw_error(200, 8) < draw_error(100, 8)
    assert abs(coupling_bound([0.1, 0.1]) - 0.19) < 1e-12
    assert abs(sidak_level(0.1, 1) - 0.1) < 1e-12 and sidak_level(0.1, 8) < 0.1 / 8 * 1.1
    assert online_miss_upper(0.1, 0.02, 10 ** 6) < 0.1001
