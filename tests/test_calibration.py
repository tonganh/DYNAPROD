import numpy as np

from dynaprod.core.calibration import Calibrator, CQRCalibrator, OnlineLevel, coherent, floor_grid, j_upper


def test_lemma1_coverage_validity():
    rng = np.random.default_rng(4)
    cal = Calibrator(min_stratum=1)
    miss, alpha = [], 0.1
    for rep in range(30):
        def draw():
            y = float(np.clip(rng.beta(5, 2), 0, 1))
            samples = np.clip(rng.normal(0.8, 0.05, 200), 0, 1)
            return y, samples
        entries = []
        for _ in range(60):
            y, s = draw()
            entries.append(dict(family="f", size="s", Y=s, C=np.ones(10), T=np.ones(10), y=y, c=1.0, t=1.0))
        cal.fit(entries, rng)
        for _ in range(100):
            y, s = draw()
            b = cal.bounds("f", "s", s[None], np.ones((1, 10)), np.ones((1, 10)))
            miss.append(y < b.ell[0])
    assert np.mean(miss) <= alpha + 0.02


def test_coherence_lowers_to_containing_products():
    ell = np.array([0.9, 0.7, 0.8])
    R = np.zeros((3, 3), bool)
    R[0, 1] = True
    out = coherent(ell, R)
    assert out[0] == 0.7 and out[1] == 0.7 and out[2] == 0.8


def test_online_step_tracks_level():
    rng = np.random.default_rng(5)
    ol = OnlineLevel(alpha=0.1, gamma=0.05)
    errs = []
    for t in range(4000):
        a = ol.level()
        p_miss = np.clip(a * (0.3 if t < 2000 else 3.0), 0, 1)
        e = int(rng.random() < p_miss)
        errs.append(e)
        ol.update(e)
    assert abs(np.mean(errs[-1500:]) - 0.1) < 0.03


def test_small_strata_give_zero_bounds(make_entries):
    rng = np.random.default_rng(2)
    ent = make_entries([(f, "small", "*", 15) for f in ("a", "b", "c")], rng)
    cal = Calibrator().fit(ent, rng)
    Y = rng.beta(5, 2, (3, 50))
    b = cal.bounds("a", "small", Y, Y, Y)
    assert np.all(b.ell == 0) and np.all(np.isinf(b.UC)) and np.all(np.isinf(b.UT))
    ent = make_entries([("a", "small", "*", 60)], rng)
    b = Calibrator().fit(ent, rng).bounds("a", "small", Y, Y, Y)
    assert np.all(np.isfinite(b.UC)) and np.all(np.isfinite(b.UT)) and b.ell.max() > 0


def test_online_level_edges(make_entries):
    rng = np.random.default_rng(3)
    cal = Calibrator().fit(make_entries([("a", "small", "*", 40)], rng), rng)
    Y = rng.beta(5, 2, (2, 50))
    assert np.all(cal.bounds("a", "small", Y, Y, Y, alpha_Y=-0.01).ell == 0)
    assert np.all(cal.bounds("a", "small", Y, Y, Y, alpha_Y=1.2).ell == cal.grid[-1])


def test_cqr_levels_risk_and_joint(make_entries):
    rng = np.random.default_rng(6)
    ent = make_entries([("a", "small", "*", 80)], rng)
    cal = CQRCalibrator().fit(ent, rng)
    Y = rng.beta(5, 2, (3, 50))
    C = T = rng.lognormal(0, 0.2, (3, 50))
    S = cal.cqr[cal.stratum("a", "small")]
    sc = np.sort(np.quantile(S["Ys"], 0.10, axis=1) - S["y"])
    assert cal.bounds("a", "small", Y, C, T).ell[0] == floor_grid(np.quantile(Y[0], 0.10) - sc[j_upper(0.10, 80) - 1], cal.grid)
    ells = [cal.bounds("a", "small", Y, C, T, alpha_Y=a).ell.mean() for a in (0.05, 0.3)]
    assert ells[0] <= ells[1] and ells[0] < ells[1] + 1 and cal.bounds("a", "small", Y, C, T, alpha_Y=0).ell.max() == 0
    b = cal.bounds("a", "small", Y, C, T)
    assert (b.risk < 1).any() and np.all(np.diff(b.risk, axis=1) >= 0)
    j = cal.joint_bounds("a", "small", Y, C, T, 0.20)
    assert j.ell.shape == (3,) and np.isfinite(j.UC).all()
