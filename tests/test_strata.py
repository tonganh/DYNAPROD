import numpy as np

from dynaprod.core.strata import family_group, family_size


def test_strata_fixed_in_advance(make_entries):
    st = family_size(30).fit(make_entries([("a", "small", "*", 40), ("a", "large", "*", 10),
                                       ("b", "small", "*", 35), ("b", "large", "*", 35),
                                       ("c", "small", "*", 20)], np.random.default_rng(0)))
    assert st.key(dict(family="a", size="small")) == ("a", "*")
    assert st.key(dict(family="a", size="large")) == ("a", "*")
    assert st.key(dict(family="b", size="large")) == ("b", "large")
    assert st.key(dict(family="c", size="small")) is None
    assert all(n >= 30 for n in st.valid.values())


def test_group_strata(make_entries):
    st = family_group(30).fit(make_entries([("m", "small", "S2", 30), ("m", "small", "L+S2", 31)],
                                       np.random.default_rng(1)))
    assert st.key(dict(family="m", size="small", group="S2")) == ("m", "S2", "small")
    assert st.valid[("m", "L+S2", "small")] == 31
    assert st.key(dict(family="m", size="small", group="S1")) is None


def test_group_strata_split_size_before_group(make_entries):
    rng = np.random.default_rng(5)
    st = family_group(30).fit(make_entries([("m", "small", "S2", 40), ("m", "small", "L+S2", 40),
                                        ("m", "xl", "S2", 15), ("m", "xl", "L+S2", 15)], rng))
    assert st.key(dict(family="m", size="xl", group="S2")) == ("m", "*", "xl")
    assert st.key(dict(family="m", size="small", group="S2")) == ("m", "S2", "small")
