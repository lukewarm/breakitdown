"""Synthetic experiment data and an independent pandas/scipy reference for the Spark tests."""

import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as st

START = dt.date(2024, 3, 1)
DAYS = 5
END = START + dt.timedelta(days=DAYS - 1)
UTC = dt.timezone.utc
FLIGHTS = ["control", "t1", "t2"]

EVENT_SCHEMA = (
    "ruid string, flight string, os string, country string, event_time timestamp, "
    "media_type string, clicks int, session_start int, latency_ms int"
)


def row(frame: pd.DataFrame, **filters) -> pd.Series:
    """The single row of ``frame`` matching all ``column=value`` filters."""
    mask = np.ones(len(frame), dtype=bool)
    for column, value in filters.items():
        mask &= (frame[column] == value).to_numpy()
    selected = frame[mask]
    assert len(selected) == 1, f"{len(selected)} rows match {filters}"
    return selected.iloc[0]


def check(actual: pd.Series, expected: Dict[str, float], rel: float = 1e-6) -> None:
    for column, value in expected.items():
        assert actual[column] == pytest_approx(value, rel), f"{column}: {actual[column]} != {value}"


def pytest_approx(value, rel):
    import pytest

    return pytest.approx(value, rel=rel, abs=1e-12)


def _at(day: int, seconds: int) -> dt.datetime:
    return dt.datetime.combine(START + dt.timedelta(days=day), dt.time(), tzinfo=UTC) + dt.timedelta(seconds=seconds)


def event_records(seed: int = 11, units: int = 2400) -> List[Dict]:
    """Event-level rows of a pre-joined table: every row carries the flight and the user segments.

    t1 gets 1.5x clicks for iOS units only (a heterogeneous effect); t2 lowers latency by 4 ms.
    Every unit is active on its first day, so the first row day is its exposure day.
    """
    rng = np.random.default_rng(seed)
    flight = rng.choice(FLIGHTS, size=units)
    os_ = rng.choice(np.array(["ios", "android", None], dtype=object), size=units, p=[0.45, 0.45, 0.10])
    country = rng.choice(["US", "GB"], size=units, p=[0.6, 0.4])
    first = rng.integers(0, DAYS, size=units)
    records = []
    for u in range(units):
        rate = 1.5 if (flight[u] == "t1" and os_[u] == "ios") else 1.0
        shift = -4 if flight[u] == "t2" else 0
        for day in range(int(first[u]), DAYS):
            if day > first[u] and rng.random() < 0.4:
                continue
            for e in range(1 + int(rng.poisson(2))):
                records.append(dict(
                    ruid=f"u{u}", flight=str(flight[u]), os=os_[u], country=str(country[u]),
                    event_time=_at(day, int(rng.integers(0, 86400))),
                    media_type="video" if rng.random() < 0.3 else "photo",
                    clicks=int(rng.poisson(rate)), session_start=1 if e == 0 else 0,
                    latency_ms=max(0, int(rng.poisson(40)) + shift),
                ))
    return records


def to_pandas(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df["day"] = df["event_time"].map(lambda t: t.date())
    if "os" in df:
        df["os"] = df["os"].fillna("(null)")
    return df


def unit_values(events: pd.DataFrame, units: pd.DataFrame, value: str, how: str = "sum") -> pd.DataFrame:
    """``units`` with a column ``v``: each unit's total (or any-positive indicator), zero if no rows."""
    keys = ["ruid", "flight"]
    if events.empty:
        per_unit = pd.Series(dtype=float, name="v")
    elif how == "sum":
        per_unit = events.groupby(keys)[value].sum().astype(float)
    else:
        per_unit = (events[value] > 0).groupby([events["ruid"], events["flight"]]).max().astype(float)
    out = units.merge(per_unit.rename("v").reset_index(), on=keys, how="left")
    out["v"] = out["v"].fillna(0.0)
    return out


def welch(values: pd.DataFrame, treatment: str) -> Dict[str, float]:
    xc = values.loc[values["flight"] == "control", "v"].to_numpy()
    xt = values.loc[values["flight"] == treatment, "v"].to_numpy()
    return dict(n_control=len(xc), n_treatment=len(xt), value_control=xc.mean(), value_treatment=xt.mean(),
                p_value=st.ttest_ind(xt, xc, equal_var=False).pvalue)


def ratio(num: pd.DataFrame, den: pd.DataFrame, treatment: str) -> Dict[str, float]:
    merged = num.merge(den, on=["ruid", "flight"], suffixes=("_num", "_den"))

    def estimate(flight):
        g = merged[merged["flight"] == flight]
        y, x = g["v_num"].to_numpy(), g["v_den"].to_numpy()
        r = y.sum() / x.sum()
        return r, np.var(y - r * x, ddof=1) / (len(g) * x.mean() ** 2)

    (rc, vc), (rt, vt) = estimate("control"), estimate(treatment)
    return dict(value_control=rc, value_treatment=rt, p_value=2 * st.norm.sf(abs(rt - rc) / np.sqrt(vc + vt)))


def interpolated_quantile(values: np.ndarray, p: float, resolution: float = 1.0) -> float:
    """The DENSITY-method quantile, written directly in numpy as an independent check."""
    v, c = np.unique(values, return_counts=True)
    cum = np.cumsum(c)
    prev = np.r_[np.nan, v[:-1]]
    nxt = np.r_[v[1:], np.nan]
    start = (np.where(np.isnan(prev), v, np.maximum(prev, v - resolution)) + v) / 2
    end = (np.where(np.isnan(nxt), v, np.minimum(nxt, v + resolution)) + v) / 2
    i = int(np.argmax(cum >= p * cum[-1]))
    return start[i] + (p * cum[-1] - (cum[i] - c[i])) / c[i] * (end[i] - start[i])


def quantile(events: pd.DataFrame, n_units: Dict[str, int], value: str, p: float, treatment: str,
             method: str = "density", h: float = 0.001) -> Dict[str, float]:
    def find(x, prob):
        if method == "density":
            return interpolated_quantile(x, prob)
        return float(np.quantile(x, prob, method="inverted_cdf"))

    def estimate(flight):
        g = events[events["flight"] == flight]
        x = g[value].to_numpy(dtype=float)
        q = find(x, p)
        density = 2 * h / (find(x, p + h) - find(x, p - h))
        per_unit = g.assign(z=(g[value] > q).astype(float)).groupby("ruid").agg(n=("z", "size"), z=("z", "sum"))
        r = per_unit["z"].sum() / per_unit["n"].sum()
        var = ((per_unit["z"] - r * per_unit["n"]) ** 2).sum() / per_unit["n"].sum() ** 2 / density ** 2
        return q, density, var

    (qc, fc, vc), (qt, _, vt) = estimate("control"), estimate(treatment)
    return dict(value_control=qc, value_treatment=qt, density_control=fc, n_control=n_units["control"],
                p_value=2 * st.norm.sf(abs(qt - qc) / np.sqrt(vc + vt)))
