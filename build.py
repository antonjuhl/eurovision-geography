"""
Builds site/index.html for "Neighbours or Diaspora?".

    python build.py            (expects votes.csv in the same folder)

analysis.py  favoritism per directed pair per show (same measure as the first site)
geo.py       distances, data-found groups, distance test, permutation test, outliers, jury vs televote
build.py     runs both and writes everything into the page

On GitHub, .github/workflows/deploy.yml runs this on every push and publishes site/ with GitHub Pages.
"""
import json
from pathlib import Path

import numpy as np

from analysis import COORDS, ISO3, NAMES, build_table
from geo import (
    HYP_SLOT, HYPOTHESIS, PERIODS, hyp_of,
    period_payload, series, tele_jury, communities
)


HERE = Path(__file__).parent


def map_vote_payload(t, y0, y1, min_years, vote_type):
    column = {
        "combined": "ex",
        "jury": "jury",
        "public": "tele",
    }[vote_type]

    s = t[
        t.year.between(y0, y1) &
        (t.stage == "F")
    ].dropna(subset=[column])

    g = s.groupby(
        ["from_country_id", "to_country_id"]
    )

    ps = g.agg(
        favoritism=(column, "mean"),
        years=("year", "nunique"),
        consistency=(column, lambda x: (x > 0).mean())
    ).reset_index()

    ps = ps[ps.years >= min_years].reset_index(drop=True)

    if ps.empty:
        return {"pairs": [], "communities": []}

    comms, _ = communities(ps)

    pairs = [
        [a, b, round(float(v), 2), int(n), round(float(c), 2)]
        for a, b, v, n, c in zip(
            ps.from_country_id,
            ps.to_country_id,
            ps.favoritism,
            ps.years,
            ps.consistency
        )
    ]

    return {
        "pairs": pairs,
        "communities": comms
    }

def payload() -> dict:
    t = build_table(HERE / "votes.csv")
    rng = np.random.default_rng(42)  # fixed seed: the permutation test gives the same result every build
    ids = sorted(set(t.from_country_id) | set(t.to_country_id))
    idx = {c: i for i, c in enumerate(ids)}
    s = series(t)
    return {
        "countries": [{"id": c, "name": NAMES.get(c, c.upper()), "iso3": ISO3.get(c),
                       "lat": COORDS.get(c, (None, None))[0], "lon": COORDS.get(c, (None, None))[1],
                       "hyp": hyp_of(c)} for c in ids],
        "hyp": {g: {"slot": HYP_SLOT[g], "members": m} for g, m in HYPOTHESIS.items()},
        "periods": {k: period_payload(t, y0, y1, mn, rng) | {"y0": y0, "y1": y1, "min": mn}
                    for k, (y0, y1, mn) in PERIODS.items()},
        
        "mapVotes": {
            k: {
                vote: map_vote_payload(t, y0, y1, mn, vote)
                for vote in (
                    ["combined", "jury", "public"]
                    if y0 >= 2016 else ["combined"]
                )
            }
            for k, (y0, y1, mn) in PERIODS.items()
        },
        "tj": tele_jury(t),
        "ser": {"y": s.year.astype(int).tolist(), "f": [idx[c] for c in s.from_country_id],
                "t": [idx[c] for c in s.to_country_id], "ex": [int(round(v * 100)) for v in s.ex]},
    }


def main():
    data = json.dumps(payload(), separators=(",", ":"), ensure_ascii=False)
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", data)
    html = html.replace("/*__TOPO__*/null", (HERE / "world_50m.json").read_text(encoding="utf-8"))
    out = HERE / "site" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
