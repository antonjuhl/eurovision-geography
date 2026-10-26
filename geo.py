"""
Geography or culture? — the analysis behind the second site.

Builds on analysis.py (favoritism per directed pair per show, see that file).
Everything here is computed per period and handed to the page as JSON:

  communities   groups the data finds by itself (Louvain on mutual favoritism)
  distance      favoritism against distance between the two countries
  matched       hypothesis clusters vs other pairs at the same distance
  permutation   are the hypothesis clusters better than random groupings?
  outliers      far-apart pairs that favour each other far more than distance predicts
  tele_jury     televote vs jury favoritism per directed pair, 2016-2023
"""
import numpy as np
import pandas as pd
import networkx as nx

from analysis import CLUSTER_SCHEMES, COORDS, NAMES, build_table, pair_summary

HYPOTHESIS = CLUSTER_SCHEMES["original"]

PERIODS = {
    "1957-2023": (1957, 2023, 5),
    "1957-1974": (1957, 1974, 3),
    "1975-2003": (1975, 2003, 4),
    "2004-2015": (2004, 2015, 4),
    "2016-2023": (2016, 2023, 3),
}

# Data communities get their colour from the anchor country they contain,
# so a group keeps its colour when the period changes.
ANCHORS = ["se", "rs", "ru", "fr", "gr"]
HYP_SLOT = {"Nordic": 0, "Balkan": 1, "Former Soviet / Baltic": 2, "Benelux": 3, "UK–Ireland–Malta": 5}

DIST_BINS = [0, 500, 1000, 1500, 2000, 2500, 3000, 4500]
MATCH_BINS = [(0, 750), (750, 1500), (1500, 2500)]


def km(a, b):
    (la1, lo1), (la2, lo2) = COORDS[a], COORDS[b]
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def hyp_of(c):
    for g, members in HYPOTHESIS.items():
        if c in members:
            return g
    return None


def europe(c):  # Australia is 13,000 km away and would flatten every distance chart
    return c in COORDS and c != "au"


# ---------------------------------------------------------------- communities
def communities(ps):
    d = {(a, b): v for a, b, v in zip(ps.from_country_id, ps.to_country_id, ps.favoritism)}
    G = nx.Graph()
    for (a, b), v in d.items():
        if a < b and (b, a) in d:
            w = (v + d[(b, a)]) / 2
            if w > 0:
                G.add_edge(a, b, weight=w)
    best = None
    for seed in range(40):  # Louvain is randomised: keep the best of 40 runs
        cs = nx.community.louvain_communities(G, weight="weight", seed=seed)
        q = nx.community.modularity(G, cs, weight="weight")
        if best is None or q > best[0] + 1e-9:
            best = (q, cs)
    q, cs = best

    out, used = [], set()
    for members in cs:
        strength = {c: sum(G[c][o]["weight"] for o in members if o != c and G.has_edge(c, o)) for c in members}
        order = sorted(members, key=lambda c: -strength[c])
        slot = next((i for i, a in enumerate(ANCHORS) if a in members and i not in used), None)
        if slot is not None:
            used.add(slot)
        out.append({"members": order, "slot": slot})
    free = iter(i for i in range(8) if i not in used and i != 5)
    for g in out:
        if g["slot"] is None:
            g["slot"] = next(free)
        top = [NAMES[c] for c in g["members"][:2]]
        g["name"] = " – ".join(top) + (f" +{len(g['members']) - 2}" if len(g["members"]) > 2 else "")
    out.sort(key=lambda g: g["slot"])
    return out, round(q, 3)


# ---------------------------------------------------------------- one period
def period_payload(t, y0, y1, min_years, rng):
    ps = pair_summary(t, y0, y1, "F")
    ps = ps[ps.years >= min_years].reset_index(drop=True)
    directed = {(a, b): (v, n, c) for a, b, v, n, c in
                zip(ps.from_country_id, ps.to_country_id, ps.favoritism, ps.years, ps.consistency)}

    comms, modularity = communities(ps)

    # undirected pairs with a distance (Europe only)
    rows = []
    for (a, b), (v, n, _) in directed.items():
        if a < b and europe(a) and europe(b):
            vals = [v] + ([directed[(b, a)][0]] if (b, a) in directed else [])
            h = hyp_of(a)
            rows.append({"a": a, "b": b, "fav": float(np.mean(vals)), "km": km(a, b),
                         "inside": bool(h and h == hyp_of(b))})
    und = pd.DataFrame(rows)

    def binned(df):
        out = []
        for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
            s = df[(df.km >= lo) & (df.km < hi)]
            if len(s) >= 5:
                out.append({"km": (lo + hi) / 2, "mean": round(float(s.fav.mean()), 2), "n": int(len(s))})
        return out

    trend = {"all": binned(und), "inside": binned(und[und.inside]), "outside": binned(und[~und.inside])}

    matched = []
    for lo, hi in MATCH_BINS:
        s = und[(und.km >= lo) & (und.km < hi)]
        matched.append({"band": f"{lo:,}–{hi:,} km".replace(",", "."),
                        "inside": round(float(s[s.inside].fav.mean()), 2) if s.inside.any() else None,
                        "n_in": int(s.inside.sum()),
                        "outside": round(float(s[~s.inside].fav.mean()), 2), "n_out": int((~s.inside).sum())})

    # how much do distance and the hypothesis clusters explain?
    y = und.fav.values
    def r2(*cols):
        X = np.column_stack([np.ones(len(und))] + list(cols))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()), beta
    logkm, ins = np.log(und.km.values), und.inside.values.astype(float)
    r2_dist, beta_dist = r2(logkm)
    r2_both, beta_both = r2(logkm, ins)
    r2_cl, _ = r2(ins)

    # permutation test: shuffle hypothesis labels between clustered countries, keep cluster sizes
    lab = {c: g for g, m in HYPOTHESIS.items() for c in m}
    members = [c for c in lab if europe(c)]
    real = float(und[und.inside].fav.mean())
    sims = []
    for _ in range(1000):
        perm = dict(zip(members, rng.permutation([lab[c] for c in members])))
        m = np.array([perm.get(a) is not None and perm.get(a) == perm.get(b) for a, b in zip(und.a, und.b)])
        sims.append(float(und.fav[m].mean()))
    sims = np.array(sims)
    hist, edges = np.histogram(sims, bins=24)

    # outliers: directed pairs far above what their distance predicts
    a0, a1 = beta_dist
    out = []
    for (a, b), (v, n, c) in directed.items():
        if not (europe(a) and europe(b)):
            continue
        d = km(a, b)
        pred = a0 + a1 * np.log(d)
        out.append((v - pred, a, b, v, n, c, d, pred))
    out.sort(reverse=True)
    far = [o for o in out if o[6] >= 1000][:14]
    outliers = [{"a": a, "b": b, "fav": round(v, 2), "years": int(n), "cons": round(c, 2), "km": round(d),
                 "pred": round(pred, 2), "rev": round(directed[(b, a)][0], 2) if (b, a) in directed else None}
                for _, a, b, v, n, c, d, pred in far]

    return {
        "pairs": [[a, b, round(v, 2), int(n), round(c, 2)] for (a, b), (v, n, c) in directed.items()],
        "communities": comms, "modularity": modularity,
        "scatter": [[r.a, r.b, round(r.fav, 2), round(r.km), int(r.inside)] for r in und.itertuples()],
        "trend": trend, "matched": matched,
        "fit": {"r2_dist": round(r2_dist, 3), "r2_cluster": round(r2_cl, 3), "r2_both": round(r2_both, 3),
                "cluster_bonus": round(float(beta_both[2]), 2), "per_doubling": round(float(beta_dist[1] * np.log(2)), 2),
                "corr": round(float(np.corrcoef(logkm, y)[0, 1]), 3), "n": int(len(und))},
        "perm": {"real": round(real, 2), "mean": round(float(sims.mean()), 2), "max": round(float(sims.max()), 2),
                 "p": float((sims >= real).mean()), "runs": len(sims),
                 "hist": [[round(float((edges[i] + edges[i + 1]) / 2), 3), int(h)] for i, h in enumerate(hist)]},
        "outliers": outliers,
    }


def tele_jury(t, min_shows=3):
    s = t[(t.year >= 2016) & (t.stage == "F")].dropna(subset=["tele", "jury"])
    g = s.groupby(["from_country_id", "to_country_id"])
    df = pd.DataFrame({"tele": g.tele.mean(), "jury": g.jury.mean(), "n": g.year.nunique()}).reset_index()
    df = df[(df.n >= min_shows) & df.from_country_id.map(europe) & df.to_country_id.map(europe)]
    df["km"] = [km(a, b) for a, b in zip(df.from_country_id, df.to_country_id)]
    return [[a, b, round(te, 2), round(ju, 2), int(n), round(k)] for a, b, te, ju, n, k in
            zip(df.from_country_id, df.to_country_id, df.tele, df.jury, df.n, df.km)]


def series(t):
    """Per directed pair per final, for the year-by-year pair view."""
    f = t[t.stage == "F"]
    return f[["year", "from_country_id", "to_country_id", "ex"]]


if __name__ == "__main__":
    t = build_table("votes.csv")
    rng = np.random.default_rng(42)
    for key, (y0, y1, mn) in PERIODS.items():
        p = period_payload(t, y0, y1, mn, rng)
        print(f"\n== {key}  modularity {p['modularity']}")
        for c in p["communities"]:
            print(f"  slot {c['slot']}  {c['name']:40s} {', '.join(c['members'])}")
        print("  fit", p["fit"])
        print("  perm", {k: v for k, v in p["perm"].items() if k != "hist"})
        print("  matched", p["matched"])
        print("  outliers", [(o["a"], o["b"], o["fav"], o["km"]) for o in p["outliers"][:8]])
    tj = tele_jury(t)
    print("\ntele/jury pairs", len(tj))
