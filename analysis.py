"""
Eurovision geographical voting favoritism — data preparation.

Favoritism for a directed pair A -> B in one show (year + round):

    share(A->B)  = points A gave B / all points A gave in that show
    expected(B)  = mean share B received from every OTHER voter in that show
    excess       = (share - expected) * 58

Using shares makes 1975 (58 points per voter), 2016+ (116 points: jury + televote)
and the early years comparable. Multiplying by 58 converts the result back into
"extra points on the classic 1-8, 10, 12 scale": +5 means A gave B about five
points more than B normally gets from a country.

Leaving A out of B's expected share (leave-one-out) stops A's own vote from
pulling the baseline towards itself.
"""
import pandas as pd
import numpy as np

POINTS_PER_SET = 58  # 1+2+...+8+10+12

NAMES = {
    "ad": "Andorra", "al": "Albania", "am": "Armenia", "at": "Austria", "au": "Australia",
    "az": "Azerbaijan", "ba": "Bosnia & Herzegovina", "be": "Belgium", "bg": "Bulgaria",
    "by": "Belarus", "ch": "Switzerland", "cs": "Serbia & Montenegro", "cy": "Cyprus",
    "cz": "Czechia", "de": "Germany", "dk": "Denmark", "ee": "Estonia", "es": "Spain",
    "fi": "Finland", "fr": "France", "gb": "United Kingdom", "ge": "Georgia", "gr": "Greece",
    "hr": "Croatia", "hu": "Hungary", "ie": "Ireland", "il": "Israel", "is": "Iceland",
    "it": "Italy", "lt": "Lithuania", "lu": "Luxembourg", "lv": "Latvia", "ma": "Morocco",
    "mc": "Monaco", "md": "Moldova", "me": "Montenegro", "mk": "North Macedonia",
    "mt": "Malta", "nl": "Netherlands", "no": "Norway", "pl": "Poland", "pt": "Portugal",
    "ro": "Romania", "rs": "Serbia", "ru": "Russia", "se": "Sweden", "si": "Slovenia",
    "sk": "Slovakia", "sm": "San Marino", "tr": "Türkiye", "ua": "Ukraine", "yu": "Yugoslavia",
}

# lat, lon (capital or population centre — used for drawing lines on the map)
COORDS = {
    "at": (47.6, 14.1), "be": (50.6, 4.6), "dk": (56.0, 9.5), "fr": (46.6, 2.4),
    "de": (51.1, 10.4), "it": (42.8, 12.5), "lu": (49.8, 6.1), "nl": (52.2, 5.5),
    "ch": (46.8, 8.2), "gb": (53.0, -1.8), "se": (59.6, 15.4), "mc": (43.74, 7.42),
    "no": (61.0, 9.0), "fi": (62.5, 25.8), "es": (40.2, -3.7), "yu": (44.0, 20.5),
    "pt": (39.6, -8.0), "ie": (53.2, -8.0), "mt": (35.9, 14.4), "il": (31.4, 35.0),
    "gr": (39.3, 22.0), "tr": (39.0, 35.0), "ma": (31.8, -7.1), "cy": (35.0, 33.2),
    "is": (64.9, -18.6), "ba": (44.2, 17.8), "hr": (45.4, 15.8), "si": (46.1, 14.8),
    "ee": (58.7, 25.5), "hu": (47.2, 19.4), "lt": (55.3, 23.9), "pl": (52.0, 19.4),
    "ro": (45.9, 24.9), "ru": (55.75, 37.6), "sk": (48.7, 19.7), "mk": (41.6, 21.7),
    "lv": (56.9, 24.6), "ua": (49.0, 31.4), "al": (41.1, 20.0), "ad": (42.5, 1.5),
    "by": (53.7, 28.0), "cs": (43.5, 20.6), "bg": (42.7, 25.3), "md": (47.2, 28.5),
    "am": (40.1, 45.0), "cz": (49.8, 15.5), "ge": (42.2, 43.5), "me": (42.7, 19.3),
    "rs": (44.0, 20.9), "az": (40.3, 47.6), "sm": (43.94, 12.46),
}

ISO3 = {
    "at": "AUT", "be": "BEL", "dk": "DNK", "fr": "FRA", "de": "DEU", "it": "ITA", "lu": "LUX",
    "nl": "NLD", "ch": "CHE", "gb": "GBR", "se": "SWE", "mc": "MCO", "no": "NOR", "fi": "FIN",
    "es": "ESP", "pt": "PRT", "ie": "IRL", "mt": "MLT", "il": "ISR", "gr": "GRC", "tr": "TUR",
    "ma": "MAR", "cy": "CYP", "is": "ISL", "ba": "BIH", "hr": "HRV", "si": "SVN", "ee": "EST",
    "hu": "HUN", "lt": "LTU", "pl": "POL", "ro": "ROU", "ru": "RUS", "sk": "SVK", "mk": "MKD",
    "lv": "LVA", "ua": "UKR", "al": "ALB", "ad": "AND", "by": "BLR", "bg": "BGR", "md": "MDA",
    "am": "ARM", "cz": "CZE", "ge": "GEO", "me": "MNE", "rs": "SRB", "az": "AZE", "sm": "SMR",
    "au": "AUS",
}

# Cluster schemes. "original" = the five clusters from the project proposal.
CLUSTER_SCHEMES = {
    "original": {
        "Nordic": ["dk", "se", "no", "fi", "is"],
        "Balkan": ["al", "ba", "bg", "hr", "me", "mk", "ro", "rs", "si"],
        "Former Soviet / Baltic": ["am", "az", "by", "ee", "ge", "lt", "lv", "md", "ru", "ua"],
        "UK–Ireland–Malta": ["gb", "ie", "mt"],
        "Benelux": ["be", "nl", "lu"],
    },
    "split": {
        "Nordic": ["dk", "se", "no", "fi", "is"],
        "Balkan": ["al", "ba", "bg", "hr", "me", "mk", "ro", "rs", "si"],
        "Baltic": ["ee", "lt", "lv"],
        "Former Soviet (non-Baltic)": ["am", "az", "by", "ge", "md", "ru", "ua"],
        "UK–Ireland–Malta": ["gb", "ie", "mt"],
        "Benelux": ["be", "nl", "lu"],
    },
}


def _excess(df: pd.DataFrame, col: str) -> pd.Series:
    """Leave-one-out normalised excess for one points column."""
    d = df[["year", "round", "from_country_id", "to_country_id", col]].dropna(subset=[col]).copy()
    show = ["year", "round"]
    given = d.groupby(show + ["from_country_id"])[col].transform("sum")
    d = d[given > 0]
    given = given[given > 0]
    d["share"] = d[col] / given
    grp = d.groupby(show + ["to_country_id"])["share"]
    tot, cnt = grp.transform("sum"), grp.transform("count")
    loo = (tot - d["share"]) / (cnt - 1)
    ex = (d["share"] - loo) * POINTS_PER_SET
    ex[cnt < 2] = np.nan
    return ex


def build_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["from_country_id"] != "wld"]          # 2023 "rest of world" televote is not a country
    df = df[df["from_country_id"] != df["to_country_id"]]
    df["stage"] = np.where(df["round"] == "final", "F", "S")

    out = df[["year", "round", "stage", "from_country_id", "to_country_id"]].copy()
    out["ex"] = _excess(df, "total_points")
    # Separate jury / televote only exists from 2016
    split = df[df["year"] >= 2016]
    out["tele"] = _excess(split, "tele_points")
    out["jury"] = _excess(split, "jury_points")
    return out.dropna(subset=["ex"])


def pair_summary(t: pd.DataFrame, y0=1993, y1=2023, stage="F") -> pd.DataFrame:
    """Mean favoritism per directed pair — mirrors what the web page computes live."""
    s = t[(t.year.between(y0, y1)) & (t.stage.isin(list(stage)))]
    g = s.groupby(["from_country_id", "to_country_id"])
    return pd.DataFrame({
        "favoritism": g["ex"].mean(),
        "years": g["year"].nunique(),
        "consistency": g["ex"].apply(lambda x: (x > 0).mean()),
    }).reset_index()


if __name__ == "__main__":
    t = build_table("votes.csv")
    ps = pair_summary(t)
    nordic = CLUSTER_SCHEMES["original"]["Nordic"]
    m = ps[ps.from_country_id.isin(nordic) & ps.to_country_id.isin(nordic)]
    print(m.pivot(index="from_country_id", columns="to_country_id", values="favoritism").round(1))
    print(ps.sort_values("favoritism", ascending=False).query("years>=5").head(15).to_string())
