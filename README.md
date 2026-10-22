# Neighbours or Diaspora?

Second site for the Eurovision favoritism project. Where the first site shows *that* clusters favour each other,
this one asks *why*: is it geography, or language, culture and emigrants?

## Run locally

    pip install -r requirements.txt
    python build.py

Open `site/index.html`.

## Files

- `analysis.py` – favoritism per directed pair per final (same measure as the first site).
- `geo.py` – distances, the groups the votes form by themselves (Louvain), distance test, permutation test
  against random clusters, outliers, jury vs televote. `python geo.py` prints the results per period.
- `build.py` – runs both and writes `site/index.html`.
- `template.html` – the page (Plotly).
- `.github/workflows/deploy.yml` – builds and publishes on GitHub Pages at every push.

## Put it online

Same steps as the first site, in a **new** repository (e.g. `eurovision-geography`):
upload the files, create `.github/workflows/deploy.yml`, set Settings → Pages → Source to GitHub Actions.
