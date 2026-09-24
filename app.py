import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "spotify_popularity_pipeline.pkl")
GEO_PATH = os.path.join(APP_DIR, "cleaned_dataset_small.csv")
GREEN, DARK = "#1DB954", "#191414"

st.set_page_config(page_title="Spotify Analytics & AI", layout="wide", page_icon="🎵")
st.markdown(
    f"""<style>
    .block-container {{padding-top: 1.5rem;}}
    div[data-testid="stMetric"] {{background:#f6f8f7;border-left:4px solid {GREEN};
        padding:12px 16px;border-radius:8px;}}
    h1, h2, h3 {{letter-spacing:-0.5px;}}
    </style>""",
    unsafe_allow_html=True,
)

ISO = dict(AE="ARE", AR="ARG", AT="AUT", AU="AUS", BE="BEL", BG="BGR", BO="BOL", BR="BRA", BY="BLR",
           CA="CAN", CH="CHE", CL="CHL", CO="COL", CR="CRI", CY="CYP", CZ="CZE", DE="DEU", DK="DNK",
           DO="DOM", EC="ECU", EE="EST", EG="EGY", ES="ESP", FI="FIN", FR="FRA", GB="GBR", GR="GRC",
           GT="GTM", HK="HKG", HN="HND", HU="HUN", ID="IDN", IE="IRL", IL="ISR", IN="IND", IS="ISL",
           IT="ITA", JP="JPN", KR="KOR", KZ="KAZ", LT="LTU", LU="LUX", LV="LVA", MA="MAR", MX="MEX",
           MY="MYS", NG="NGA", NI="NIC", NL="NLD", NO="NOR", NZ="NZL", PA="PAN", PE="PER", PH="PHL",
           PK="PAK", PL="POL", PT="PRT", PY="PRY", RO="ROU", SA="SAU", SE="SWE", SG="SGP", SK="SVK",
           SV="SLV", TH="THA", TR="TUR", TW="TWN", UA="UKR", US="USA", UY="URY", VE="VEN", VN="VNM",
           ZA="ZAF")


# =========================================================
# Data + model loading (cached)
# =========================================================
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)


SAMPLE_FRAC = 1.0  # lower (e.g. 0.5) if your machine is still short on RAM
NEEDED = ["spotify_id", "name", "artists", "daily_rank", "country", "snapshot_date", "popularity",
          "is_explicit", "danceability", "energy", "valence", "acousticness", "speechiness",
          "instrumentalness", "liveness", "loudness", "tempo"]


# cache_resource returns the same object every rerun. cache_data would copy 2M+ rows each time.
@st.cache_resource(show_spinner="Loading chart data…")
def load_geo():
    header = pd.read_csv(GEO_PATH, nrows=0).columns
    cols = [c for c in NEEDED if c in header]
    dtypes = {c: "category" for c in ["spotify_id", "name", "artists", "country", "is_explicit"] if c in cols}
    dtypes.update({c: "float32" for c in cols if c not in dtypes and c != "snapshot_date"})
    skip = None
    if SAMPLE_FRAC < 1:
        rng = np.random.default_rng(1)
        skip = lambda i: i > 0 and rng.random() > SAMPLE_FRAC
    df = pd.read_csv(GEO_PATH, usecols=cols, dtype=dtypes, parse_dates=["snapshot_date"], skiprows=skip)
    if df["country"].isna().any():
        if "GLOBAL" not in df["country"].cat.categories:
            df["country"] = df["country"].cat.add_categories("GLOBAL")
        df["country"] = df["country"].fillna("GLOBAL")
    if "is_explicit" in df:
        df["is_explicit"] = df["is_explicit"].cat.rename_categories(lambda x: str(x).lower()) == "true"
    return df


@st.cache_resource(show_spinner=False)
def artist_index(_df):
    t = _df.drop_duplicates("spotify_id")[["spotify_id", "artists"]].astype(str)
    t = t.assign(artist=t["artists"].str.split(",")).explode("artist")
    t["artist"] = t["artist"].str.strip()
    return t[["spotify_id", "artist"]]


def apply_filters(df, d0, d1, countries, all_countries, explicit):
    m = df["snapshot_date"].between(d0, d1)
    if len(countries) < len(all_countries):
        m &= df["country"].isin(countries)
    if explicit != "All" and "is_explicit" in df:
        m &= df["is_explicit"] == (explicit == "Explicit only")
    return df if m.all() else df[m]  # no copy when nothing is filtered


try:
    artifacts = load_model()
    df_geo = load_geo()
except FileNotFoundError:
    st.error(f"Cannot find the model or data files. Keep them in: **{APP_DIR}**")
    st.stop()

pipeline = artifacts["pipeline"]
feature_columns = artifacts["feature_columns"]
num_defaults = artifacts.get("num_defaults", {})
# Read the column groups + known categories straight from the fitted pipeline
_, encoder, cat_cols = pipeline.named_steps["prep"].transformers_[0]
num_cols = [c for c in feature_columns if c not in cat_cols]
cat_options = {c: sorted(map(str, cats)) for c, cats in zip(cat_cols, encoder.categories_)}

AUDIO = [c for c in ["danceability", "energy", "valence", "acousticness", "speechiness",
                     "instrumentalness", "liveness"] if c in df_geo.columns]
idx = artist_index(df_geo)
top_artists = idx["artist"].value_counts().head(3000).index.tolist()


def build_rows(overrides, n=1):
    """One fully typed input row: strings for categoricals, numbers for numerics."""
    row = {}
    for c in feature_columns:
        row[c] = "unknown" if c in cat_cols else num_defaults.get(c, np.nan)
    row.update({k: v for k, v in overrides.items() if k in feature_columns})
    df = pd.DataFrame([row] * n)[feature_columns]
    df[cat_cols] = df[cat_cols].astype(str)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


# =========================================================
# Sidebar: global filters
# =========================================================
st.sidebar.title("🎛️ Filters")
dmin, dmax = df_geo["snapshot_date"].min().date(), df_geo["snapshot_date"].max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
d0, d1 = (dr if isinstance(dr, tuple) and len(dr) == 2 else (dmin, dmax))
all_countries = sorted(df_geo["country"].unique())
countries = st.sidebar.multiselect("Markets", all_countries, default=all_countries)
explicit_f = st.sidebar.radio("Content", ["All", "Explicit only", "Clean only"], horizontal=True)
f = apply_filters(df_geo, pd.Timestamp(d0), pd.Timestamp(d1), countries, all_countries, explicit_f)

st.title("🎵 Spotify Analytics & AI Dashboard")
st.caption("Market intelligence from daily Spotify charts, plus an ML popularity predictor.")
if f.empty:
    st.warning("No data for the current filters.")
    st.stop()

k = st.columns(5)
k[0].metric("Chart records", f"{len(f):,}")
k[1].metric("Unique tracks", f"{f['spotify_id'].nunique():,}")
k[2].metric("Markets", f"{f['country'].nunique()}")
k[3].metric("Avg popularity", f"{f['popularity'].mean():.1f}" if "popularity" in f else "n/a")
k[4].metric("Explicit share", f"{f['is_explicit'].mean():.0%}" if "is_explicit" in f else "n/a")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Market Overview", "🌍 Geo Explorer", "🎤 Artist Deep Dive", "🤖 AI Predictor", "📁 Data"])

# =========================================================
# Tab 1: Market overview
# =========================================================
with tab1:
    c1, c2 = st.columns(2)
    top_n = c1.slider("Top N artists", 5, 30, 15)
    per_track = f["spotify_id"].value_counts()
    per_track = per_track[per_track > 0].rename("n").rename_axis("spotify_id").reset_index()
    per_track["spotify_id"] = per_track["spotify_id"].astype(str)
    top = (per_track.merge(idx, on="spotify_id").groupby("artist")["n"].sum()
           .nlargest(top_n).sort_values().reset_index())
    top.columns = ["artist", "chart_appearances"]
    c1.plotly_chart(px.bar(top, x="chart_appearances", y="artist", orientation="h",
                           title="Most-charted artists", color_discrete_sequence=[GREEN]),
                    use_container_width=True)

    ycol = "popularity" if "popularity" in f else "daily_rank"
    daily = f.groupby("snapshot_date")[ycol].mean()
    fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, line=dict(color=GREEN)))
    fig.update_layout(title="Average popularity over time", yaxis_title="popularity")
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature explorer")
    num_avail = [c for c in f.select_dtypes("number").columns if c not in ("daily_rank",)]
    s1, s2, s3 = st.columns(3)
    xv = s1.selectbox("X axis", num_avail, index=num_avail.index("energy") if "energy" in num_avail else 0)
    yv = s2.selectbox("Y axis", num_avail, index=num_avail.index("valence") if "valence" in num_avail else 1)
    cv = s3.selectbox("Colour", ["country"] + num_avail,
                      index=(num_avail.index("popularity") + 1) if "popularity" in num_avail else 0)
    sample = f.sample(min(len(f), 15000), random_state=1)
    st.plotly_chart(px.scatter(sample, x=xv, y=yv, color=cv, opacity=0.5, hover_name="name",
                               title=f"{yv} vs {xv} (15k-row sample)"), use_container_width=True)

    if len(AUDIO) > 2:
        corr_cols = AUDIO + (["popularity"] if "popularity" in f else [])
        corr = f[corr_cols].sample(min(len(f), 50000), random_state=1).corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, title="Audio feature correlations"),
                        use_container_width=True)

# =========================================================
# Tab 2: Geo explorer
# =========================================================
with tab2:
    g1, g2, g3 = st.columns([2, 2, 1])
    metric = g1.selectbox("Map metric", ["Chart appearances", "Unique tracks", "Avg daily rank", "Avg popularity"])
    who = g2.selectbox("Artist (optional)", ["All artists"] + top_artists)
    g = f
    if who != "All artists":
        g = f[f["spotify_id"].isin(idx.loc[idx["artist"] == who, "spotify_id"])]
    if g.empty:
        st.warning("No mappable data for this selection.")
    else:
        agg = g.groupby("country", observed=True).agg(
            appearances=("spotify_id", "size"), tracks=("spotify_id", "nunique"),
            avg_rank=("daily_rank", "mean"),
            avg_pop=("popularity", "mean") if "popularity" in g else ("daily_rank", "mean")).reset_index()
        agg["iso3"] = agg["country"].astype(str).map(ISO)
        agg = agg.dropna(subset=["iso3"])
        col = {"Chart appearances": "appearances", "Unique tracks": "tracks",
               "Avg daily rank": "avg_rank", "Avg popularity": "avg_pop"}[metric]
        scale = "Greens_r" if col == "avg_rank" else "Greens"
        st.plotly_chart(px.choropleth(agg, locations="iso3", color=col, hover_name="country",
                                      color_continuous_scale=scale,
                                      title=f"{metric}: {who}"), use_container_width=True)
        st.subheader("Market drill-down")
        mk = st.selectbox("Pick a market", sorted(agg["country"]))
        m = g[g["country"] == mk]
        t = (m.groupby(["name", "artists"], observed=True).agg(days_on_chart=("snapshot_date", "nunique"),
                                                best_rank=("daily_rank", "min"),
                                                avg_rank=("daily_rank", "mean"))
             .sort_values(["best_rank", "days_on_chart"], ascending=[True, False]).head(15).reset_index())
        st.dataframe(t.round(1), use_container_width=True, hide_index=True)

# =========================================================
# Tab 3: Artist deep dive
# =========================================================
with tab3:
    a1, a2 = st.columns(2)
    art = a1.selectbox("Artist", top_artists, index=0)
    rival = a2.selectbox("Compare with (optional)", ["Market average"] + top_artists)

    def artist_frame(name):
        return f[f["spotify_id"].isin(idx.loc[idx["artist"] == name, "spotify_id"])]

    ad = artist_frame(art)
    if ad.empty:
        st.info("This artist has no chart records under the current filters.")
    else:
        m = st.columns(4)
        m[0].metric("Chart appearances", f"{len(ad):,}")
        m[1].metric("Tracks charted", f"{ad['spotify_id'].nunique()}")
        m[2].metric("Markets reached", f"{ad['country'].nunique()}")
        m[3].metric("Best rank", f"#{int(ad['daily_rank'].min())}")

        r1, r2 = st.columns(2)
        base = f if rival == "Market average" else artist_frame(rival)
        if len(AUDIO) >= 3 and not base.empty:
            fig = go.Figure()
            for nm, d in [(art, ad), (rival, base)]:
                vals = [d[c].mean() for c in AUDIO]
                fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=AUDIO + AUDIO[:1], fill="toself", name=nm))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), title="Acoustic profile")
            r1.plotly_chart(fig, use_container_width=True)

        reach = ad.groupby("country", observed=True).size().sort_values(ascending=False).head(15).reset_index(name="appearances")
        r2.plotly_chart(px.bar(reach, x="country", y="appearances", title="Top markets",
                               color_discrete_sequence=[GREEN]), use_container_width=True)

        tracks = ad["name"].value_counts().head(5).index
        tl = ad[ad["name"].isin(tracks)].groupby(["snapshot_date", "name"], observed=True)["daily_rank"].min().reset_index()
        fig = px.line(tl, x="snapshot_date", y="daily_rank", color="name", title="Best daily rank of top tracks")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# Tab 4: AI predictor
# =========================================================
with tab4:
    st.subheader("Predict track popularity")
    known_artists = set(map(str, encoder.categories_[cat_cols.index("track_artist")])) if "track_artist" in cat_cols else set()
    p1, p2, p3 = st.columns(3)
    with p1:
        artist = st.text_input("Artist", "The Weeknd")
        g1_opts = cat_options.get("genre_l1", ["unknown"])
        genre1 = st.selectbox("Primary genre", g1_opts, index=g1_opts.index("pop") if "pop" in g1_opts else 0)
        g2_opts = cat_options.get("genre_l2", ["unknown"])
        genre2 = st.selectbox("Sub-genre", g2_opts)
        explicit = st.checkbox("Explicit")
    with p2:
        dance = st.slider("Danceability", 0.0, 1.0, 0.6)
        energy = st.slider("Energy", 0.0, 1.0, 0.7)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        loud = st.slider("Loudness (dB)", -60.0, 0.0, -6.0)
    with p3:
        tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0)
        dur = st.slider("Duration (min)", 1.0, 8.0, 3.3)
        year = st.slider("Release year", 1960, 2026, 2024)
    with st.expander("Advanced audio features"):
        e1, e2, e3, e4 = st.columns(4)
        acoustic = e1.slider("Acousticness", 0.0, 1.0, 0.2)
        speech = e2.slider("Speechiness", 0.0, 1.0, 0.08)
        instr = e3.slider("Instrumentalness", 0.0, 1.0, 0.0)
        live = e4.slider("Liveness", 0.0, 1.0, 0.15)
        key = e1.selectbox("Key", list(range(12)), index=5)
        mode = e2.selectbox("Mode", [1, 0], format_func=lambda v: "Major" if v else "Minor")
        tsig = e3.selectbox("Time signature", [3, 4, 5], index=1)

    ov = dict(track_artist=artist.lower().strip(), genre_l1=genre1, genre_l2=genre2,
              danceability=dance, energy=energy, valence=valence, loudness=loud, tempo=tempo,
              duration_ms=dur * 60000, release_year=year, explicit_flag=int(explicit),
              acousticness=acoustic, speechiness=speech, instrumentalness=instr, liveness=live,
              key=key, mode=mode, time_signature=tsig)

    if st.button("Predict score", type="primary"):
        pred = float(np.clip(pipeline.predict(build_rows(ov))[0], 0, 100))
        ref = float(df_geo["popularity"].mean()) if "popularity" in df_geo else 50.0
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred, delta={"reference": ref, "valueformat": ".1f"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN},
                   "threshold": {"line": {"color": DARK, "width": 3}, "value": ref}},
            title={"text": "Predicted popularity (delta vs charting-song average)"}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        if known_artists and ov["track_artist"] not in known_artists:
            st.info("Artist not seen in training data, so the model used an average artist effect.")
        st.session_state["ov"] = ov

    if "ov" in st.session_state:
        st.subheader("What-if analysis")
        sweep_opts = [c for c in ["danceability", "energy", "valence", "loudness", "tempo", "acousticness",
                                  "speechiness", "release_year"] if c in feature_columns]
        sv = st.selectbox("Vary a feature", sweep_opts)
        lo, hi = {"loudness": (-40, 0), "tempo": (60, 200), "release_year": (1980, 2026)}.get(sv, (0, 1))
        grid = np.linspace(lo, hi, 30)
        rows = build_rows(st.session_state["ov"], n=len(grid))
        rows[sv] = grid
        curve = np.clip(pipeline.predict(rows), 0, 100)
        fig = px.line(x=grid, y=curve, labels={"x": sv, "y": "predicted popularity"},
                      title=f"Sensitivity of predicted popularity to {sv}")
        fig.update_traces(line_color=GREEN)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Derived columns (e.g. mood or ratio indices) stay at typical values, so this shows direct effects only.")

    with st.expander("Model insights: what drives predictions"):
        try:
            names = pipeline.named_steps["prep"].get_feature_names_out()
            imp = (pd.Series(pipeline.named_steps["model"].feature_importances_, index=names)
                   .sort_values().tail(15))
            imp.index = [i.split("__", 1)[-1] for i in imp.index]
            st.plotly_chart(px.bar(imp, orientation="h", title="Top 15 feature importances",
                                   color_discrete_sequence=[GREEN]), use_container_width=True)
        except Exception as e:
            st.write(f"Feature importances unavailable: {e}")

# =========================================================
# Tab 5: Data
# =========================================================
with tab5:
    st.caption(f"{len(f):,} rows match the current filters. Showing the first 1,000.")
    st.dataframe(f.head(1000), use_container_width=True)
    st.download_button("Download filtered data (max 100k rows)",
                       f.head(100000).to_csv(index=False).encode("utf-8"),
                       "filtered_spotify_charts.csv", "text/csv")