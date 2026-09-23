"""Memory-lean, leak-safe preprocessing + training for Spotify track popularity."""
import os
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

# =========================================================
# 0. Dynamic Path Setup (Forces saves to the current folder)
# =========================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # If running as a .py script
except NameError:
    BASE_DIR = os.getcwd() # If running in a Jupyter Notebook

# Accurately mapping to the files in your DATA ANALYTICS WITH AI folder
FEATURES_CSV = os.path.join(BASE_DIR, "C:/Users/ROHIT SHUKLA/Downloads/data analytics with ai/spotify_full.csv")
GEO_IN = os.path.join(BASE_DIR, "C:/Users/ROHIT SHUKLA/Downloads/data analytics with ai/universal_top_spotify_songs.csv")
GEO_OUT = os.path.join(BASE_DIR, "cleaned_universal_top_songs.csv")
PKL_OUT = os.path.join(BASE_DIR, "spotify_popularity_pipeline.pkl")

TARGET = "track_popularity"
SAMPLE_ROWS = 1_000_000   # rows used for training; set None to use everything (needs lots of RAM)
SPLIT_BY_ARTIST = False   # True = test on artists never seen in training

# =========================================================
# 1. Geo dataset: cleaned in chunks, never held in memory whole
# =========================================================
print("Processing Geo Dataset...")
first = True
for chunk in pd.read_csv(GEO_IN, chunksize=250_000):
    chunk = chunk.dropna(subset=["spotify_id", "artists", "daily_rank"])
    chunk["country"] = chunk["country"].fillna("GLOBAL")  # missing country = Global chart
    chunk["artists"] = chunk["artists"].astype(str).str.lower().str.strip()
    chunk.to_csv(GEO_OUT, mode="w" if first else "a", header=first, index=False)
    first = False
print(f"Geo data saved successfully to: {GEO_OUT}")

# =========================================================
# 2. Features dataset: load only what is needed, in compact dtypes
# =========================================================
print("\nLoading and processing Features Dataset...")
# Built from popularity / streams / chart performance -> leakage, unavailable for new songs
LEAKY = [
    "popularity_imputed_aug", "hit_probability", "streams_total", "streams_total_estimated",
    "stream_density", "chart_peak_position", "weeks_on_chart", "chart_date_first",
    "hit_rate", "artist_stream_mean_log", "artist_stream_median_log", "longevity_score",
]
# Raw text/dates and duplicates of other columns
NOISE = ["title", "title_normalized", "release_date", "genre_raw", "key_name", "mode_name",
         "anomaly_flags", "source_dataset", "streams_source"]

header = pd.read_csv(FEATURES_CSV, nrows=0).columns
cols = [c for c in header if c not in set(LEAKY + NOISE)]   # spotify_id kept for de-duplication

# Infer compact dtypes from a small head read: float32 for floats, category for text
head = pd.read_csv(FEATURES_CSV, usecols=cols, nrows=5000)
dtypes = {c: ("float32" if t == "float64" else "category")
          for c, t in head.dtypes.items() if t in ("float64", "object")}

n_rows = sum(1 for _ in open(FEATURES_CSV, "rb")) - 1
frac = 1.0 if not SAMPLE_ROWS or n_rows <= SAMPLE_ROWS else SAMPLE_ROWS / n_rows
rng = np.random.default_rng(42)
skip = (lambda i: i > 0 and rng.random() > frac) if frac < 1 else None

df = pd.read_csv(FEATURES_CSV, usecols=cols, dtype=dtypes, skiprows=skip)
print(f"Loaded {len(df):,} of {n_rows:,} rows, {df.shape[1]} columns, {df.memory_usage(deep=True).sum()/1e6:.0f} MB")

for c in ["track_artist", "genre_l1", "genre_l2"]:          # normalise text, one column at a time
    if c in df:
        df[c] = df[c].astype("string").str.lower().astype("category")
        gc.collect()

df = df.dropna(subset=[TARGET]).drop_duplicates(subset="spotify_id")  # one row per track
groups = df["track_artist"].astype(str) if SPLIT_BY_ARTIST and "track_artist" in df else None
y = df[TARGET].astype("float32")
X = df.drop(columns=[TARGET, "spotify_id"])
del df
gc.collect()

for c in X.select_dtypes("bool"):                            # bools are skipped by select_dtypes('number')
    X[c] = X[c].astype("int8")
    
cat_cols = X.select_dtypes(exclude="number").columns.tolist()   # bools already converted above
for c in cat_cols:
    if X[c].dtype != "category":
        X[c] = X[c].astype("category")
    if "unknown" not in X[c].cat.categories:
        X[c] = X[c].cat.add_categories("unknown")
    X[c] = X[c].fillna("unknown")
    
num_cols = X.select_dtypes(include="number").columns.tolist()
print(f"Categorical features: {len(cat_cols)}")
print(f"Numeric features: {len(num_cols)}")

# =========================================================
# 3. Split BEFORE any fitted step (index-based, avoids extra copies)
# =========================================================
if SPLIT_BY_ARTIST and groups is not None:
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
else:
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    
X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
feature_columns = X.columns.tolist()
del X, y, groups
gc.collect()

# Leak sanity check on a sample: anything this correlated with the target deserves a look
s = X_train.sample(min(len(X_train), 200_000), random_state=0)
corr = s[num_cols].corrwith(y_train.loc[s.index]).abs().sort_values(ascending=False)
print("\nTop |corr| with target (investigate anything > ~0.6):")
print(corr.head(10).round(3))
del s

# =========================================================
# 4. Pipeline (everything fitted on train only)
# =========================================================
pre = ColumnTransformer([
    ("cat", TargetEncoder(target_type="continuous", cv=5, smooth="auto", random_state=42), cat_cols),
    ("num", "passthrough", num_cols),
])

pipe = Pipeline([
    ("prep", pre),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8,
                           tree_method="hist", random_state=42, n_jobs=-1)),
])

print("\nTraining Model...")
pipe.fit(X_train, y_train)

# =========================================================
# 5. Evaluation vs. a naive baseline
# =========================================================
pred = pipe.predict(X_test)
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))
print(f"\nBaseline RMSE (predict the mean): {rmse(y_test, np.full(len(y_test), y_train.mean())):.3f}")
print(f"Model RMSE: {rmse(y_test, pred):.3f}")
print(f"Model MAE : {mean_absolute_error(y_test, pred):.3f}")
print(f"Model R2  : {r2_score(y_test, pred):.3f}")

names = pipe.named_steps["prep"].get_feature_names_out()
imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop features (one feature dominating = likely leak):")
print(imp.head(15).round(3))

# =========================================================
# 6. Export (Using forced absolute path)
# =========================================================
joblib.dump({
    "pipeline": pipe, 
    "feature_columns": feature_columns,
    "cat_cols": cat_cols, 
    "num_cols": num_cols,
    "num_defaults": X_train[num_cols].median().to_dict()
}, PKL_OUT)

print(f"\nPipeline safely saved to: {PKL_OUT}")