import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

import joblib

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf


# ------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Explainable Rainfall Forecasting AI",
    page_icon="🌧️",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM UI THEME — Deep Storm Aesthetic
# ------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:        #050d1a;
  --surface:   #0b1929;
  --surface2:  #0f2035;
  --border:    #1a3a5c;
  --accent:    #00c8ff;
  --accent2:   #0066ff;
  --accent3:   #7b2fff;
  --rain:      #38bdf8;
  --text:      #cce7ff;
  --muted:     #5b8ab0;
  --danger:    #ff4d6d;
  --warn:      #fbbf24;
  --success:   #34d399;
}

html, body, [class*="css"], .stApp, p, label, div {
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,102,255,0.18) 0%, transparent 70%),
    radial-gradient(ellipse 40% 30% at 90% 80%, rgba(123,47,255,0.12) 0%, transparent 60%) !important;
}

.block-container {
  max-width: 1280px !important;
  padding: 1.75rem 2rem 2rem !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Syne', sans-serif !important;
  letter-spacing: -0.02em;
}
h1 { color: #ffffff !important; font-weight: 800; font-size: 2.2rem !important; }
h2 { color: var(--accent) !important; font-weight: 700; }
h3 { color: var(--rain) !important; font-weight: 700; }

div[data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 12px !important;
  padding: 4px 6px !important;
  gap: 4px !important;
  border: 1px solid var(--border) !important;
}
button[data-baseweb="tab"] {
  font-family: 'Space Mono', monospace !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em;
  color: var(--muted) !important;
  border-radius: 8px !important;
  padding: 6px 14px !important;
  border: none !important;
  background: transparent !important;
  text-transform: uppercase;
}
button[aria-selected="true"] {
  background: var(--accent2) !important;
  color: #ffffff !important;
  box-shadow: 0 0 16px rgba(0,102,255,0.45) !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text) !important;
  background: var(--surface2) !important;
}

div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 18px 20px !important;
  box-shadow: 0 0 24px rgba(0,200,255,0.06) !important;
  transition: box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
  box-shadow: 0 0 30px rgba(0,102,255,0.2) !important;
}
[data-testid="stMetricLabel"] p {
  font-family: 'Space Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  font-family: 'Space Mono', monospace !important;
}

.stButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1.5px solid var(--accent2) !important;
  border-radius: 10px !important;
  padding: 8px 22px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: var(--accent2) !important;
  color: #fff !important;
  box-shadow: 0 0 20px rgba(0,102,255,0.4) !important;
}

.stSelectbox div[data-baseweb="select"] > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div {
  color: var(--text) !important;
}
ul[role="listbox"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
ul[role="listbox"] li {
  color: var(--text) !important;
  background: transparent !important;
}
ul[role="listbox"] li:hover {
  background: var(--accent2) !important;
  color: #fff !important;
}

.stDateInput input {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

[data-baseweb="calendar"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-baseweb="calendar"] * { color: var(--text) !important; }
[data-baseweb="calendar"] [aria-selected="true"] {
  background: var(--accent2) !important;
  border-radius: 50% !important;
}

.stSelectbox label, .stDateInput label, .stSlider label {
  color: var(--muted) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }
thead tr th {
  background: var(--surface2) !important;
  color: var(--accent) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  border-bottom: 1px solid var(--border) !important;
}
tbody tr td {
  color: var(--text) !important;
  background: var(--surface) !important;
  border-bottom: 1px solid rgba(26,58,92,0.5) !important;
}
tbody tr:hover td { background: var(--surface2) !important; }

div[data-testid="stAlert"] {
  border-radius: 12px !important;
  border-left-width: 3px !important;
  font-weight: 600;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent2); }

.stCaption p, small {
  color: var(--muted) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 11px !important;
}

hr { border-color: var(--border) !important; }

.js-plotly-plot .plotly { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------
DATASET = "weatherAUS.csv"
MODEL_FILE = "rain_lstm_model_v4.h5"
PREPROCESSOR_FILE = "preprocessor_v4.pkl"
THRESHOLD_FILE = "best_threshold_v4.pkl"
META_FILE = "model_meta_v4.json"


# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def choose_best_threshold(y_true, y_prob, min_accuracy=0.84):
    best_threshold = 0.50
    best_metrics = None
    best_recall = -1.0

    for t in np.arange(0.15, 0.71, 0.01):
        y_pred = (y_prob >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if acc >= min_accuracy and rec > best_recall:
            best_recall = rec
            best_threshold = round(float(t), 2)
            best_metrics = (acc, prec, rec, f1)

    if best_metrics is None:
        best_f1 = -1.0
        for t in np.arange(0.15, 0.71, 0.01):
            y_pred = (y_prob >= t).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(t), 2)
                best_metrics = (acc, prec, rec, f1)

    return best_threshold, best_metrics


def build_safe_input_df(base_record, selected_year, selected_month, selected_day, feature_columns):
    row = base_record.copy()

    row["Year"] = selected_year
    row["Month"] = selected_month
    row["Day"] = selected_day

    if "DayOfYear" in feature_columns:
        try:
            row["DayOfYear"] = pd.Timestamp(
                year=selected_year,
                month=selected_month,
                day=selected_day
            ).dayofyear
        except Exception:
            row["DayOfYear"] = 1

    if "Month_sin" in feature_columns:
        row["Month_sin"] = np.sin(2 * np.pi * selected_month / 12)
    if "Month_cos" in feature_columns:
        row["Month_cos"] = np.cos(2 * np.pi * selected_month / 12)

    if "Day_sin" in feature_columns or "Day_cos" in feature_columns:
        try:
            doy = pd.Timestamp(
                year=selected_year,
                month=selected_month,
                day=selected_day
            ).dayofyear
        except Exception:
            doy = 1

        if "Day_sin" in feature_columns:
            row["Day_sin"] = np.sin(2 * np.pi * doy / 365.25)
        if "Day_cos" in feature_columns:
            row["Day_cos"] = np.cos(2 * np.pi * doy / 365.25)

    for col in feature_columns:
        if col not in row.index:
            row[col] = 0

    return row[feature_columns].to_frame().T


def fix_graph(fig, height=None):
    fig.update_layout(
        paper_bgcolor="#0b1929",
        plot_bgcolor="#0b1929",
        font=dict(color="#cce7ff", family="Space Mono, monospace", size=12),
        title_font=dict(color="#00c8ff", family="Syne, sans-serif", size=16),
        legend=dict(
            bgcolor="rgba(11,25,41,0.8)",
            bordercolor="#1a3a5c",
            borderwidth=1,
            font=dict(color="#cce7ff", size=11)
        ),
        margin=dict(l=16, r=16, t=48, b=16),
    )
    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(26,58,92,0.5)",
        zeroline=False,
        tickfont=dict(color="#5b8ab0", family="Space Mono, monospace", size=11),
        title_font=dict(color="#5b8ab0"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(26,58,92,0.5)",
        zeroline=False,
        tickfont=dict(color="#5b8ab0", family="Space Mono, monospace", size=11),
        title_font=dict(color="#5b8ab0"),
    )
    return fig


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Day_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["Day_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
    if "RainToday" in df.columns:
        df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

    # Base engineered features
    if {"MaxTemp", "MinTemp"}.issubset(df.columns):
        df["TempRange"] = df["MaxTemp"] - df["MinTemp"]

    if {"Humidity9am", "Humidity3pm"}.issubset(df.columns):
        df["HumidityDiff"] = df["Humidity9am"] - df["Humidity3pm"]
        df["AvgHumidity"] = (df["Humidity9am"] + df["Humidity3pm"]) / 2

    if {"Pressure9am", "Pressure3pm"}.issubset(df.columns):
        df["PressureDiff"] = df["Pressure9am"] - df["Pressure3pm"]
        df["AvgPressure"] = (df["Pressure9am"] + df["Pressure3pm"]) / 2

    if {"WindSpeed3pm", "WindSpeed9am"}.issubset(df.columns):
        df["WindChange"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
        df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2

    if {"WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"}.issubset(df.columns):
        avg_wind = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2
        df["GustToWindRatio"] = df["WindGustSpeed"] / (avg_wind + 1)

    if {"Temp9am", "Temp3pm"}.issubset(df.columns):
        df["TempChange"] = df["Temp3pm"] - df["Temp9am"]
        df["AvgTemp"] = (df["Temp9am"] + df["Temp3pm"]) / 2

    if {"Humidity3pm", "Rainfall"}.issubset(df.columns):
        df["RainHumidityInteraction"] = df["Humidity3pm"] * (df["Rainfall"] + 1)

    if {"Pressure3pm", "Humidity3pm"}.issubset(df.columns):
        df["PressureHumidityInteraction"] = df["Pressure3pm"] * df["Humidity3pm"]


    if "Rainfall" in df.columns:
        df["Rainfall_lag1"] = df.groupby("Location")["Rainfall"].shift(1)
        df["Rainfall_lag2"] = df.groupby("Location")["Rainfall"].shift(2)
        df["Rainfall_lag3"] = df.groupby("Location")["Rainfall"].shift(3)

    if "MaxTemp" in df.columns:
        df["MaxTemp_lag1"] = df.groupby("Location")["MaxTemp"].shift(1)

    if "MinTemp" in df.columns:
        df["MinTemp_lag1"] = df.groupby("Location")["MinTemp"].shift(1)

    if "Humidity3pm" in df.columns:
        df["Humidity3pm_lag1"] = df.groupby("Location")["Humidity3pm"].shift(1)

    if "Pressure3pm" in df.columns:
        df["Pressure3pm_lag1"] = df.groupby("Location")["Pressure3pm"].shift(1)

    if "Temp3pm" in df.columns:
        df["Temp3pm_lag1"] = df.groupby("Location")["Temp3pm"].shift(1)


    if "Rainfall" in df.columns:
        df["Rainfall_roll3"] = (
            df.groupby("Location")["Rainfall"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )
        df["Rainfall_roll7"] = (
            df.groupby("Location")["Rainfall"]
            .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        )

    if "MaxTemp" in df.columns:
        df["MaxTemp_roll3"] = (
            df.groupby("Location")["MaxTemp"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

    if "MinTemp" in df.columns:
        df["MinTemp_roll3"] = (
            df.groupby("Location")["MinTemp"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

    if "Humidity3pm" in df.columns:
        df["Humidity3pm_roll3"] = (
            df.groupby("Location")["Humidity3pm"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

    if "Pressure3pm" in df.columns:
        df["Pressure3pm_roll3"] = (
            df.groupby("Location")["Pressure3pm"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

    # Drop noisy columns if too many missing values
    noisy_candidates = ["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
    drop_cols = []

    for col in noisy_candidates:
        if col in df.columns:
            missing_ratio = df[col].isna().mean()
            if missing_ratio > 0.35:
                drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.dropna(subset=["RainTomorrow"], inplace=True)

    # Remove rows made invalid because of lag features
    df = df.dropna().copy()

    return df


df = load_data()


# ------------------------------------------------
# FEATURE SPLIT
# ------------------------------------------------
X = df.drop(["RainTomorrow", "Date"], axis=1)
y = df["RainTomorrow"]

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()


# ------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, numeric),
    ("cat", cat_pipe, categorical)
])


# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = np.array([f"Feature_{i}" for i in range(X_train.shape[1])])

joblib.dump(preprocessor, PREPROCESSOR_FILE)


# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------
def train_model():
    X_dense = to_dense(X_train).astype(np.float32)

    X_sub, X_val, y_sub, y_val = train_test_split(
        X_dense,
        y_train.values,
        test_size=0.15,
        stratify=y_train.values,
        random_state=SEED
    )

    X_sub_lstm = X_sub.reshape((X_sub.shape[0], 1, X_sub.shape[1]))
    X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    model = Sequential()
    model.add(Input(shape=(1, X_dense.shape[1])))

    model.add(Bidirectional(LSTM(160, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Bidirectional(LSTM(80, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_sub.astype(int)
    )

    class_weights = {
        0: float(weights[0]),
        1: float(weights[1] * 0.85)
    }

    early_stop = EarlyStopping(
        patience=7,
        restore_best_weights=True,
        monitor="val_loss"
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=0
    )

    model.fit(
        X_sub_lstm,
        y_sub,
        epochs=40,
        batch_size=64,
        validation_data=(X_val_lstm, y_val),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )

    val_prob = model.predict(X_val_lstm, verbose=0).ravel()
    best_threshold, _ = choose_best_threshold(y_val, val_prob, min_accuracy=0.845)

    model.save(MODEL_FILE)
    joblib.dump(best_threshold, THRESHOLD_FILE)

    meta = {
        "input_dim": int(X_dense.shape[1]),
        "feature_count": int(X_dense.shape[1]),
        "seed": SEED
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return model, best_threshold


def load_or_train_model():
    current_dim = X_train.shape[1]

    if os.path.exists(MODEL_FILE):
        try:
            loaded_model = load_model(MODEL_FILE)
            input_dim = loaded_model.input_shape[-1]

            if input_dim == current_dim:
                if os.path.exists(THRESHOLD_FILE):
                    best_threshold = joblib.load(THRESHOLD_FILE)
                else:
                    best_threshold = 0.50
                return loaded_model, float(best_threshold)
        except Exception:
            pass

    return train_model()


model, best_threshold = load_or_train_model()


# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------
X_train_dense = to_dense(X_train).astype(np.float32)
X_test_dense = to_dense(X_test).astype(np.float32)

X_test_lstm = X_test_dense.reshape((X_test_dense.shape[0], 1, X_test_dense.shape[1]))

pred_prob = model.predict(X_test_lstm, verbose=0).ravel()
pred = (pred_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)
roc_auc = roc_auc_score(y_test, pred_prob)
cm = confusion_matrix(y_test, pred)


# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown(f"""
<div style="
  background: linear-gradient(135deg, rgba(0,102,255,0.12) 0%, rgba(123,47,255,0.08) 100%);
  border: 1px solid rgba(0,200,255,0.18);
  border-radius: 20px;
  padding: 28px 36px 22px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute; top: 0; right: 0;
    width: 300px; height: 200px;
    background: radial-gradient(ellipse at top right, rgba(0,102,255,0.15), transparent 70%);
    pointer-events: none;
  "></div>
  <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 6px;">
    <span style="font-size: 2.2rem;">🌧️</span>
    <h1 style="
      margin: 0;
      font-family: Syne, sans-serif;
      font-size: 1.9rem;
      font-weight: 800;
      color: #ffffff;
      letter-spacing: -0.03em;
    ">Rainfall Forecast AI</h1>
  </div>
  <p style="
    margin: 0 0 20px 0;
    font-family: Space Mono, monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5b8ab0;
  ">LSTM · LIME Explainability · Climate Intelligence · Australia Weather Dataset</p>
  <div style="display: flex; gap: 24px; flex-wrap: wrap;">
    <div style="
      background: rgba(0,200,255,0.07);
      border: 1px solid rgba(0,200,255,0.2);
      border-radius: 12px;
      padding: 12px 20px;
    ">
      <div style="font-family: Space Mono, monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #5b8ab0; margin-bottom: 4px;">Model Accuracy</div>
      <div style="font-family: Space Mono, monospace; font-size: 1.8rem; font-weight: 700; color: #00c8ff;">{accuracy*100:.2f}<span style="font-size: 1rem; color: #5b8ab0;">%</span></div>
    </div>
    <div style="
      background: rgba(0,200,255,0.07);
      border: 1px solid rgba(0,200,255,0.2);
      border-radius: 12px;
      padding: 12px 20px;
    ">
      <div style="font-family: Space Mono, monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #5b8ab0; margin-bottom: 4px;">Dataset Size</div>
      <div style="font-family: Space Mono, monospace; font-size: 1.8rem; font-weight: 700; color: #00c8ff;">{len(df):,}</div>
    </div>
    <div style="
      background: rgba(0,200,255,0.07);
      border: 1px solid rgba(0,200,255,0.2);
      border-radius: 12px;
      padding: 12px 20px;
    ">
      <div style="font-family: Space Mono, monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #5b8ab0; margin-bottom: 4px;">Locations</div>
      <div style="font-family: Space Mono, monospace; font-size: 1.8rem; font-weight: 700; color: #00c8ff;">{df['Location'].nunique()}</div>
    </div>
    <div style="
      background: rgba(0,200,255,0.07);
      border: 1px solid rgba(0,200,255,0.2);
      border-radius: 12px;
      padding: 12px 20px;
    ">
      <div style="font-family: Space Mono, monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #5b8ab0; margin-bottom: 4px;">Threshold</div>
      <div style="font-family: Space Mono, monospace; font-size: 1.8rem; font-weight: 700; color: #00c8ff;">{best_threshold:.2f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TABS
# ------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset",
    "Model Insights",
    "Forecast",
    "LIME Explainability",
    "Climate Analytics"
])


# ------------------------------------------------
# DATASET
# ------------------------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

   


# ------------------------------------------------
# MODEL INSIGHTS
# ------------------------------------------------
with tab2:
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Accuracy", f"{accuracy*100:.2f}%")
    m2.metric("Precision", f"{precision*100:.2f}%")
    m3.metric("Recall", f"{recall*100:.2f}%")
    m4.metric("F1 Score", f"{f1*100:.2f}%")
    m5.metric("ROC-AUC", f"{roc_auc*100:.2f}%")

    st.caption(f"Best threshold selected automatically: {best_threshold:.2f}")

    st.subheader("Confusion Matrix")
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No Rain", "Predicted Rain"],
        y=["Actual No Rain", "Actual Rain"],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        colorscale=[[0, "#0b1929"], [0.3, "#0f2035"], [0.6, "#0066ff"], [1, "#00c8ff"]],
        showscale=True
    ))
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label"
    )
    fig_cm = fix_graph(fig_cm, 520)
    st.plotly_chart(fig_cm, use_container_width=True, key="cm")

    fpr, tpr, _ = roc_curve(y_test, pred_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash")))

    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curve"
    )
    fig_roc = fix_graph(fig_roc, 550)
    st.plotly_chart(fig_roc, use_container_width=True, key="roc")


# ------------------------------------------------
# FORECAST
# ------------------------------------------------
with tab3:
    st.subheader("Rainfall Prediction")

    col1, col2 = st.columns(2)

    location = col1.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique()),
        key="selected_location"
    )

    selected_date = col2.date_input("Prediction Date")

    location_data = df[df["Location"] == location]
    base_record = location_data.iloc[-1].copy()

    input_df = build_safe_input_df(
        base_record=base_record.drop(labels=["RainTomorrow", "Date"]),
        selected_year=selected_date.year,
        selected_month=selected_date.month,
        selected_day=selected_date.day,
        feature_columns=X.columns
    )

    X_input = preprocessor.transform(input_df)
    X_input = to_dense(X_input).astype(np.float32)
    X_input_lstm = X_input.reshape((1, 1, X_input.shape[1]))

    prob = float(model.predict(X_input_lstm, verbose=0)[0][0])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Rainfall Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00c8ff"},
            "bgcolor": "#0b1929",
            "borderwidth": 1,
            "bordercolor": "#1a3a5c"
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0b1929",
        font=dict(color="#cce7ff", family="Space Mono, monospace", size=12),
        height=450,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    st.plotly_chart(fig, use_container_width=True, key="gauge")

    if prob >= 0.75:
        st.error("⚠️ Extreme Rainfall Risk")
    elif prob >= best_threshold:
        st.warning("🌧️ Moderate Rain Expected")
    else:
        st.success("☀️ Weather Likely Dry")

    st.subheader("7 Day Rainfall Probability Forecast")

    future_days = pd.date_range(selected_date, periods=7)
    future_probs = []

    for d in future_days:
        temp_input = build_safe_input_df(
            base_record=base_record.drop(labels=["RainTomorrow", "Date"]),
            selected_year=d.year,
            selected_month=d.month,
            selected_day=d.day,
            feature_columns=X.columns
        )

        X_temp = preprocessor.transform(temp_input)
        X_temp = to_dense(X_temp).astype(np.float32)
        X_temp = X_temp.reshape((1, 1, X_temp.shape[1]))

        p = float(model.predict(X_temp, verbose=0)[0][0])
        future_probs.append(p * 100)

    future_df = pd.DataFrame({
        "Date": future_days,
        "Rain Probability": future_probs
    })

    fig_future = px.line(
        future_df,
        x="Date",
        y="Rain Probability",
        markers=True,
        template="plotly",
        title=f"7 Day Rainfall Forecast for {location}",
        color_discrete_sequence=["#00c8ff"]
    )
    fig_future.update_layout(
        xaxis_title="Date",
        yaxis_title="Rain Probability (%)"
    )
    fig_future = fix_graph(fig_future, 480)
    st.plotly_chart(fig_future, use_container_width=True, key="future_forecast")

    st.caption("Note: Future dates reuse the latest available lag and rolling values from the selected location.")


# ------------------------------------------------
# LIME EXPLAINABILITY
# ------------------------------------------------
with tab4:
    st.subheader("Explain Prediction with LIME")

    try:
        _ = X_input
    except NameError:
        st.warning("Please go to the Forecast tab first and select a location/date to generate a prediction.")
    else:
        X_train_dense = to_dense(X_train).astype(np.float32)

        explainer = LimeTabularExplainer(
            training_data=X_train_dense,
            feature_names=feature_names,
            class_names=["No Rain", "Rain"],
            mode="classification"
        )

        def predict_fn(x):
            x = np.array(x, dtype=np.float32)
            x = x.reshape((x.shape[0], 1, x.shape[1]))
            preds = model.predict(x, verbose=0)
            return np.hstack([1 - preds, preds])

        lime_input = X_input.reshape((X_input.shape[1],))

        exp = explainer.explain_instance(
            lime_input,
            predict_fn,
            num_features=8
        )

        lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"])
        lime_df["Direction"] = lime_df["Impact"].apply(
            lambda v: "Increases Rain Chance" if v > 0 else "Decreases Rain Chance"
        )

        st.markdown("### Prediction Explanation")
        st.info(
            "This explanation shows which input features increased or decreased the predicted rain probability "
            "for the currently selected location and date."
        )

        st.markdown("### LIME Visual Explanation")
        st.pyplot(exp.as_pyplot_figure())

        st.markdown("### Feature Contribution to Prediction")
        fig_lime = px.bar(
            lime_df.sort_values("Impact"),
            x="Impact",
            y="Feature",
            color="Direction",
            orientation="h",
            template="plotly",
            title="Top Features Influencing This Prediction",
            color_discrete_map={
                "Increases Rain Chance": "#00c8ff",
                "Decreases Rain Chance": "#7b2fff"
            }
        )
        fig_lime.update_layout(
            xaxis_title="Contribution",
            yaxis_title="Feature"
        )
        fig_lime = fix_graph(fig_lime, 500)
        st.plotly_chart(fig_lime, use_container_width=True, key="lime_bar")

        pos_df = lime_df[lime_df["Impact"] > 0].sort_values("Impact", ascending=False).head(3)
        neg_df = lime_df[lime_df["Impact"] < 0].sort_values("Impact", ascending=True).head(3)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Top Factors Increasing Rain")
            if len(pos_df) > 0:
                st.dataframe(pos_df[["Feature", "Impact"]].reset_index(drop=True))
            else:
                st.write("No strong positive factors found.")

        with c2:
            st.markdown("### Top Factors Decreasing Rain")
            if len(neg_df) > 0:
                st.dataframe(neg_df[["Feature", "Impact"]].reset_index(drop=True))
            else:
                st.write("No strong negative factors found.")

        st.success(
            "LIME gives local interpretability, which means it explains this specific prediction rather than the whole model."
        )


# ------------------------------------------------
# CLIMATE ANALYTICS
# ------------------------------------------------
with tab5:
    st.subheader("Climate Analytics")

    location = st.session_state.get("selected_location", df["Location"].iloc[0])
    loc_df = df[df["Location"] == location].copy()

    yearly_avg = loc_df.groupby("Year", as_index=False)["Rainfall"].mean()

    fig1 = px.line(
        yearly_avg,
        x="Year",
        y="Rainfall",
        template="plotly",
        title=f"Rainfall Trend in {location} (Yearly Average)",
        markers=True,
        color_discrete_sequence=["#00c8ff"]
    )
    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Rainfall"
    )
    fig1 = fix_graph(fig1, 500)
    st.plotly_chart(fig1, use_container_width=True, key="trend")

    st.subheader("Weather Feature Correlation Heatmap")

    heatmap_cols = [
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm",
        "Rainfall_lag1", "Rainfall_roll3", "Rainfall_roll7"
    ]
    heatmap_cols = [c for c in heatmap_cols if c in loc_df.columns]

    if len(heatmap_cols) >= 2:
        corr_df = loc_df[heatmap_cols].corr(numeric_only=True).round(2)

        fig_heat = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[
                [0.0, "#0b1929"],
                [0.5, "#0066ff"],
                [1.0, "#00c8ff"]
            ],
            title=f"Correlation Heatmap for {location}"
        )
        fig_heat = fix_graph(fig_heat, 650)
        st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")
    else:
        st.info("Not enough numeric weather features available for correlation heatmap.")
