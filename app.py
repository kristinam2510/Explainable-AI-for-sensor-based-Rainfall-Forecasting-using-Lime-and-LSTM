import os
import warnings
warnings.filterwarnings("ignore")

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
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from lime.lime_tabular import LimeTabularExplainer


# ------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Explainable Rainfall Forecasting AI",
    page_icon="🌧️",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM UI THEME
# ------------------------------------------------

st.markdown("""
<style>

/* ---------------- MAIN BACKGROUND ---------------- */

.stApp {
background: linear-gradient(135deg,#ffd6e8,#ffe6f2,#f3e6ff);
color:#800000;
}

/* ---------------- GLOBAL TEXT ---------------- */

html, body, p, label {
color:#800000 !important;
}

/* ---------------- TITLES ---------------- */

h1 {
color:#c2185b !important;
font-weight:800;
}

h3 {
color:#7a004f !important;
}

/* ---------------- METRIC CARDS ---------------- */

div[data-testid="metric-container"] {
background: linear-gradient(135deg,#d6ecff,#b3daff);
padding:20px;
border-radius:14px;
border:1px solid #99ccff;
box-shadow:0px 4px 14px rgba(0,0,0,0.08);
}

[data-testid="stMetricLabel"] {
color:#660033 !important;
font-weight:700;
}

[data-testid="stMetricValue"] {
color:#ff4da6 !important;
font-size:30px;
font-weight:800;
}

[data-testid="stMetricDelta"] {
color:#800000 !important;
}

/* ---------------- TABS ---------------- */

button[data-baseweb="tab"] {
font-size:15px;
font-weight:600;
color:#6a0dad !important;
border-radius:8px;
}

button[aria-selected="true"] {
background:#f7c6e3 !important;
color:#6a0dad !important;
}

/* ---------------- TABLE HEADER ---------------- */

thead tr th {
background-color:#ffe6f7 !important;
color:#80004d !important;
}

/* ---------------- BUTTONS ---------------- */

.stButton>button {
background:#f7c6e3;
color:#5a0040;
border:1px solid #e6a8d2;
border-radius:8px;
padding:6px 16px;
font-weight:600;
}

.stButton>button:hover {
background:#f2b6d6;
}

/* ---------------- LOCATION SELECT BOX ---------------- */

.stSelectbox div[data-baseweb="select"] {
background-color:#000000 !important;
color:white !important;
border-radius:8px;
}

/* selected value */

.stSelectbox div[data-baseweb="select"] span {
color:white !important;
}

/* dropdown menu */

ul[role="listbox"] {
background-color:#000000 !important;
}

/* dropdown options */

ul[role="listbox"] li {
color:white !important;
background-color:#000000 !important;
}

/* hover option */

ul[role="listbox"] li:hover {
background-color:#222222 !important;
color:white !important;
}

/* ---------------- DATE INPUT ---------------- */

.stDateInput input {
background-color:#1e1e1e !important;
color:white !important;
border-radius:8px;
}

/* ---------------- CALENDAR ---------------- */

[data-baseweb="calendar"] {
background:#000000 !important;
color:white !important;
}

[data-baseweb="calendar"] * {
color:white !important;
}

[data-baseweb="calendar"] th {
color:white !important;
}

[data-baseweb="calendar"] td {
color:white !important;
}

[data-baseweb="calendar"] [aria-selected="true"] {
background:#333333 !important;
color:white !important;
}

/* ---------------- LABELS ---------------- */

.stSelectbox label {
color:white !important;
font-weight:600;
}

.stDateInput label {
color:white !important;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

DATASET = "weatherAUS.csv"
MODEL_FILE = "rain_lstm_model.h5"
PREPROCESSOR_FILE = "preprocessor.pkl"


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(DATASET)

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes":1,"No":0})
    df["RainToday"] = df["RainToday"].map({"Yes":1,"No":0})

    df.dropna(subset=["RainTomorrow"], inplace=True)

    return df


df = load_data()


# ------------------------------------------------
# FEATURE SPLIT
# ------------------------------------------------

X = df.drop(["RainTomorrow","Date"],axis=1)
y = df["RainTomorrow"]

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]


# ------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------

num_pipe = Pipeline([
("imputer",SimpleImputer(strategy="median")),
("scaler",StandardScaler())
])

cat_pipe = Pipeline([
("imputer",SimpleImputer(strategy="most_frequent")),
("encoder",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
("num",num_pipe,numeric),
("cat",cat_pipe,categorical)
])


# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,stratify=y,random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

joblib.dump(preprocessor,PREPROCESSOR_FILE)


# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------

def train_model():

    X_dense = X_train.toarray()
    X_lstm = X_dense.reshape((X_dense.shape[0],1,X_dense.shape[1]))

    model = Sequential()

    model.add(LSTM(64,return_sequences=True,input_shape=(1,X_dense.shape[1])))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1,activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(patience=5,restore_best_weights=True)

    model.fit(
        X_lstm,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    model.save(MODEL_FILE)

    return model


if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
else:
    model = train_model()


# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------

X_test_dense = X_test.toarray()
X_test_lstm = X_test_dense.reshape((X_test_dense.shape[0],1,X_test_dense.shape[1]))

pred_prob = model.predict(X_test_lstm)
pred = (pred_prob>0.5).astype(int)

accuracy = accuracy_score(y_test,pred)
cm = confusion_matrix(y_test,pred)


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown("""
# 🌧️ Explainable AI Rainfall Forecasting Dashboard
### Deep Learning + Climate Intelligence
""")

col1,col2 = st.columns(2)

col1.metric("Model Accuracy",f"{accuracy*100:.2f}%")
col2.metric("Dataset Size",df.shape[0])


# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
"Dataset",
"Model Insights",
"Forecast",
"LIME Explainability",
"Climate Graphs",
"Seasonal Rainfall Trend"
])


# ------------------------------------------------
# DATASET
# ------------------------------------------------

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    fig = px.histogram(
    df,
    x="Rainfall",
    nbins=50,
    title="Rainfall Distribution",
    template="plotly_white",
    color_discrete_sequence=["#ff66b3"]
)

    st.plotly_chart(fig,use_container_width=True,key="rainfall_hist")


# ------------------------------------------------
# MODEL INSIGHTS
# ------------------------------------------------

with tab2:

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)

    m1,m2,m3,m4 = st.columns(4)

    m1.metric("Accuracy",f"{accuracy*100:.2f}%")
    m2.metric("Precision",f"{precision*100:.2f}%")
    m3.metric("Recall",f"{recall*100:.2f}%")
    m4.metric("F1 Score",f"{f1*100:.2f}%")

    st.subheader("Confusion Matrix")

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No Rain","Predicted Rain"],
        y=["Actual No Rain","Actual Rain"],
        colorscale=[[0,"#fff0f6"],[0.4,"#ffb3d9"],[0.7,"#ff66b3"],[1,"#e60073"]]
    ))

    st.plotly_chart(fig_cm,use_container_width=True,key="cm")

    fpr,tpr,_ = roc_curve(y_test,pred_prob)
    roc_auc = auc(fpr,tpr)

    fig_roc = go.Figure()

    fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash")))

    fig_roc.update_layout(
    template="plotly_white",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    title="ROC Curve",
    font=dict(color="black")
)

    st.plotly_chart(fig_roc,use_container_width=True,key="roc")


# ------------------------------------------------
# FORECAST
# ------------------------------------------------

with tab3:

    st.subheader("Rainfall Prediction")

    col1,col2 = st.columns(2)

    location = col1.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique()),
        key="selected_location"
    )

    selected_date = col2.date_input("Prediction Date")

    location_data = df[df["Location"]==location]

    base_record = location_data.iloc[-1].copy()

    base_record["Year"] = selected_date.year
    base_record["Month"] = selected_date.month
    base_record["Day"] = selected_date.day

    input_df = base_record.drop(["RainTomorrow","Date"]).to_frame().T

    X_input = preprocessor.transform(input_df).toarray()

    X_input_lstm = X_input.reshape((1,1,X_input.shape[1]))

    prob = model.predict(X_input_lstm)[0][0]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Rainfall Probability"},
        gauge={'axis':{'range':[0,100]}}
    ))

    st.plotly_chart(fig,use_container_width=True,key="gauge")

    if prob>=0.7:
        st.error("⚠️ Extreme Rainfall Risk")
    elif prob>=0.5:
        st.warning("🌧️ Moderate Rain Expected")
    else:
        st.success("☀️ Weather Likely Dry")

# -----------------------------------------------
# 7 DAY FUTURE RAIN PROBABILITY
# -----------------------------------------------

    st.subheader("7 Day Rainfall Probability Forecast")

    future_days = pd.date_range(selected_date, periods=7)

    future_probs = []

    for d in future_days:

        base_record["Year"] = d.year
        base_record["Month"] = d.month
        base_record["Day"] = d.day

        temp_input = base_record.drop(["RainTomorrow","Date"]).to_frame().T

        X_temp = preprocessor.transform(temp_input).toarray()
        X_temp = X_temp.reshape((1,1,X_temp.shape[1]))

        p = model.predict(X_temp)[0][0]

        future_probs.append(p*100)

    future_df = pd.DataFrame({
        "Date":future_days,
        "Rain Probability":future_probs
    })

    fig_future = px.line(
        future_df,
        x="Date",
        y="Rain Probability",
        markers=True,
        template="plotly_white",
        color_discrete_sequence=["#ff66b3","#9966ff","#6699ff"]
    )

    st.plotly_chart(fig_future,use_container_width=True,key="future_forecast")

# -----------------------------------------------
# FUTURE RAIN RISK TIMELINE
# -----------------------------------------------

    st.subheader("Rainfall Risk Timeline")

    fig_timeline = px.bar(
        future_df,
        x="Date",
        y="Rain Probability",
        template="plotly_white",
        color="Rain Probability",
        color_continuous_scale=["#ff66b3","#9966ff","#6699ff"]
    )

    st.plotly_chart(fig_timeline,use_container_width=True,key="timeline")
# ------------------------------------------------
# LIME
# ------------------------------------------------

with tab4:

    st.subheader("Explain Prediction with LIME")

    X_train_dense = X_train.toarray()

    explainer = LimeTabularExplainer(
        X_train_dense,
        feature_names=feature_names,
        class_names=["No Rain","Rain"],
        mode="classification"
    )

    def predict_fn(x):

        x = np.array(x)
        x = x.reshape((x.shape[0],1,x.shape[1]))

        preds = model.predict(x)

        return np.hstack([1-preds,preds])

    lime_input = X_input.reshape((X_input.shape[1],))

    exp = explainer.explain_instance(
        lime_input,
        predict_fn,
        num_features=10
    )

    st.pyplot(exp.as_pyplot_figure())

    factor_df = pd.DataFrame(exp.as_list(),columns=["Feature","Impact"])
    st.dataframe(factor_df)


# ------------------------------------------------
# CLIMATE GRAPHS
# ------------------------------------------------

with tab5:

    location = st.session_state.get("selected_location", df["Location"].iloc[0])

    loc_df = df[df["Location"]==location]

    fig1 = px.line(
        loc_df,
        x="Year",
        y="Rainfall",
        template="plotly_white",
        title=f"Rainfall Trend in {location}",
        color_discrete_sequence=["#ff66b3","#9966ff","#6699ff"]
    )

    st.plotly_chart(fig1,use_container_width=True,key="trend")

    fig2 = px.scatter(
    loc_df,
    x="MaxTemp",
    y="Rainfall",
    color="RainTomorrow",
    template="plotly_white",
    color_discrete_sequence=["#ff66b3","#9966ff"]
)

    st.plotly_chart(fig2,use_container_width=True,key="scatter")
    # -----------------------------------------------
    # INTERACTIVE CLIMATE MAP
    # -----------------------------------------------

    st.subheader("Rainfall Intensity Across Locations")

    rain_map = df.groupby("Location")["Rainfall"].mean().reset_index()

    fig_map = px.scatter(
        rain_map,
        x="Location",
        y="Rainfall",
        size="Rainfall",
        color="Rainfall",
        template="plotly_white",
        color_continuous_scale=["#ff66b3","#9966ff","#6699ff"]
    )

    st.plotly_chart(fig_map,use_container_width=True,key="climate_map")

# ------------------------------------------------
# SEASONAL TREND
# ------------------------------------------------

with tab6:

    st.subheader("Weather Analytics Dashboard")

    location = st.session_state.get("selected_location", df["Location"].iloc[0])

    loc_df = df[df["Location"] == location]

    c1, c2 = st.columns(2)

    monthly_avg = loc_df.groupby("Month")["Rainfall"].mean().reset_index()

    fig1 = px.line(
        monthly_avg,
        x="Month",
        y="Rainfall",
        markers=True,
        template="plotly_white",
        title=f"Average Monthly Rainfall in {location}",
        color_discrete_sequence=["#ff66b3","#9966ff","#6699ff"]
    )

    c1.plotly_chart(fig1, use_container_width=True,key="monthly")

    rain_counts = loc_df["RainTomorrow"].value_counts().reset_index()
    rain_counts.columns = ["RainTomorrow","Count"]

    fig2 = px.pie(
        rain_counts,
        names="RainTomorrow",
        values="Count",
        template="plotly_white",
        title="Rain vs No Rain Distribution",
        color_discrete_sequence=["#ff66b3","#9966ff"]
    )

    c2.plotly_chart(fig2,use_container_width=True,key="pie")

    st.subheader("Top 10 Rainiest Locations")

    rain_loc = df.groupby("Location")["Rainfall"].mean().reset_index()
    rain_loc = rain_loc.sort_values("Rainfall",ascending=False).head(10)

    fig3 = px.bar(
        rain_loc,
        x="Location",
        y="Rainfall",
        template="plotly_white",
        title="Top 10 Locations with Highest Average Rainfall",
        color_discrete_sequence=["#cc66ff"]
    )

    st.plotly_chart(fig3,use_container_width=True,key="top_locations")

    st.info(
        "This dashboard highlights seasonal rainfall patterns, rain occurrence frequency, "
        "and the locations with the highest rainfall levels."
    )