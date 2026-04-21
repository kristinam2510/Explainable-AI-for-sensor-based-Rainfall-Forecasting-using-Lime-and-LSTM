Overview

This project is an end-to-end Explainable AI system for rainfall prediction using a Bidirectional LSTM model. It not only predicts whether it will rain but also explains the prediction using LIME (Local Interpretable Model-Agnostic Explanations).

The system is deployed using Streamlit for interactive forecasting, visualization, and climate analytics.

Key Features
Rainfall prediction using deep learning
Bidirectional LSTM model
Explainability using LIME
Dynamic threshold optimization to improve recall
Interactive dashboard with Streamlit
7-day rainfall forecast
Climate analytics with trend and correlation visualizations

Model Architecture
Input Layer
→ Bidirectional LSTM (160 units)
→ Batch Normalization + Dropout
→ Bidirectional LSTM (80 units)
→ Dense Layer (ReLU)
→ Output Layer (Sigmoid)
Loss Function: Binary Crossentropy
Optimizer: Adam

Dataset
Dataset: Australian Weather Dataset
Target Variable: RainTomorrow

Feature Engineering
Lag features (Rainfall, Temperature)
Rolling averages (3-day, 7-day)
Cyclical encoding (Month and Day using sin/cos)
Interaction features (e.g., Humidity × Rainfall)

Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
Threshold Optimization

Instead of using a fixed threshold of 0.5:

The model searches thresholds from 0.15 to 0.70
Selects the best threshold that maintains accuracy above 84% while maximizing recall
Explainability

LIME is used to provide local explanations for individual predictions:

Identifies features that increase rainfall probability
Identifies features that decrease rainfall probability
Visualizations
ROC Curve
Confusion Matrix Heatmap
Rainfall Trend (Year vs Rainfall)
Correlation Heatmap
7-day Forecast Plot

Project Structure
├── app.py
├── weatherAUS.csv
├── rain_lstm_model_v4.h5
├── preprocessor_v4.pkl
├── best_threshold_v4.pkl
├── model_meta_v4.json
├── README.md

How to Run
1. Clone Repository
git clone https://github.com/kristinam2510/Explainable-AI-for-sensor-based-Rainfall-Forecasting-using-Lime-and-LSTM.git
cd Explainable-AI-for-sensor-based-Rainfall-Forecasting-using-Lime-and-LSTM
2. Install Dependencies
pip install numpy==1.26.4 pandas scikit-learn tensorflow==2.15.0 keras==2.15.0 streamlit plotly lime joblib
3. Run Application
streamlit run app.py

Concepts Used
Time Series Forecasting
Deep Learning (LSTM, Bidirectional LSTM)
Feature Engineering
Class Imbalance Handling
Explainable AI

Use Cases
Weather forecasting
Flood risk analysis
Agriculture planning
Climate trend analysis

Notes
Future predictions reuse latest available lag features
Model performance depends on dataset quality
Designed for balance between interpretability and performance


## Disclaimer

This project is for **research and educational purposes only**.
