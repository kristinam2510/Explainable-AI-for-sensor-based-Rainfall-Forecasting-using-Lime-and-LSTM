# Explainable AI for Sensor-Based Rainfall Forecasting using LIME and LSTM

## Overview

This project predicts rainfall using **sensor-based weather data** and deep learning techniques. A **Long Short-Term Memory (LSTM)** model is used to analyze time-series environmental data such as temperature, humidity, pressure, and other atmospheric conditions.

To improve model transparency, the system integrates **Explainable AI (XAI)** using **LIME (Local Interpretable Model-Agnostic Explanations)**, which explains how different features influence rainfall predictions.

## Features

* Rainfall prediction using **LSTM neural networks**
* Processing of **sensor-based weather data**
* **Explainable AI using LIME**
* Interactive visualization using **Plotly**
* Web interface built with **Streamlit**

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy & Pandas
* Scikit-learn
* Plotly
* Streamlit
* LIME (Explainable AI)

## Project Workflow

1. Load and preprocess sensor-based weather dataset
2. Train an **LSTM model** for rainfall prediction
3. Evaluate model performance using accuracy and confusion matrix
4. Generate **LIME explanations** to understand feature importance
5. Display results through a **Streamlit interactive dashboard**

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Applications

* Weather forecasting systems
* Smart agriculture and irrigation planning
* Climate monitoring and environmental research

## Disclaimer

This project is for **research and educational purposes only**.
