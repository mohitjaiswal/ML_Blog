# ML Metrics Interactive Showcase

import streamlit as st
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="ML Metrics Playground",
    page_icon="📊",
    layout="centered",
)


# ---------------------------------------------------------
# Title and introduction
# ---------------------------------------------------------

st.title("📊 ML Metrics Playground")

st.write(
    "Explore how different regression evaluation metrics "
    "respond when model predictions change."
)

st.info(
    "💡 Try changing one prediction at a time and observe "
    "how MAE, MSE, RMSE and R² respond."
)


# ---------------------------------------------------------
# Actual values
# ---------------------------------------------------------

st.subheader("1️⃣ Actual Values")

y_true = [10, 20, 30, 40, 50]

st.write(y_true)


# ---------------------------------------------------------
# Prediction inputs
# ---------------------------------------------------------

st.subheader("2️⃣ Model Predictions")

st.write(
    "These values represent predictions produced by a model. "
    "Change them to simulate better or worse predictions."
)


# Default prediction values
default_predictions = [12.0, 18.0, 33.0, 37.0, 48.0]


# Reset button
if st.button("🔄 Reset Predictions"):
    st.session_state.pred_1 = default_predictions[0]
    st.session_state.pred_2 = default_predictions[1]
    st.session_state.pred_3 = default_predictions[2]
    st.session_state.pred_4 = default_predictions[3]
    st.session_state.pred_5 = default_predictions[4]


# Prediction inputs
pred_1 = st.number_input(
    "Prediction 1",
    value=default_predictions[0],
    key="pred_1",
)

pred_2 = st.number_input(
    "Prediction 2",
    value=default_predictions[1],
    key="pred_2",
)

pred_3 = st.number_input(
    "Prediction 3",
    value=default_predictions[2],
    key="pred_3",
)

pred_4 = st.number_input(
    "Prediction 4",
    value=default_predictions[3],
    key="pred_4",
)

pred_5 = st.number_input(
    "Prediction 5",
    value=default_predictions[4],
    key="pred_5",
)


y_predicted = [
    pred_1,
    pred_2,
    pred_3,
    pred_4,
    pred_5,
]


# ---------------------------------------------------------
# Calculate evaluation metrics
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_true,
    y_predicted,
)

mse = mean_squared_error(
    y_true,
    y_predicted,
)

rmse = mse ** 0.5

r2 = r2_score(
    y_true,
    y_predicted,
)


# ---------------------------------------------------------
# Display metrics
# ---------------------------------------------------------

st.subheader("3️⃣ Evaluation Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "MAE",
    f"{mae:.4f}",
)

col2.metric(
    "MSE",
    f"{mse:.4f}",
)

col3.metric(
    "RMSE",
    f"{rmse:.4f}",
)

col4.metric(
    "R²",
    f"{r2:.4f}",
)


# ---------------------------------------------------------
# Actual vs Predicted chart
# ---------------------------------------------------------

st.subheader("4️⃣ Actual vs Predicted")

chart_data = {
    "Actual": y_true,
    "Predicted": y_predicted,
}

st.line_chart(chart_data)


# ---------------------------------------------------------
# Current interpretation
# ---------------------------------------------------------

st.subheader("5️⃣ What Do These Results Tell Us?")

st.write(
    "MAE shows the average absolute prediction error. "
    "MSE gives greater penalty to large errors. "
    "RMSE is the square root of MSE and is expressed in "
    "the same units as the target. R² indicates how well "
    "the predictions explain the variation in the actual values."
)

st.caption(
    "Experiment: change one prediction at a time and observe "
    "how the metrics and chart respond."
)