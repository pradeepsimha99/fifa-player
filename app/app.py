import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="FIFA Player Rating Predictor",
    page_icon="⚽",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "models/random_forest_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚽ FIFA Predictor")

st.sidebar.markdown("""
This application predicts a player's FIFA Overall Rating
using a Machine Learning Random Forest model.

Model Performance:

✅ MAE = 0.10

✅ R² = 0.998
""")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚽ FIFA Player Rating Predictor")

st.markdown(
    """
Enter player attributes and click **Predict Rating**
to estimate the player's FIFA Overall Rating.
"""
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        min_value=16,
        max_value=45,
        value=25
    )

    height_cm = st.slider(
        "Height (cm)",
        min_value=150,
        max_value=220,
        value=180
    )

    weight_kg = st.slider(
        "Weight (kg)",
        min_value=50,
        max_value=120,
        value=75
    )

    potential = st.slider(
        "Potential",
        min_value=40,
        max_value=99,
        value=85
    )

    pace = st.slider(
        "Pace",
        min_value=1,
        max_value=99,
        value=80
    )

with col2:

    shooting = st.slider(
        "Shooting",
        min_value=1,
        max_value=99,
        value=80
    )

    passing = st.slider(
        "Passing",
        min_value=1,
        max_value=99,
        value=80
    )

    dribbling = st.slider(
        "Dribbling",
        min_value=1,
        max_value=99,
        value=80
    )

    defending = st.slider(
        "Defending",
        min_value=1,
        max_value=99,
        value=50
    )

    physic = st.slider(
        "Physic",
        min_value=1,
        max_value=99,
        value=70
    )

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

if st.button("Predict Rating"):

    player_df = pd.DataFrame([{
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "potential": potential,
        "pace": pace,
        "shooting": shooting,
        "passing": passing,
        "dribbling": dribbling,
        "defending": defending,
        "physic": physic
    }])

    prediction = model.predict(player_df)

    st.success("Prediction Complete!")

    st.metric(
        label="Predicted FIFA Overall Rating",
        value=f"{prediction[0]:.1f}"
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📊 Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.metric("Model", "Random Forest")

with info_col2:
    st.metric("R² Score", "0.998")

# --------------------------------------------------
# PLOTS
# --------------------------------------------------

st.divider()

st.subheader("📈 Exploratory Data Analysis")

plot1 = Path("plots/overall_distribution.png")
plot2 = Path("plots/random_forest_feature_importance.png")
plot3 = Path("plots/model_r2_comparison.png")

if plot1.exists():
    st.image(
        str(plot1),
        caption="Overall Rating Distribution"
    )

if plot2.exists():
    st.image(
        str(plot2),
        caption="Random Forest Feature Importance"
    )

if plot3.exists():
    st.image(
        str(plot3),
        caption="Model Comparison"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
Built using:

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Random Forest Regression
"""
)