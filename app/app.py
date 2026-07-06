import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

#INISIALISASI SESSION STATE
if "history" not in st.session_state:

    st.session_state.history = []

#PAGE CONFIG
st.set_page_config(
    page_title="Textile Engineering AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PATH PROJECT
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "textile_engineering_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model.pkl"
)

CSS_PATH = Path(__file__).parent / "assets" / "style.css"

#lOAD CSS
with open(CSS_PATH) as css:

    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

#CACHE DATASET
@st.cache_data
def load_dataset():

    return pd.read_csv(DATASET_PATH)

#CACHE MODEL
@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)

#LOAD
df = load_dataset()

model = load_model()

#SPLIT DATA
X = df.drop(
    "defective_product",
    axis=1
)

y = df["defective_product"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.3,
    random_state=42

)

#DATASET STATISTIC
total_data = len(df)

total_feature = X.shape[1]

total_defective = (
    df["defective_product"] == 1
).sum()

total_non_defective = (
    df["defective_product"] == 0
).sum()

#SECTION HEADER
st.markdown("""
<div class="card">

<div class="main-title">

🏭 Textile Engineering AI

</div>

<div class="subtitle">

AI-powered Dashboard for Textile Defect Prediction

</div>

</div>
""", unsafe_allow_html=True)
st.write("")

#SECTION DASHBOARD METRICS
st.subheader("📊 Dashboard Overview")

# Baris pertama
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Dataset", total_data)

with m2:
    st.metric("Features", total_feature)

with m3:
    st.metric("Defective", total_defective)

with m4:
    st.metric("Non Defective", total_non_defective)

# Baris kedua
m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric("Accuracy", "100%")

with m6:
    st.metric("Model", "Random Forest")

with m7:
    st.metric("Algorithm", "Classification")

with m8:
    st.metric("Classes", "2")

st.divider()

# ==========================
# SECTION SIDEBAR
# ==========================

from pathlib import Path

logo_path = Path(__file__).parent / "assets" / "logoUntag.png"

st.sidebar.image(
    str(logo_path),
    width=130
)

st.sidebar.markdown("# 🏭 Textile Engineering AI")

st.sidebar.caption("Machine Learning Dashboard")

st.sidebar.markdown("---")

# MODEL
st.sidebar.markdown("## 🤖 Model")

st.sidebar.success("Random Forest Classifier")

# DATASET INFO
st.sidebar.markdown("## 📊 Dataset Statistics")

st.sidebar.metric(
    "Total Dataset",
    total_data
)

st.sidebar.metric(
    "Features",
    total_feature
)

st.sidebar.metric(
    "Accuracy",
    "100%"
)

st.sidebar.markdown("---")

# PROJECT INFORMATION
st.sidebar.markdown("## 📂 Project Information")

st.sidebar.info("""
**Dataset**

Textile Engineering Dataset

**Target**

Defective Product

**Algorithm**

Random Forest Classifier
""")

st.sidebar.markdown("---")

# DEVELOPER
st.sidebar.markdown("## Developer")
st.sidebar.caption("""
Christa Belly Zakharia
""")

st.sidebar.markdown("## Assistant Developer")
st.sidebar.caption("""
Noval Ferdiansyah\n            
Mochamad Rizal Febriansyah\n          
Fedhika Yudhistira
""")

st.sidebar.markdown("## Project Version")
st.sidebar.caption("""
Machine Learning Deployment\n
Version 1.0
""")

#SECTION DATA VISUALIZATION
chart1, chart2 = st.columns(2)

with chart1:

    pie_df = df.copy()

    pie_df["defective_product"] = pie_df[
        "defective_product"
    ].map({
        0: "Non Defective",

        1: "Defective"
    })

    fig = px.pie(
        pie_df,

        names="defective_product",

        title="Product Quality Distribution",

        hole=0.55
    )

    st.plotly_chart(
        fig,

        use_container_width=True
    )

with chart2:
    feature_count = pd.DataFrame({

        "Category": [

            "Defective",

            "Non Defective"
        ],

        "Total": [
            total_defective,

            total_non_defective
        ]
    })

    fig2 = px.bar(

        feature_count,

        x="Category",

        y="Total",

        color="Category",

        text="Total",

        title="Defective vs Non Defective"
    )

    fig2.update_layout(

        showlegend=False

    )

    st.plotly_chart(

        fig2,

        use_container_width=True
    )

st.divider()

#SECTION DATASET PREVIEW
st.subheader("📋 Dataset Preview")

st.dataframe(

    df.head(10),

    use_container_width=True,

    hide_index=True
)

st.divider()

#INPUT SECTION
st.markdown("""

<div class="card">

<div class="card-title">

⚙ Production Parameters

</div>

</div>

""", unsafe_allow_html=True)

left, right = st.columns(2)

with left:

    #INPUT SECTION MACHINE SPEED
    machine_speed_rpm = st.slider(
    "⚙ Machine Speed (RPM)",
    min_value=500,
    max_value=2500,
    value=1500,
    step=10
    )

    #INPUT SECTION TEMPERATURE
    temperature_c = st.slider(
    "🌡 Temperature (°C)",
    min_value=10.0,
    max_value=80.0,
    value=30.0
    )   

    #INPUT SECTION HUMIDITY 
    humidity_percent = st.slider(
    "💧 Humidity (%)",
    min_value=10.0,
    max_value=100.0,
    value=50.0
    )

    #INPUT SECTION VIBRATION
    vibration_level = st.slider(
    "📈 Vibration",
    min_value=0.0,
    max_value=10.0,
    value=2.0
    )

    #INPUT SECTION ENERGY
    energy_usage_kwh = st.slider(
    "⚡ Energy Usage",
    min_value=0.0,
    max_value=500.0,
    value=100.0
    )

with right:

    #PRODUCTION COUNT
    production_count = st.slider(
    "🏭 Production Count",
    min_value=0,
    max_value=5000,
    value=500
    )

    #DEFECT COUNT
    defect_count = st.slider(
    "❌ Defect Count",
    min_value=0,
    max_value=500,
    value=5
    )

    #MAINTENACE HOUR
    hours_since_last_maintenance = st.slider(
    "🛠 Hours Since Last Maintenance",
    min_value=0,
    max_value=1000,
    value=100
    )

    maintenance_required = st.toggle(
    "Maintenance Required"
    )
    maintenance_required = int(maintenance_required)

    alert_triggered = st.toggle(
    "Alert Triggered"
    )

    alert_triggered = int(alert_triggered)

st.divider()

# SECTION PREDICTION BUTTON
predict = st.button(

    "Predict Product",

    use_container_width=True

)

if predict:

    with st.spinner("Machine Learning Model is Predicting..."):
        
        #=>INPUT DATA
        input_data = pd.DataFrame({

            "machine_speed_rpm":[machine_speed_rpm],
            "temperature_c":[temperature_c],
            "humidity_percent":[humidity_percent],
            "vibration_level":[vibration_level],
            "energy_usage_kwh":[energy_usage_kwh],
            "production_count":[production_count],
            "defect_count":[defect_count],
            "hours_since_last_maintenance":[hours_since_last_maintenance],
            "maintenance_required":[maintenance_required],
            "alert_triggered":[alert_triggered]
        })

        #MODEL PREDICTION
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        confidence = max(probability) * 100

        #MENYIMPAN HASIL PREDIKSI KE SESSION STATE
        prediction_label = (
            "Defective"
            if prediction == 1
            else "Non Defective"
        )

        st.session_state.history.append({

            "Machine Speed": machine_speed_rpm,

            "Temperature": temperature_c,

            "Humidity": humidity_percent,

            "Prediction": prediction_label,

            "Confidence (%)": round(confidence, 2)

        })

        #HASIL PREDIKSI
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:

            if prediction == 1:

                st.error("## ❌ Product is predicted as DEFECTIVE")

                st.warning(
                    "The production parameters indicate a high probability "
                    "that this product will be defective."
                )

            else:

                st.success("## ✅ Product is predicted as NON DEFECTIVE")

                st.info(
                    "The production parameters indicate the product is likely "
                    "to meet the quality standard."
                )
        with result_col2:

            st.metric(
                "🎯 Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(confidence / 100)

            st.divider()

            st.subheader("Prediction Probability")

            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:

                st.metric(
                    "✅ Non Defective",
                    f"{probability[0]*100:.2f}%"
                )

            with prob_col2:

                st.metric(
                    "❌ Defective",
                    f"{probability[1]*100:.2f}%"
                )

            st.subheader("Input Summary")

            st.dataframe(

                input_data.T.rename(

                    columns={0: "Value"}

                ),

                use_container_width=True
            )

            st.divider()

        # FEATURE IMPORTANCE
        st.subheader("📊 Feature Importance")

        if hasattr(model, "feature_importances_"):

            feature_names = [

                "machine_speed_rpm",

                "temperature_c",

                "humidity_percent",

                "vibration_level",

                "energy_usage_kwh",

                "production_count",

                "defect_count",

                "hours_since_last_maintenance",

                "maintenance_required",

                "alert_triggered"

            ]

            importance = pd.DataFrame({

                "Feature": feature_names,

                "Importance": model.feature_importances_

            })

            importance = importance.sort_values(

                by="Importance",

                ascending=False

            )

            fig_importance = px.bar(

                importance,

                x="Importance",

                y="Feature",

                orientation="h",

                color="Importance",

                text="Importance",

                title="Random Forest Feature Importance"

            )

            fig_importance.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_importance.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

        st.divider()

        # DATASET INFORMATION
        with st.expander("📄 Dataset Information"):

            st.write(f"**Total Dataset :** {total_data}")

            st.write(f"**Training Data :** {len(X_train)}")

            st.write(f"**Testing Data :** {len(X_test)}")

            st.write(f"**Number of Features :** {total_feature}")

            st.write(f"**Defective Product :** {total_defective}")

            st.write(f"**Non Defective Product :** {total_non_defective}")

            st.write("")

            st.write("### Feature List")

            st.write(list(X.columns))

        st.divider()

        #MENAMPILKAN HISTORY PREDIKSI
        st.divider()

        st.header("📝 Prediction History")

        if len(st.session_state.history) > 0:

            history_df = pd.DataFrame(

                st.session_state.history

            )

            st.dataframe(

                history_df,

                use_container_width=True

            )

        else:

            st.info(

                "No prediction has been made yet."

            )

        #MENGHAPUS HISTORY PREDIKSI
        if st.button("🗑 Clear Prediction History"):

            st.session_state.history = []

            st.rerun()

        # FOOTER
        st.markdown(
            """
            <hr>

            <div style="text-align:center;">

            <h4>🏭 Textile Engineering AI</h4>

            Machine Learning Deployment using Streamlit

            <br><br>

            Developed by <b>Christa Belly Zakharia</b>

            <br>

            Random Forest Classifier | Scikit-Learn | Streamlit

            <br><br>

            © 2026 All Rights Reserved

            </div>

            """,
            unsafe_allow_html=True
        )