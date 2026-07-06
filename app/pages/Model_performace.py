#Metrik Model
from importlib.resources import path
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

#LOAD CSS
st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

css_file = Path(__file__).parent.parent / "assets" / "style.css"

with open(css_file) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

#LOAD DATASET
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "dataset",
        "textile_engineering_dataset.csv"
    )
)

#PEMISAHAN FEATURE DAN TARGET
X = df.drop(
    "defective_product",
    axis=1
)

y = df["defective_product"]

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

#LOAD MODEL
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model.pkl"
)

model = joblib.load(MODEL_PATH)

#PREDIKSI
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

#PENGHITUNG SEMUA METRIK
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

#PENGHITUNG CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    y_pred
)

#CLASSIFICATION REPORT
report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Non Defective",
        "Defective"
    ],
    output_dict=True
)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

#HEADER
st.title("📈 Model Performance")

st.write(
    "Evaluation results of the Random Forest model."
)

st.divider()

#METRIK CARDS
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )

st.divider()

st.header("📊 Confusion Matrix")

#HEATMAP 
fig = px.imshow(

    cm,

    text_auto=True,

    color_continuous_scale="Blues",

    x=["Non Defective", "Defective"],

    y=["Non Defective", "Defective"],

    title="Confusion Matrix"

)

fig.update_layout(
    xaxis_title="Predicted",
    yaxis_title="Actual"

)

st.plotly_chart(
    fig,
    use_container_width=True

)   

#CLASSIFICATION REPORT
st.divider()
st.header("📄 Classification Report")

report_df = (
    pd.DataFrame(report)
    .transpose()
)

st.dataframe(
    report_df,
    use_container_width=True
)

#ROC CURVE
st.divider()

st.header("📈 ROC Curve")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {roc_auc:.3f})"
    )
)

fig.add_trace(
    go.Scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        line=dict(dash="dash"),
        name="Random Guess"
    )
)

fig.update_layout(

    title="Receiver Operating Characteristic",

    xaxis_title="False Positive Rate",

    yaxis_title="True Positive Rate",

    height=550

)

st.plotly_chart(
    fig,
    use_container_width=True
)

#FEATURE IMPORTANCE
st.divider()

st.header("🌳 Feature Importance")

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

#=>pengurutan fitur berdasarkan nilai importance dari yang tertinggi ke terendah
importance = importance.sort_values(

    by="Importance",

    ascending=False

)

#=>Grafik Feature Importance
fig = px.bar(

    importance,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    text="Importance",

    title="Random Forest Feature Importance"

)

fig.update_traces(

    texttemplate="%{text:.3f}",

    textposition="outside"

)

fig.update_layout(

    showlegend=False,

    yaxis=dict(

        categoryorder="total ascending"

    ),

    height=550

)

st.plotly_chart(

    fig,

    use_container_width=True

)

#MODEL INFORMATION
st.divider()

st.header("ℹ️ Model Information")

col1, col2 = st.columns(2)

#left column
with col1:

    st.info("""
### Model Details

- **Model Name :** Random Forest Classifier
- **Algorithm :** Ensemble Learning
- **Number of Trees :** 100
- **Random State :** 42
""")
    
#Right column
with col2:

    st.info(f"""
### Dataset Details

- **Total Dataset :** {len(df)}
- **Training Data :** {len(X_train)}
- **Testing Data :** {len(X_test)}
- **Features :** {len(X.columns)}
- **Target Classes :** 2
""")
    
st.divider()