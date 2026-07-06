import os
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

# ===============================
# LOAD CSS
# ===============================

css_file = Path(__file__).parent.parent / "assets" / "style.css"

with open(css_file) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ===============================
# LOAD DATASET
# ===============================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

df = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "dataset",
        "textile_engineering_dataset.csv"
    )
)

# ===============================
# HEADER
# ===============================

st.title("📊 Dataset Analytics")

st.write(
    "Visualisasi dan analisis dataset Textile Engineering."
)

# ===============================
# METRICS
# ===============================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Dataset",
    len(df)
)

c2.metric(
    "Features",
    df.shape[1]-1
)

c3.metric(
    "Defective",
    (df["defective_product"]==1).sum()
)

c4.metric(
    "Non Defective",
    (df["defective_product"]==0).sum()
)

#PIE CHART
st.divider()

col1, col2 = st.columns(2)

with col1:

    pie_df = df.copy()

    pie_df["Quality"] = pie_df["defective_product"].map({
        0: "Non Defective",
        1: "Defective"
    })

    fig = px.pie(
        pie_df,
        names="Quality",
        hole=0.65,
        color="Quality",
        color_discrete_map={
            "Non Defective": "#00C853",
            "Defective": "#FF5252"
        },
        title="Product Quality Distribution"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

#BAR CHART
count_df = pie_df["Quality"].value_counts().reset_index()

count_df.columns = ["Quality","Count"]

fig = px.bar(
    count_df,
    x="Quality",
    y="Count",
    color="Quality",
    text="Count",
    color_discrete_map={
        "Non Defective":"#00C853",
        "Defective":"#FF5252"
    },
    title="Number of Products"
)

fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    showlegend=False

)

fig.update_traces(

    textposition="outside"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

#HISTOGRAM INTERAKTIF
st.divider()

st.subheader("📈 Feature Distribution")

feature = st.selectbox(

    "Choose Feature",

    [

        "machine_speed_rpm",

        "temperature_c",

        "humidity_percent",

        "vibration_level",

        "energy_usage_kwh",

        "production_count",

        "defect_count",

        "hours_since_last_maintenance"

    ]

)

fig = px.histogram(

    df,

    x=feature,

    nbins=35,

    color_discrete_sequence=["#4FC3F7"]

)

fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

#CORELATION HEATMAP
st.divider()

st.subheader("🔥 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig = px.imshow(

    corr,

    text_auto=".2f",

    aspect="auto",

    color_continuous_scale="RdBu_r"

)

fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

#BOXPLOT (OUTLIER DETECTION)
st.divider()

st.subheader("📦 Outlier Detection")

box_feature = st.selectbox(

    "Feature",

    [

        "machine_speed_rpm",

        "temperature_c",

        "humidity_percent",

        "vibration_level",

        "energy_usage_kwh",

        "production_count",

        "defect_count",

        "hours_since_last_maintenance"

    ],

    key="box"

)

fig = px.box(

    df,

    y=box_feature,

    color_discrete_sequence=["#7E57C2"]

)

fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)"

)

st.plotly_chart(

    fig,

    use_container_width=True
)

#DATASET EXPLORER
st.divider()

st.header("📋 Dataset Explorer")

#FILTER DATA AFTER EXPLORE
filter_option = st.selectbox(
    "Filter Product Quality",
    [
        "All",
        "Non Defective",
        "Defective"
    ]
)

#SEARCH DATA
search = st.text_input(
    "🔍 Search Data"
)

rows = st.slider(
    "Rows to Display",
    min_value=5,
    max_value=100,
    value=10
)

#FILTER DATASET
filtered_df = df.copy()

if filter_option == "Defective":
    filtered_df = filtered_df[
        filtered_df["defective_product"] == 1
    ]

elif filter_option == "Non Defective":
    filtered_df = filtered_df[
        filtered_df["defective_product"] == 0
    ]

#SEARCH SELURUH KOLOM
if search != "":

    filtered_df = filtered_df[
        filtered_df.astype(str)
        .apply(
            lambda x: x.str.contains(
                search,
                case=False
            )
        )
        .any(axis=1)
    ]

st.dataframe(
    filtered_df.head(rows),
    use_container_width=True
)

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv,
    file_name="textile_dataset.csv",
    mime="text/csv"
)

#STATISIAL SUMMARY
st.divider()

st.subheader("📋 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

#MISSING VALUE CHECKER
st.divider()

st.subheader("🛠 Missing Value Checker")

missing = df.isnull().sum()

if missing.sum() == 0:

    st.success("✅ No Missing Values Found")

else:

    st.dataframe(missing)

#DATASET PREVIEW
st.divider()

st.subheader("📄 Dataset Preview")

rows = st.slider(
    "Number of Rows",
    5,
    50,
    10
)

st.dataframe(
    df.head(rows),
    use_container_width=True
)