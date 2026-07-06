import os

import pandas as pd

import streamlit as st

from pathlib import Path

#page config
st.set_page_config(

    page_title="Dataset",

    page_icon="🗂",

    layout="wide"

)

#load css
BASE_DIR = Path(__file__).resolve().parent.parent

css_file = BASE_DIR / "assets" / "style.css"

with open(css_file) as f:

    st.markdown(

        f"<style>{f.read()}</style>",

        unsafe_allow_html=True

    )

#load dataset
df = pd.read_csv(

    BASE_DIR.parent /

    "dataset" /

    "textile_engineering_dataset.csv"

)

#HEADER
st.markdown("""

# 🗂 Dataset Explorer

Explore Textile Engineering Dataset

""")

#STATISTICS
c1,c2,c3,c4 = st.columns(4)

c1.metric(

    "Rows",

    len(df)

)

c2.metric(

    "Columns",

    len(df.columns)

)

c3.metric(

    "Missing",

    df.isna().sum().sum()

)

c4.metric(

    "Duplicate",

    df.duplicated().sum()

)

#PREVIEW DATASET
st.divider()

st.subheader("Dataset Preview")

#SEARCH DATASET
search = st.text_input(
    "🔍 Search Dataset",
    placeholder="Search any value..."
)

rows = st.selectbox(
    "Rows per page",
    [10, 25, 50, 100],
    index=1
)

filtered_df = df.copy()

if search:

    mask = filtered_df.astype(str).apply(
        lambda col: col.str.contains(
            search,
            case=False,
            na=False
        )
    ).any(axis=1)

    filtered_df = filtered_df[mask]

st.dataframe(
    filtered_df.head(rows),
    use_container_width=True
)

#DATASET STATISTICS
st.divider()

st.subheader("📈 Dataset Statistics")

statistics = df.describe()

st.dataframe(
    statistics,
    use_container_width=True
)

#DATA TYPES
st.divider()

st.subheader("📋 Data Types")

datatype_df = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

st.dataframe(
    datatype_df,
    use_container_width=True
)

#MISSING VALUES ANALYSIS
st.divider()

st.subheader("❌ Missing Value Analysis")

missing_df = pd.DataFrame({

    "Feature": df.columns,

    "Missing Value": df.isnull().sum().values,

    "Percentage (%)": (
        df.isnull().sum() / len(df) * 100
    ).round(2)

})

st.dataframe(
    missing_df,
    use_container_width=True
)

#DOWNLOAD DATASET
st.divider()

st.subheader("📥 Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(

    label="⬇ Download CSV",

    data=csv,

    file_name="textile_engineering_dataset.csv",

    mime="text/csv"

)

#FEATURE DESCRIPTION
st.divider()

st.subheader("📝 Feature Description")

feature_description = pd.DataFrame({

    "Feature":[

        "machine_speed_rpm",

        "temperature_c",

        "humidity_percent",

        "vibration_level",

        "energy_usage_kwh",

        "production_count",

        "defect_count",

        "hours_since_last_maintenance",

        "maintenance_required",

        "alert_triggered",

        "defective_product"

    ],

    "Description":[

        "Machine operating speed",

        "Machine temperature",

        "Humidity around machine",

        "Machine vibration level",

        "Energy consumption",

        "Number of products produced",

        "Number of defective products",

        "Hours since last maintenance",

        "Maintenance requirement flag",

        "Machine alert status",

        "Prediction target"

    ]

})

st.dataframe(

    feature_description,

    use_container_width=True

)