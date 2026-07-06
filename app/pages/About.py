import streamlit as st
from pathlib import Path

#PAGE CONFIG
st.set_page_config(

    page_title="About",
    page_icon="ℹ",
    layout="wide"

)

#LOAD CSS
css_file = Path(__file__).parent.parent / "assets" / "style.css"

with open(css_file) as f:

    st.markdown(

        f"<style>{f.read()}</style>",

        unsafe_allow_html=True

    )

#HEADER
st.markdown("""

<div class="main-title">

🏭 Textile Engineering AI

</div>

""", unsafe_allow_html=True)

st.markdown("""

<div class="subtitle">

Machine Learning Deployment using Streamlit

</div>

""", unsafe_allow_html=True)

st.divider()

#ABOUT PROJECT
st.header("📖 About Project")

st.write("""

This application is developed to predict whether a textile product
is defective or non-defective using a Machine Learning model.

The project integrates data analytics, visualization,
and predictive modeling into a web application using Streamlit.

""")

#TECHNOLOGIES STACK
st.divider()

st.header("🛠 Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:

    st.success("🐍 Python")

    st.success("📊 Pandas")

with col2:

    st.success("🤖 Scikit-Learn")

    st.success("📈 Plotly")

with col3:

    st.success("🌐 Streamlit")

    st.success("💾 Joblib")

#PROJECT STRUCTURE
st.divider()

st.header("📂 Project Modules")

modules = [

    "🏠 Home",

    "📊 Analytics",

    "🗂 Dataset",

    "📈 Model Performance",

    "🤖 Prediction"

]

for module in modules:

    st.write("✅", module)

#DEVELOPERS
st.divider()

st.header("👨‍💻 Developer")

st.info("""

**Developer**

Christa Belly Zakharia
        
**Assistant Developer**      
Noval Ferdiansyah ||
Mochamad Rizal Febriansyah ||
Fendhika Yudhistira

**Project**

Machine Learning Deployment

**Model**

Random Forest Classifier

**Year**

2026

""")

#PROJECT OBJECTIVE
st.divider()

st.header("🎯 Project Objectives")

st.markdown("""

- Predict textile product defects using Machine Learning.

- Provide interactive analytics dashboard.

- Visualize production dataset.

- Deploy Machine Learning model using Streamlit.

- Support decision making in textile manufacturing.

""")

#FOOTER
st.divider()

st.markdown("""

<div style="text-align:center;">

<h3>🏭 Textile Engineering AI</h3>

Machine Learning Deployment Project

Developed using

Python • Streamlit • Scikit-Learn • Plotly

<br><br>

© 2026 Christa Belly Zakharia

</div>

""", unsafe_allow_html=True)