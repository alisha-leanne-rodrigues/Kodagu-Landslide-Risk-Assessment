import streamlit as st
from app.styles import apply_styles
from app.views.location_assessment import render_location_assessment
from app.views.area_assessment import render_area_assessment

st.set_page_config(
    page_title="Kodagu Landslide Assessment",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

st.sidebar.title("⛰️ Kodagu Landslide")
st.sidebar.caption("Decision Support System")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Location Assessment",
        "Area Assessment",
        "Model Information",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Study Area: Kodagu District, Karnataka")


if page == "Dashboard":

    st.title("Kodagu Landslide Assessment")
    st.caption(
        "Explainable AI-Based Decision Making System "
        "| Western Ghats, Karnataka"
    )

    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg, #183B32, #246B50);
            padding: 28px;
            border-radius: 14px;
            margin: 20px 0;
            color: white;
        ">
            <h2 style="color:white; margin:0 0 10px 0;">
                Understanding Landslide Susceptibility
            </h2>
            <p style="color:#E5F0E8; font-size:16px; margin:0;">
                Explore environmental conditions, assess geographical
                locations and understand the factors influencing
                landslide susceptibility across Kodagu district.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Study Area", "Kodagu")
    col2.metric("Prediction Model", "Random Forest")
    col3.metric("Input Features", "24")
    col4.metric("Decision Threshold", "0.25")

    st.divider()

    st.subheader("Assessment Tools")
    st.write(
        "Choose an assessment method to explore landslide "
        "susceptibility."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="
                background:white;
                padding:22px;
                border-radius:12px;
                border:1px solid #E1E8E2;
                min-height:170px;
            ">
                <h3>📍 Location Assessment</h3>
                <p style="color:#52645A;">
                    Select a geographical location on the interactive
                    map and evaluate its landslide susceptibility.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                background:white;
                padding:22px;
                border-radius:12px;
                border:1px solid #E1E8E2;
                min-height:170px;
            ">
                <h3>🗺️ Area Assessment</h3>
                <p style="color:#52645A;">
                    Draw a geographical polygon on the map to assess
                    landslide susceptibility across a selected area.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("How the System Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("**01 — Select Location**")
        st.write(
            "Choose a location or draw an area within Kodagu district."
        )

    with step2:
        st.markdown("**02 — Analyse Environmental Factors**")
        st.write(
            "The system uses geographical and environmental features "
            "to generate predictions."
        )

    with step3:
        st.markdown("**03 — Understand the Results**")
        st.write(
            "View susceptibility predictions and SHAP-based "
            "explanations of influential factors."
        )


elif page == "Location Assessment":
    render_location_assessment()


elif page == "Area Assessment":
    render_area_assessment()


elif page == "Model Information":

    st.title("Model Information")
    st.caption("Details of the deployed machine learning model.")

    st.subheader("Prediction Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Algorithm", "Random Forest")
        st.metric("Number of Features", "24")

    with col2:
        st.metric("Number of Estimators", "500")
        st.metric("Classification Threshold", "0.25")

    st.divider()

    st.subheader("Explainability")

    st.write(
        """
        SHAP (SHapley Additive exPlanations) is used to interpret
        model predictions by identifying how individual environmental
        features contribute to the predicted susceptibility.
        """
    )

    st.subheader("Prediction Output")

    st.write(
        """
        The model generates a landslide susceptibility probability.
        A classification threshold of 0.25 is used to determine
        the predicted class.
        """
    )
