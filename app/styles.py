import streamlit as st


def apply_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F5F7F4;
        }

        [data-testid="stSidebar"] {
            background-color: #183B32;
        }

        [data-testid="stSidebar"] * {
            color: #FFFFFF;
        }

        h1, h2, h3 {
            color: #183B32;
        }

        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E1E8E2;
            padding: 18px;
            border-radius: 12px;
        }

        [data-testid="stMetricValue"] {
            color: #246B50;
        }

        .stButton > button {
            background-color: #246B50;
            color: white;
            border-radius: 8px;
            border: none;
        }

        .stButton > button:hover {
            background-color: #183B32;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
