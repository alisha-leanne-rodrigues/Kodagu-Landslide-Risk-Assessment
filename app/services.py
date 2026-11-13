from pathlib import Path

import streamlit as st

from backend.features import DataStore
from backend.model_service import ModelService
from backend.support import SupportModel
from backend.explain import ShapExplainer


@st.cache_resource(show_spinner="Loading assessment services...")
def load_services():
    data_dir = str(Path(__file__).resolve().parents[1] / "data")

    store = DataStore(data_dir)
    model = ModelService(data_dir)
    support = SupportModel(data_dir, model.feature_names)
    explainer = ShapExplainer(model.model, model.feature_names)

    return store, model, support, explainer
