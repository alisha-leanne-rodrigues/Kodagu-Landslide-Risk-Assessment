"""Location-based landslide susceptibility decision support."""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.map_view import (
    create_base_map,
    add_location_marker,
    display_map,
)
from app.services import load_services
from backend.geo import lonlat_to_rowcol
from backend.features import extract_pixels
from backend.feature_meta import label


def get_score_category(score):
    if score < 0.15:
        return "Low", "The model assigns a relatively low landslide score."
    if score < 0.25:
        return "Medium", "The model identifies an intermediate level of susceptibility."
    return "High", "The model assigns a relatively high landslide score."


def get_decision_guidance(category, support_level):
    if category == "High":
        message = (
            "Prioritise this location for closer review. Consider field "
            "verification of slope conditions, drainage and nearby land-use "
            "changes before making a site-level decision."
        )
    elif category == "Medium":
        message = (
            "Include this location in targeted monitoring. Review local "
            "terrain, drainage and recent environmental changes, especially "
            "if other evidence indicates instability."
        )
    else:
        message = (
            "The model score is relatively low. Retain routine monitoring "
            "and consider other local evidence when planning land use."
        )

    if support_level == 2:
        message += (
            " Evidence coverage is limited at this location, so give priority "
            "to checking the local conditions before relying on the model result."
        )
    elif support_level == 1:
        message += (
            " Training-data coverage is partial; interpret the result "
            "alongside local evidence."
        )

    return message


def render_location_assessment():

    st.title("Location Assessment")
    st.caption(
        "Assess landslide susceptibility at a selected location in Kodagu "
        "and review the evidence behind the model result."
    )

    st.markdown(
        "Click anywhere on the map to select a location within Kodagu."
    )

    selected_location = st.session_state.get("selected_location")
    map_view = create_base_map()

    if selected_location:
        map_view = add_location_marker(
            map_view,
            selected_location["latitude"],
            selected_location["longitude"],
        )

    map_data = display_map(map_view, height=550)

    if map_data and map_data.get("last_clicked"):
        clicked_location = map_data["last_clicked"]
        latitude = float(clicked_location["lat"])
        longitude = float(clicked_location["lng"])

        st.session_state["selected_location"] = {
            "latitude": latitude,
            "longitude": longitude,
        }
        selected_location = st.session_state["selected_location"]

    st.divider()
    st.subheader("Selected Location")

    if not selected_location:
        st.info("No location selected. Click on the map to begin.")
        return

    latitude = selected_location["latitude"]
    longitude = selected_location["longitude"]

    col1, col2 = st.columns(2)
    col1.metric("Latitude", f"{latitude:.6f}")
    col2.metric("Longitude", f"{longitude:.6f}")

    st.divider()
    st.subheader("Susceptibility Assessment")

    with st.spinner("Assessing the selected location..."):
        try:
            store, model, support, explainer = load_services()

            row_float, col_float = lonlat_to_rowcol(longitude, latitude)
            row = int(np.floor(float(row_float)))
            col = int(np.floor(float(col_float)))

            rows = np.array([row])
            cols = np.array([col])

            if not store.in_window(rows, cols)[0]:
                st.error(
                    "This location is outside the prepared Kodagu "
                    "assessment raster."
                )
                return

            if not store.district_mask_at(rows, cols)[0]:
                st.error(
                    "This location is outside the Kodagu district boundary."
                )
                return

            X, valid = extract_pixels(
                store,
                rows,
                cols,
                model.feature_names,
            )

            if not valid[0]:
                st.error(
                    "Environmental data is incomplete at this location. "
                    "An assessment cannot be generated."
                )
                return

            scores = model.predict_scores(X)
            score = float(scores[0])
            prediction = int(model.classify(scores)[0])

            shap_values, base_value = explainer.explain(X)
            support_result = support.assess(rows, cols, X)

        except Exception as error:
            st.error(f"Assessment failed: {error}")
            return

    category, category_description = get_score_category(score)

    st.success("Location assessment completed.")

    score_col, category_col, decision_col = st.columns(3)
    score_col.metric("Model score", f"{score:.4f}")
    category_col.metric("Score category", category)
    decision_col.metric(
        "Deployed classification",
        "Positive" if prediction == 1 else "Negative",
    )

    st.write(category_description)

    st.caption(
        "The score is the Random Forest's mean tree probability, not a "
        "calibrated real-world probability of a landslide. The deployed "
        f"binary classification threshold remains {model.threshold:.2f}."
    )

    with st.expander("How are the score categories defined?"):
        st.markdown(
            """
            | Category | Model score |
            |---|---:|
            | Low | Below 0.15 |
            | Medium | 0.15 to below 0.25 |
            | High | 0.25 or above |

            These screening categories are based on validation results.
            They are not calibrated probabilities or official hazard classes.
            """
        )

        st.write(
            "Observed landslide-labelled samples in the validation data:"
        )

        validation_evidence = pd.DataFrame({
            "Category": ["Low", "Medium", "High"],
            "Landslide-labelled samples": [38, 77, 327],
            "Total samples": [679, 310, 588],
            "Observed proportion": ["5.6%", "24.8%", "55.6%"],
        })

        st.dataframe(
            validation_evidence,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These proportions describe the validation dataset. They do "
            "not represent the real-world probability of a future landslide."
        )

    st.divider()
    st.subheader("Decision Guidance")

    guidance = get_decision_guidance(
        category,
        int(support_result["level"][0]),
    )

    if category == "High":
        st.warning(guidance)
    elif category == "Medium":
        st.info(guidance)
    else:
        st.success(guidance)

    st.divider()
    st.subheader("Training Data Evidence")

    level_names = {
        0: "Well supported",
        1: "Partially supported",
        2: "Limited support / extrapolation",
    }

    support_level = int(support_result["level"][0])
    distance = float(support_result["distance_km"][0])
    out_count = int(support_result["n_out_of_range"][0])

    support_col1, support_col2, support_col3 = st.columns(3)

    support_col1.metric(
        "Evidence coverage",
        level_names[support_level],
    )
    support_col2.metric(
        "Distance to nearest training landslide",
        f"{distance:.2f} km",
    )
    support_col3.metric(
        "Features outside training range",
        out_count,
    )

    if out_count:
        out_names = support.out_of_range_names(
            support_result["oor"][0]
        )
        st.write("Features outside the training range:")
        st.write(", ".join(label(name) for name in out_names))
    else:
        st.write(
            "All continuous features fall within their training ranges."
        )

    st.caption(
        "Evidence coverage measures proximity to training landslide pixels "
        "and whether continuous feature values fall within training ranges. "
        "It is a heuristic indicator, not a statistical confidence interval."
    )

    st.divider()
    st.subheader("Why Did the Model Produce This Score?")

    st.write(
        "SHAP explains how each feature contributed to the model score "
        "relative to the model's expected score."
    )

    contributions = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": [
            label(name) for name in model.feature_names
        ],
        "Feature value": X[0],
        "SHAP contribution": contributions,
    })

    explanation["Absolute contribution"] = (
        explanation["SHAP contribution"].abs()
    )

    increasing = (
        explanation[explanation["SHAP contribution"] > 0]
        .sort_values("Absolute contribution", ascending=False)
        .head(5)
        .drop(columns=["Absolute contribution"])
    )

    decreasing = (
        explanation[explanation["SHAP contribution"] < 0]
        .sort_values("Absolute contribution", ascending=False)
        .head(5)
        .drop(columns=["Absolute contribution"])
    )

    increase_col, decrease_col = st.columns(2)

    with increase_col:
        st.markdown("**Features increasing the model score**")
        if increasing.empty:
            st.write("No positive SHAP contributions for this location.")
        else:
            st.dataframe(
                increasing.round(5),
                hide_index=True,
                use_container_width=True,
            )

    with decrease_col:
        st.markdown("**Features decreasing the model score**")
        if decreasing.empty:
            st.write("No negative SHAP contributions for this location.")
        else:
            st.dataframe(
                decreasing.round(5),
                hide_index=True,
                use_container_width=True,
            )

    st.caption(
        "A positive contribution pushes the model score upward; a negative "
        "contribution pushes it downward. These are model associations, "
        "not proof that a feature physically causes or prevents landslides."
    )

    with st.expander("View all feature contributions"):
        all_features = explanation.sort_values(
            "Absolute contribution",
            ascending=False,
        ).drop(columns=["Absolute contribution"])

        st.dataframe(
            all_features.round(5),
            hide_index=True,
            use_container_width=True,
        )

    st.write(f"SHAP expected score: **{base_value:.5f}**")
    st.caption(f"Explanation engine: {explainer.engine}")
