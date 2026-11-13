"""Area-based landslide susceptibility and decision-support assessment."""

import numpy as np
import pandas as pd
import streamlit as st

from app.components.map_view import create_base_map, add_polygon_drawing, display_map
from app.services import load_services
from backend.features import extract_window, valid_mask
from backend.geo import lonlat_to_rowcol, rasterize_rings, ring_is_valid
from backend.feature_meta import label

LOW_LIMIT = 0.15
ROW_CHUNK_SIZE = 128
PIXEL_AREA_M2 = 900.0
SHAP_SAMPLE_LIMIT = 256


def _prepare_polygons(drawings, store):
    polygons = []

    for drawing in drawings:
        geometry = drawing.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        if geometry_type == "Polygon":
            components = [coordinates]
        elif geometry_type == "MultiPolygon":
            components = coordinates
        else:
            continue

        for component in components:
            if not component:
                continue

            outer = np.asarray(component[0], dtype=float)
            if not ring_is_valid(outer):
                continue

            holes = [
                np.asarray(ring, dtype=float)
                for ring in component[1:]
                if ring_is_valid(np.asarray(ring, dtype=float))
            ]

            rows, cols = lonlat_to_rowcol(outer[:, 0], outer[:, 1])
            if not np.isfinite(rows).all() or not np.isfinite(cols).all():
                continue

            r0 = max(store.r0, int(np.floor(rows.min())) - 1)
            r1 = min(store.r1, int(np.ceil(rows.max())) + 1)
            c0 = max(store.c0, int(np.floor(cols.min())) - 1)
            c1 = min(store.c1, int(np.ceil(cols.max())) + 1)

            if r1 > r0 and c1 > c0:
                polygons.append({
                    "outer": outer, "holes": holes,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                })

    return polygons


def _assess_area(drawings, store, model, support, explainer):
    polygons = _prepare_polygons(drawings, store)
    if not polygons:
        raise ValueError("No valid polygon overlaps the prepared Kodagu assessment area.")

    r0 = min(p["r0"] for p in polygons)
    r1 = max(p["r1"] for p in polygons)
    c0 = min(p["c0"] for p in polygons)
    c1 = max(p["c1"] for p in polygons)

    selected_pixels = valid_pixels = positive_count = 0
    score_sum = 0.0
    score_min, score_max = float("inf"), float("-inf")
    category_counts = np.zeros(3, dtype=np.int64)

    support_counts = np.zeros(3, dtype=np.int64)
    distance_sum = 0.0
    distance_min, distance_max = float("inf"), 0.0
    oor_counts = np.zeros(len(support.cont_idx), dtype=np.int64)

    shap_features = []

    for row_start in range(r0, r1, ROW_CHUNK_SIZE):
        row_end = min(row_start + ROW_CHUNK_SIZE, r1)
        selected_mask = np.zeros((row_end - row_start, c1 - c0), dtype=bool)

        for polygon in polygons:
            if polygon["r1"] <= row_start or polygon["r0"] >= row_end:
                continue

            selected_mask |= rasterize_rings(
                [polygon["outer"]], row_start, row_end, c0, c1,
                holes_lonlat=polygon["holes"],
            )

        if not selected_mask.any():
            continue

        features = extract_window(store, row_start, row_end, c0, c1)
        inside_mask = selected_mask & features["inside_district"]
        selected_pixels += int(inside_mask.sum())

        assessment_mask = selected_mask & valid_mask(features, model.feature_names)
        if not assessment_mask.any():
            continue

        X = np.column_stack([
            features[name][assessment_mask] for name in model.feature_names
        ]).astype(np.float64)

        scores = model.predict_scores(X)
        valid_pixels += len(scores)
        score_sum += float(scores.sum())
        score_min = min(score_min, float(scores.min()))
        score_max = max(score_max, float(scores.max()))

        category_counts += np.array([
            np.count_nonzero(scores < LOW_LIMIT),
            np.count_nonzero((scores >= LOW_LIMIT) & (scores < model.threshold)),
            np.count_nonzero(scores >= model.threshold),
        ])
        positive_count += int(np.count_nonzero(model.classify(scores)))

        local_rows, local_cols = np.where(assessment_mask)
        global_rows = local_rows + row_start
        global_cols = local_cols + c0

        support_result = support.assess(global_rows, global_cols, X)
        support_counts += np.bincount(support_result["level"], minlength=3)

        distances = support_result["distance_km"]
        distance_sum += float(distances.sum())
        distance_min = min(distance_min, float(distances.min()))
        distance_max = max(distance_max, float(distances.max()))
        oor_counts += support_result["oor"].sum(axis=0)

        # Retain a bounded candidate sample from each raster chunk.
        sample_count = min(SHAP_SAMPLE_LIMIT, len(X))
        sample_indices = np.linspace(0, len(X) - 1, sample_count, dtype=int)
        shap_features.append(X[sample_indices])

    if selected_pixels == 0:
        raise ValueError("The drawn area does not contain raster pixels inside Kodagu.")
    if valid_pixels == 0:
        raise ValueError("No selected Kodagu pixels have complete environmental data.")

    candidate_X = np.vstack(shap_features)
    shap_limit = min(SHAP_SAMPLE_LIMIT, explainer.max_rows, len(candidate_X))
    shap_indices = np.linspace(0, len(candidate_X) - 1, shap_limit, dtype=int)
    shap_X = candidate_X[shap_indices]

    shap_values, base_value = explainer.explain(shap_X)

    return {
        "polygon_count": len(polygons),
        "selected_pixels": selected_pixels,
        "valid_pixels": valid_pixels,
        "missing_pixels": selected_pixels - valid_pixels,
        "minimum": score_min,
        "maximum": score_max,
        "average": score_sum / valid_pixels,
        "category_counts": category_counts.tolist(),
        "positive_count": positive_count,
        "threshold": model.threshold,
        "support_counts": support_counts.tolist(),
        "mean_distance": distance_sum / valid_pixels,
        "minimum_distance": distance_min,
        "maximum_distance": distance_max,
        "oor_counts": oor_counts.tolist(),
        "oor_features": [model.feature_names[i] for i in support.cont_idx],
        "shap_mean": np.mean(shap_values, axis=0).tolist(),
        "shap_mean_abs": np.mean(np.abs(shap_values), axis=0).tolist(),
        "shap_feature_means": np.mean(shap_X, axis=0).tolist(),
        "shap_base": float(base_value),
        "shap_sample_size": int(shap_limit),
        "shap_engine": explainer.engine,
        "feature_names": list(model.feature_names),
    }


def _render_shap_results(result):
    st.subheader("What Influences the Model Across This Area?")
    st.write(
        "SHAP explains how environmental features contribute to the model "
        "score relative to its expected score. These results summarize "
        "representative pixels from the selected area."
    )

    explanation = pd.DataFrame({
        "Feature": [label(name) for name in result["feature_names"]],
        "Mean feature value": result["shap_feature_means"],
        "Mean SHAP contribution": result["shap_mean"],
        "Mean absolute SHAP": result["shap_mean_abs"],
    }).sort_values("Mean absolute SHAP", ascending=False)

    positive = explanation[
        explanation["Mean SHAP contribution"] > 0
    ].head(5)

    negative = explanation[
        explanation["Mean SHAP contribution"] < 0
    ].sort_values("Mean absolute SHAP", ascending=False).head(5)

    positive_col, negative_col = st.columns(2)

    with positive_col:
        st.markdown("**Features pushing the score upward on average**")
        if positive.empty:
            st.write("No positive average contributions in this sample.")
        else:
            st.dataframe(positive.round(5), hide_index=True, use_container_width=True)

    with negative_col:
        st.markdown("**Features pushing the score downward on average**")
        if negative.empty:
            st.write("No negative average contributions in this sample.")
        else:
            st.dataframe(negative.round(5), hide_index=True, use_container_width=True)

    st.markdown("**Strongest influencing features**")
    st.dataframe(explanation.round(5), hide_index=True, use_container_width=True)

    st.caption(
        f"SHAP sample: {result['shap_sample_size']} pixels. "
        f"Expected model score: {result['shap_base']:.5f}. "
        f"Engine: {result['shap_engine']}. Positive contributions push "
        "the model score upward; negative contributions push it downward. "
        "These are model associations, not proof of physical causation."
    )


def _render_decision_support(result):
    st.subheader("Training Data Evidence")

    names = [
        "Well supported",
        "Partially supported",
        "Limited support / extrapolation",
    ]
    counts = result["support_counts"]
    total = result["valid_pixels"]

    table = pd.DataFrame({
        "Evidence coverage": names,
        "Pixels": counts,
        "Percentage": [f"{n / total * 100:.1f}%" for n in counts],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)

    columns = st.columns(3)
    columns[0].metric("Average distance to training landslide", f"{result['mean_distance']:.2f} km")
    columns[1].metric("Nearest training landslide", f"{result['minimum_distance']:.2f} km")
    columns[2].metric("Farthest training distance", f"{result['maximum_distance']:.2f} km")

    oor_rows = [
        {
            "Feature": label(name),
            "Pixels outside training range": count,
            "Percentage": f"{count / total * 100:.1f}%",
        }
        for name, count in zip(result["oor_features"], result["oor_counts"])
        if count
    ]

    if oor_rows:
        st.markdown("**Continuous features outside their training ranges**")
        st.dataframe(pd.DataFrame(oor_rows), hide_index=True, use_container_width=True)
    else:
        st.write("All assessed continuous feature values fall within their training ranges.")

    low, medium, high = result["category_counts"]
    high_pct = high / total * 100
    medium_high_pct = (medium + high) / total * 100
    limited_pct = counts[2] / total * 100

    st.subheader("Decision Guidance")

    if high_pct >= 50:
        guidance = (
            "A substantial share of this area is in the High screening category. "
            "Prioritize its higher-scoring locations for closer terrain and drainage review."
        )
    elif medium_high_pct >= 50:
        guidance = (
            "At least half of this area is in the Medium or High screening categories. "
            "Review the higher-scoring portions first."
        )
    else:
        guidance = (
            "Most assessed pixels are in the Low screening category. "
            "Pay attention to local Medium and High pockets rather than relying only "
            "on the area average."
        )

    if limited_pct >= 25:
        guidance += (
            " A considerable portion has limited training-data support, "
            "so interpret its model scores with additional caution."
        )

    st.info(guidance)
    st.caption(
        "Training-data support is a heuristic based on distance to training "
        "landslide pixels and continuous feature ranges. It is not a confidence "
        "interval. This guidance is decision support, not an official hazard designation."
    )


def render_area_assessment():
    st.title("Area Assessment")
    st.caption(
        "Assess landslide susceptibility across a selected area in Kodagu "
        "using pixel-level predictions from the deployed Random Forest model."
    )
    st.markdown(
        "Draw one or more polygons or rectangles on the map. "
        "Only pixels inside the Kodagu district boundary will be assessed."
    )

    st.session_state.setdefault("selected_areas", [])
    map_view = add_polygon_drawing(create_base_map())
    map_data = display_map(map_view, height=600)

    if map_data and "all_drawings" in map_data:
        drawings = map_data.get("all_drawings") or []
        if drawings != st.session_state["selected_areas"]:
            st.session_state["selected_areas"] = drawings
            st.session_state.pop("area_assessment_result", None)

    drawings = st.session_state["selected_areas"]

    st.divider()
    st.subheader("Selected Area")

    if drawings:
        st.success(f"{len(drawings)} drawn shape(s) currently selected.")
        st.caption("Overlapping shapes are combined; each raster pixel is assessed only once.")
    else:
        st.info("No area selected yet. Use the polygon or rectangle tool on the map to draw an area.")

    if st.button(
        "Run Area Assessment",
        type="primary",
        disabled=not drawings,
        use_container_width=True,
    ):
        with st.spinner(
            "Predicting pixels, measuring training support and calculating SHAP explanations..."
        ):
            try:
                store, model, support, explainer = load_services()
                result = _assess_area(drawings, store, model, support, explainer)
                st.session_state["area_assessment_result"] = result
            except Exception as error:
                st.session_state.pop("area_assessment_result", None)
                st.error(f"Area assessment failed: {error}")

    result = st.session_state.get("area_assessment_result")

    st.divider()
    st.subheader("Area Susceptibility Results")

    if not result:
        st.info("Draw an area and select **Run Area Assessment** to calculate its susceptibility statistics.")
        return

    st.success("Area assessment completed.")

    columns = st.columns(3)
    columns[0].metric("Minimum model score", f"{result['minimum']:.4f}")
    columns[1].metric("Average model score", f"{result['average']:.4f}")
    columns[2].metric("Maximum model score", f"{result['maximum']:.4f}")

    columns = st.columns(3)
    columns[0].metric("Valid pixels assessed", f"{result['valid_pixels']:,}")
    columns[1].metric("Assessed area", f"{result['valid_pixels'] * PIXEL_AREA_M2 / 1_000_000:.2f} km²")
    columns[2].metric("Deployed positive classifications", f"{result['positive_count']:,}")

    if result["missing_pixels"]:
        st.warning(
            f"{result['missing_pixels']:,} selected pixels inside Kodagu were excluded "
            "because one or more required environmental features were unavailable."
        )

    st.caption(
        f"Each assessed pixel represents approximately 900 m² (30 m × 30 m). "
        f"The assessment used {result['polygon_count']} valid polygon component(s)."
    )

    st.divider()
    st.subheader("Susceptibility Distribution")

    counts = result["category_counts"]
    total = result["valid_pixels"]

    distribution = pd.DataFrame({
        "Category": ["Low", "Medium", "High"],
        "Model score range": [
            f"Below {LOW_LIMIT:.2f}",
            f"{LOW_LIMIT:.2f} to below {result['threshold']:.2f}",
            f"{result['threshold']:.2f} or above",
        ],
        "Pixels": counts,
        "Percentage": [f"{n / total * 100:.1f}%" for n in counts],
    })

    st.dataframe(distribution, hide_index=True, use_container_width=True)
    st.bar_chart(pd.DataFrame({"Pixels": counts}, index=["Low", "Medium", "High"]))

    st.metric(
        "Pixels classified positive at the deployed threshold",
        f"{result['positive_count']:,} ({result['positive_count'] / total * 100:.1f}%)",
    )

    st.caption(
        "The deployed binary classification threshold remains "
        f"{result['threshold']:.2f}. Low, Medium and High are screening "
        "categories, not official hazard classes."
    )

    st.divider()
    _render_shap_results(result)

    st.divider()
    _render_decision_support(result)

    st.divider()
    st.subheader("Overall Interpretation")
    st.write(
        f"The average model score is **{result['average']:.4f}**, with scores "
        f"ranging from **{result['minimum']:.4f}** to **{result['maximum']:.4f}**. "
        f"{result['positive_count'] / total * 100:.1f}% of valid pixels meet "
        "the deployed positive-classification threshold."
    )

    st.info(
        "Scores are the Random Forest's mean tree probabilities, not calibrated "
        "real-world probabilities of landslides. Area statistics summarize model "
        "outputs across raster pixels; they do not establish that a landslide will occur."
    )
