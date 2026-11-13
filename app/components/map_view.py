"""Interactive Leaflet map creation for Kodagu assessment."""

import json
from pathlib import Path

import folium

KODAGU_CENTER = [12.4244, 75.7382]
DEFAULT_ZOOM = 10


def get_map_center():
    return KODAGU_CENTER.copy()


def get_default_zoom():
    return DEFAULT_ZOOM


def _ring_area(ring):
    area = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def add_kodagu_boundary(map_view):
    """Shade the area outside Kodagu red and outline the district."""

    boundary_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "kodagu_boundary.geojson"
    )

    with open(boundary_path, encoding="utf-8") as file:
        boundary_data = json.load(file)

    boundary_feature = boundary_data["features"][0]
    boundary_ring = boundary_feature["geometry"]["coordinates"][0]

    outer_ring = [
        [-180, -85],
        [180, -85],
        [180, 85],
        [-180, 85],
        [-180, -85],
    ]

    # GeoJSON exterior rings are counter-clockwise.
    # Interior rings must have the opposite orientation.
    if _ring_area(boundary_ring) > 0:
        boundary_ring = list(reversed(boundary_ring))

    outside_data = {
        "type": "Feature",
        "properties": {"name": "Outside Kodagu"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [outer_ring, boundary_ring],
        },
    }

    folium.GeoJson(
        outside_data,
        name="Outside Kodagu",
        style_function=lambda feature: {
            "fillColor": "#ff3333",
            "fillOpacity": 0.30,
            "color": "#ff3333",
            "opacity": 0,
            "weight": 0,
        },
        control=False,
    ).add_to(map_view)

    folium.GeoJson(
        boundary_data,
        name="Kodagu District Boundary",
        style_function=lambda feature: {
            "fillOpacity": 0,
            "color": "#263238",
            "weight": 3,
            "opacity": 1,
        },
        control=False,
    ).add_to(map_view)

    return map_view


def create_base_map():
    """Create the default interactive Kodagu map."""

    map_view = folium.Map(
        location=get_map_center(),
        zoom_start=get_default_zoom(),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_kodagu_boundary(map_view)

    return map_view


def add_location_marker(map_view, latitude, longitude):
    """Add a selected location marker to the map."""

    folium.Marker(
        location=[latitude, longitude],
        popup=(
            f"Latitude: {latitude:.6f}<br>"
            f"Longitude: {longitude:.6f}"
        ),
        tooltip="Selected Location",
        icon=folium.Icon(
            color="green",
            icon="map-marker",
        ),
    ).add_to(map_view)

    return map_view


def add_polygon_drawing(map_view):
    """Enable polygon drawing and editing on the map."""

    from folium.plugins import Draw

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(map_view)

    return map_view


def display_map(map_view, height=550):
    """Display a Folium map inside Streamlit."""

    from streamlit_folium import st_folium

    return st_folium(
        map_view,
        width=None,
        height=height,
    )
