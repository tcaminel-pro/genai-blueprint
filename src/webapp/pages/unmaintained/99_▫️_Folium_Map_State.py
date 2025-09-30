import base64
import tempfile

import folium
import streamlit as st
import streamlit.components.v1 as components


def create_toulouse_map() -> folium.Map:
    """Create a Folium map centered on Toulouse with some markers."""
    # Toulouse coordinates
    toulouse_coords = (43.6045, 1.4442)

    # Create map centered on Toulouse
    m = folium.Map(location=toulouse_coords, zoom_start=13)

    # Add some markers
    folium.Marker(location=(43.6045, 1.4442), popup="Capitole de Toulouse", icon=folium.Icon(color="red")).add_to(m)

    folium.Marker(location=(43.6083, 1.4437), popup="Place du Capitole", icon=folium.Icon(color="blue")).add_to(m)

    return m


def main():
    st.title("Toulouse Map1")

    # Create the map
    m = create_toulouse_map()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp_path = tmp.name
        m.save(tmp_path)

    HtmlFile = open(tmp_path, "r")
    raw_html = HtmlFile.read().encode("utf-8")
    raw_html = base64.b64encode(raw_html).decode()
    components.iframe(f"data:text/html;base64,{raw_html}", height=400)


main()
