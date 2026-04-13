import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


def estimate_deer_age(image_file) -> dict | None:
    try:
        response = requests.post(
            f"{API_BASE_URL}/estimate",
            files={"image": (image_file.name, image_file, image_file.type)},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        import streamlit as st
        st.error(f"API error: {e}")
        return None
