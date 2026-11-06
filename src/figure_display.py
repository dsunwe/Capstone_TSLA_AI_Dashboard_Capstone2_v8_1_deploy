"""Render Matplotlib figures without a separate Streamlit media request."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st


def show_figure(figure: Figure | None) -> None:
    """Send a PNG data URI so charts do not depend on the /media endpoint."""
    if figure is None:
        st.info("No chart is available for the selected data.")
        return

    try:
        with BytesIO() as image:
            figure.savefig(image, format="png", bbox_inches="tight", dpi=200)
            encoded = base64.b64encode(image.getvalue()).decode("ascii")
        st.image(f"data:image/png;base64,{encoded}", width="stretch")
    finally:
        plt.close(figure)

