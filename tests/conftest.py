import pandas as pd
import pytest

from src.data import generate_demo_data, preprocess_stock_data
from src.context import build_context_frame
from src.features import add_features


@pytest.fixture
def cleaned_data() -> pd.DataFrame:
    cleaned, _ = preprocess_stock_data(generate_demo_data(rows=320))
    return cleaned


@pytest.fixture
def engineered_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    context = build_context_frame(cleaned_data["Date"]).frame
    return add_features(cleaned_data, context_data=context)
