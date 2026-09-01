import pandas as pd
import pytest


# fixtures used by io and validation
@pytest.fixture
def csv_data():
    return pd.DataFrame.from_dict(
        {
            "Species": ["SF", "SF", "SFP", "SFP"],
            "HOUR": [1.99, 2.01, 1.99, 2.01],
            "XMic": [128, 128, 128, 128],
            "RaW_Od": [0.2, 0.18, 0.25, 0.17],
        }
    )


@pytest.fixture
def required_columns():
    return ["species", "hour", "xmic", "raw_od"]
