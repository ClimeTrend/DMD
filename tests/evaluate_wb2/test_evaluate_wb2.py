from WeatherDMD.evaluate_wb2 import evaluate_wb2
from pyprojroot import here
import os


def test_evaluate_wb2():
    """
    Test the evaluate_wb2 function.
    """
    obs_path = os.path.join(
        here(), "tests/evaluate_wb2/data/input/era5_slice_test.zarr"
    )
    forecast_path = os.path.join(
        here(), "tests/evaluate_wb2/data/input/era5_dmd_forecast_test.zarr"
    )

    evaluate_wb2(obs_path, forecast_path)
