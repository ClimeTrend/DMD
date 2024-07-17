from WeatherDMD.evaluate_wb2 import evaluate_wb2
from pyprojroot import here
import os
import pytest
import xarray as xr


input_data_path = os.path.join(here(), "tests/evaluate_wb2/data/input")
output_data_path = os.path.join(here(), "tests/evaluate_wb2/data/output")


@pytest.fixture(scope="module")
def temp_data():
    """
    Temporarily save NetCDF files as Zarr files for testing.
    """
    obs_path = os.path.join(input_data_path, "era5_slice_test.nc")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.nc")

    obs = xr.open_dataset(obs_path)
    forecast = xr.open_dataset(forecast_path)

    obs_path = os.path.join(input_data_path, "era5_slice_test.zarr")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.zarr")

    obs.to_zarr(obs_path, mode="w")
    forecast.to_zarr(forecast_path, mode="w")

    yield

    os.system(f"rm -r {obs_path}")
    os.system(f"rm -r {forecast_path}")


def test_evaluate_wb2(temp_data):
    """
    Test the evaluate_wb2 function.
    """
    obs_path = os.path.join(input_data_path, "era5_slice_test.zarr")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.zarr")

    evaluate_wb2(obs_path, forecast_path, output_dir=output_data_path)
