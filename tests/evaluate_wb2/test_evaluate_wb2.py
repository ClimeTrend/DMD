from pyprojroot import here
import os
import pytest
import xarray as xr
from WeatherDMD.evaluate_wb2 import (
    evaluate_wb2,
    _set_up_data_config,
    _set_up_eval_config,
)
from weatherbench2 import config
from weatherbench2.metrics import RMSESqrtBeforeTimeAvg, SpatialMSE
from weatherbench2.regions import SliceRegion

from WeatherDMD.constants import (
    wb2_variables,
    wb2_forecast_dimensions,
    wb2_obs_dimensions,
)

input_data_path = os.path.join(here(), "tests/evaluate_wb2/data/input")
output_data_path = os.path.join(here(), "tests/evaluate_wb2/data/output")


@pytest.fixture(scope="module")
def temp_input_data():
    """
    Temporarily save NetCDF files as Zarr files for testing.
    """
    obs_path = os.path.join(input_data_path, "era5_slice_test.nc")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.nc")

    obs = xr.open_dataset(obs_path)
    forecast = xr.open_dataset(forecast_path)

    obs_path_zarr = os.path.join(input_data_path, "era5_slice_test.zarr")
    forecast_path_zarr = os.path.join(input_data_path, "era5_dmd_forecast_test.zarr")

    obs.to_zarr(obs_path_zarr, mode="w")
    forecast.to_zarr(forecast_path_zarr, mode="w")

    yield

    # Tear down
    os.system(f"rm -r {obs_path_zarr}")
    os.system(f"rm -r {forecast_path_zarr}")


@pytest.fixture(scope="module")
def temp_output_data(temp_input_data):
    """
    Temporarily generate the weatherbench2 evaluation of the provided input forecast.
    So it can be compared to golden output
    """

    obs_path = os.path.join(input_data_path, "era5_slice_test.zarr")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.zarr")

    evaluate_wb2(obs_path, forecast_path, output_dir=output_data_path)

    yield

    # Tear down
    os.system(
        f"rm -r {os.path.join(output_data_path, 'era5_dmd_forecast_test_spatial.nc')}"
    )
    os.system(
        f"rm -r {os.path.join(output_data_path, 'era5_dmd_forecast_test_non_spatial.nc')}"
    )


def test_format_of_input_files(temp_input_data):
    """
    Test the format of the incoming data: names of dimensions, structure, etc
    """

    obs = xr.open_dataset(
        os.path.join(input_data_path, "era5_slice_test.zarr"), engine="zarr"
    )
    forecast = xr.open_dataset(
        os.path.join(input_data_path, "era5_dmd_forecast_test.zarr"), engine="zarr"
    )

    # Check that obs and forecast variables are members of a list of accepted variables
    assert set(obs.data_vars).issubset(
        wb2_variables
    ), "Observed variables are not all accepted"
    assert set(forecast.data_vars).issubset(
        wb2_variables
    ), "Forecasted variables are not all accepted"

    # Check that the dimensions of the forecast and obs files are correct
    assert set(obs.dims).issubset(
        wb2_obs_dimensions
    ), "Observed dimensions are not all accepted"
    assert set(forecast.dims).issubset(
        wb2_forecast_dimensions
    ), "Forecasted dimensions are not all accepted"

    # Check that data type of the "time" dimension is datetime64[ns]
    assert (
        obs.time.dtype == "datetime64[ns]"
    ), "Time dimension in observations is not datetime64[ns]"
    assert (
        forecast.time.dtype == "datetime64[ns]"
    ), "Time dimension in forecast is not datetime64[ns]"

    assert (
        obs.latitude.dtype == "float32"
    ), "Latitude dimension in observations is not float32"
    assert (
        forecast.latitude.dtype == "float32"
    ), "Latitude dimension in forecast is not float32"

    assert (
        obs.longitude.dtype == "float32"
    ), "Latitude dimension in observations is not float32"
    assert (
        forecast.longitude.dtype == "float32"
    ), "Latitude dimension in forecast is not float32"

    assert obs.level.dtype == "int64", "level dimension in observations is not int64"
    assert forecast.level.dtype == "int64", "level dimension in forecast is not int64"

    # Test that each data variable in observations is of type float32
    for var in obs.data_vars:
        assert (
            obs[var].dtype == "float32"
        ), f"Variable {var} in observations is not float32"

    # Test that each data variable in forecast is of type float32
    for var in forecast.data_vars:
        assert (
            forecast[var].dtype == "float32"
        ), f"Variable {var} in forecast is not float32"


def test_set_up_data_config(temp_input_data):
    """
    Test the _set_up_data_config function.
    """
    obs_path = os.path.join(input_data_path, "era5_slice_test.zarr")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.zarr")
    output_dir = output_data_path
    variables = ["temperature", "pressure"]
    levels = [1000, 850]
    start_date = "2020-01-01"
    end_date = "2020-01-31"

    data_config = _set_up_data_config(
        obs_path=obs_path,
        forecast_path=forecast_path,
        output_dir=output_dir,
        variables=variables,
        levels=levels,
        start_date=start_date,
        end_date=end_date,
    )

    assert isinstance(
        data_config, config.Data
    ), "data_config is not an instance of config.Data"
    assert data_config.paths.obs == obs_path, "Observation path is not set correctly"
    assert (
        data_config.paths.forecast == forecast_path
    ), "Forecast path is not set correctly"
    assert (
        data_config.paths.output_dir == output_dir
    ), "Output directory is not set correctly"
    assert (
        data_config.selection.variables == variables
    ), "Variables are not set correctly"
    assert data_config.selection.levels == levels, "Levels are not set correctly"
    assert data_config.selection.time_slice == slice(
        start_date, end_date
    ), "Time slice is not set correctly"


def test_set_up_eval_config():
    """
    Test the _set_up_eval_config function.
    """
    regions = {
        "region1": (slice(10, 20), slice(30, 40)),
        "region2": (slice(50, 60), slice(70, 80)),
    }

    eval_config = _set_up_eval_config(regions=regions)

    assert isinstance(eval_config, dict), "eval_config is not a dictionary"
    assert "spatial" in eval_config, "spatial key is missing in eval_config"
    assert "non_spatial" in eval_config, "non_spatial key is missing in eval_config"
    assert isinstance(
        eval_config["spatial"], config.Eval
    ), "spatial config is not an instance of config.Eval"
    assert isinstance(
        eval_config["non_spatial"], config.Eval
    ), "non_spatial config is not an instance of config.Eval"
    assert (
        "spatial_mse" in eval_config["spatial"].metrics
    ), "spatial_mse metric is missing in spatial config"
    assert (
        "rmse" in eval_config["non_spatial"].metrics
    ), "rmse metric is missing in non_spatial config"
    assert isinstance(
        eval_config["spatial"].metrics["spatial_mse"], SpatialMSE
    ), "spatial_mse metric is not an instance of SpatialMSE"
    assert isinstance(
        eval_config["non_spatial"].metrics["rmse"], RMSESqrtBeforeTimeAvg
    ), "rmse metric is not an instance of RMSESqrtBeforeTimeAvg"
    assert (
        "region1" in eval_config["spatial"].regions
    ), "region definition is not set correctly, does not match inputs."
    assert (
        "region2" in eval_config["spatial"].regions
    ), "region definition is not set correctly, does not match inputs."
    assert isinstance(
        eval_config["spatial"].regions["region1"], SliceRegion
    ), "One of the input regions is not an instance of SliceRegion"
    assert isinstance(
        eval_config["spatial"].regions["region2"], SliceRegion
    ), "One of the input regions is not an instance of SliceRegion"


def test_output_against_golden_output(temp_output_data):
    """
    Test the output of the evaluate_wb2 function against the golden output.
    """

    # Load the golden output _non_spatial
    golden_output_path_non_spatial = os.path.join(
        input_data_path, "golden_output_era5_dmd_forecast_test_non_spatial.nc"
    )
    golden_output_non_spatial = xr.open_dataset(
        golden_output_path_non_spatial, engine="netcdf4"
    )

    # Load the current output _non_spatial
    current_output_path_non_spatial = os.path.join(
        output_data_path, "era5_dmd_forecast_test_non_spatial.nc"
    )
    current_output_non_spatial = xr.open_dataset(
        current_output_path_non_spatial, engine="netcdf4"
    )

    # Compare the golden output to current output _non_spatial
    assert golden_output_non_spatial.equals(
        current_output_non_spatial
    ), "The current non-spatial evaluation output does not match the golden output"

    # Load the golden output _non_spatial
    golden_output_path_spatial = os.path.join(
        input_data_path, "golden_output_era5_dmd_forecast_test_spatial.nc"
    )
    golden_output_spatial = xr.open_dataset(
        golden_output_path_spatial, engine="netcdf4"
    )

    # Load the current output _non_spatial
    current_output_path_spatial = os.path.join(
        output_data_path, "era5_dmd_forecast_test_spatial.nc"
    )
    current_output_spatial = xr.open_dataset(
        current_output_path_spatial, engine="netcdf4"
    )

    # Compare the golden output to current output _non_spatial
    assert golden_output_spatial.equals(
        current_output_spatial
    ), "The current spatial evaluation output does not match the golden output"


# Skip this test for now
# @pytest.mark.skip(reason="Not implemented yet")
def test_format_of_output_files(temp_output_data):
    """
    Test the format of the output data: names of dimensions, structure, etc
    """

    output = xr.open_dataset(
        os.path.join(output_data_path, "era5_dmd_forecast_test.nc"), engine="netcdf4"
    )

    # Check that the dimensions of the forecast and obs files are correct
    assert set(output.dims).issubset(
        wb2_forecast_dimensions
    ), "Forecasted dimensions are not all accepted"

    # Check that data type of the "time" dimension is datetime64[ns]
    assert (
        output.time.dtype == "datetime64[ns]"
    ), "Time dimension in forecast is not datetime64[ns]"

    assert (
        output.latitude.dtype == "float32"
    ), "Latitude dimension in forecast is not float32"

    assert (
        output.longitude.dtype == "float32"
    ), "Latitude dimension in forecast is not float32"

    assert output.level.dtype == "int64", "level dimension in forecast is not int64"

    # Test that each data variable in forecast is of type float32
    for var in output.data_vars:
        assert (
            output[var].dtype == "float32"
        ), f"Variable {var} in forecast is not float32"


# Skip this test for now
@pytest.mark.skip(reason="Not implemented yet")
def test_evaluate_wb2(temp_input_data):
    """
    Test the evaluate_wb2 function.
    """
    obs_path = os.path.join(input_data_path, "era5_slice_test.zarr")
    forecast_path = os.path.join(input_data_path, "era5_dmd_forecast_test.zarr")

    evaluate_wb2(obs_path, forecast_path, output_dir=output_data_path)
