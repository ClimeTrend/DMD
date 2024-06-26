"""
Tests for the data pipeline module.
"""

import xarray as xr
import numpy as np
import os
import pytest
from WeatherDMD.data_pipeline import load_data, dataset_to_array, array_to_dataarray
from pyprojroot import here


file_name = "era5_slice.nc"


@pytest.fixture(scope="module")
def temp_data():
    """
    Copy the data file to a temporary location for testing.
    """
    path = os.path.join(here(), "tests/data_pipeline/data/input", file_name)
    temp_path = os.path.join(here(), "data/input/temp_" + file_name)
    os.system(f"cp {path} {temp_path}")
    yield
    os.system(f"rm {temp_path}")


@pytest.mark.dependency()
def test_load_data(temp_data):
    """
    Test the load_data function.
    """
    ds = load_data("temp_" + file_name)
    assert isinstance(ds, xr.Dataset)


@pytest.mark.dependency(depends=["test_load_data"])
def test_dataset_to_array():
    """
    Test the dataset_to_array function.
    """
    ds = load_data("temp_" + file_name)
    lat_slice = slice(-20, 20)
    lon_slice = slice(100, 140)
    data, _, coords, _ = dataset_to_array(
        ds, "temperature", downsample=2, lat_slice=lat_slice, lon_slice=lon_slice
    )
    assert isinstance(data, np.ndarray)
    assert np.all(coords["latitude"] >= -20) and np.all(coords["latitude"] <= 20)
    assert np.all(coords["longitude"] >= 100) and np.all(coords["longitude"] <= 140)


@pytest.mark.dependency(depends=["test_load_data", "test_dataset_to_array"])
def test_array_to_dataarray():
    """
    Test the array_to_dataarray function.
    """
    ds = load_data("temp_" + file_name)
    da1 = ds["temperature"].isel(level=0)
    data, attrs, coords, dims = dataset_to_array(ds, "temperature")
    da2 = array_to_dataarray(data, attrs, coords, dims)
    assert isinstance(da2, xr.DataArray)
    xr.testing.assert_equal(da1, da2)
