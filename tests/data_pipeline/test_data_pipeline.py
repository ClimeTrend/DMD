"""
Tests for the data pipeline module.
"""

import xarray as xr
import numpy as np
import os
import pytest
from WeatherDMD.data_pipeline import (
    load_data,
    dataset_to_array,
    array_to_dataarray,
    datarray_to_zarr,
)
from pyprojroot import here


file_name = "era5_slice_test"


@pytest.fixture(scope="module")
def temp_data():
    """
    Copy the data file to a temporary location for testing.
    """
    path = os.path.join(here(), "tests/data_pipeline/data/input", file_name + ".nc")
    temp_input_path = os.path.join(here(), "data/input/temp_" + file_name + ".nc")
    temp_output_path = os.path.join(here(), "data/output/temp_" + file_name + ".zarr")
    os.system(f"cp {path} {temp_input_path}")
    yield
    os.system(f"rm {temp_input_path}")
    os.system(f"rm -r {temp_output_path}")


@pytest.mark.dependency()
def test_load_data(temp_data):
    """
    Test the load_data function.
    """
    ds = load_data("temp_" + file_name + ".nc")
    assert isinstance(ds, xr.Dataset)


@pytest.mark.dependency(depends=["test_load_data"])
def test_dataset_to_array():
    """
    Test the dataset_to_array function.
    """
    ds = load_data("temp_" + file_name + ".nc")
    lat_slice = slice(20, -20)
    lon_slice = slice(100, 140)
    data, _, coords, _ = dataset_to_array(
        ds, "temperature", downsample=2, lat_slice=lat_slice, lon_slice=lon_slice
    )
    assert isinstance(data, np.ndarray)
    assert data.ndim == 3
    assert not any([dim == 0 for dim in data.shape])
    assert np.all(coords["latitude"] >= -20) and np.all(coords["latitude"] <= 20)
    assert np.all(coords["longitude"] >= 100) and np.all(coords["longitude"] <= 140)


@pytest.mark.dependency(depends=["test_load_data", "test_dataset_to_array"])
def test_array_to_dataarray():
    """
    Test the array_to_dataarray function.
    """
    ds = load_data("temp_" + file_name + ".nc")
    da1 = ds["temperature"].isel(level=0)
    data, attrs, coords, dims = dataset_to_array(ds, "temperature")
    da2 = array_to_dataarray(data, attrs, coords, dims)
    assert isinstance(da2, xr.DataArray)
    xr.testing.assert_equal(da1, da2)


@pytest.mark.dependency(
    depends=["test_load_data", "test_dataset_to_array", "test_array_to_dataarray"]
)
def test_datarray_to_zarr():
    """
    Test the datarray_to_zarr function.
    """
    ds = load_data("temp_" + file_name + ".nc")
    data, attrs, coords, dims = dataset_to_array(ds, "temperature")
    da = array_to_dataarray(data, attrs, coords, dims)
    datarray_to_zarr(da, "temp_" + file_name, prepend_time=False)
    assert os.path.exists(
        os.path.join(here(), "data/output/temp_" + file_name + ".zarr")
    )
    ds = xr.open_zarr(os.path.join(here(), "data/output/temp_" + file_name + ".zarr"))
    assert isinstance(ds, xr.Dataset)
    assert "temperature" in ds.variables
