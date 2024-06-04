"""
Tests for the data pipeline module.
"""

import xarray as xr
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


def test_load_data(temp_data):
    """
    Test the load_data function.
    """
    ds = load_data("temp_" + file_name)
    assert isinstance(ds, xr.Dataset)


def test_dataset_to_array():
    pass


def test_array_to_dataarray():
    pass
