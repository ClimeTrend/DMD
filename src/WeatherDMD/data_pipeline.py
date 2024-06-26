import xarray as xr
import os
from pyprojroot import here
import numpy as np


def load_data(file_name: str) -> xr.Dataset:
    """
    Load dataset from netCDF file or Zarr store.

    Parameters
    ----------
    file_name : str
        Name of the file to load. The file must be located in the data/input directory.

    Returns
    -------
    ds : xarray.Dataset
        Dataset loaded from the file.
    """

    path = os.path.join(here(), "data/input", file_name)

    try:
        if path.endswith(".nc"):
            ds = xr.open_dataset(path)
        elif path.endswith(".zarr"):
            ds = xr.open_zarr(path)
        else:
            raise ValueError("File format not supported")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    return ds


def dataset_to_array(
    ds: xr.Dataset,
    variable: str,
    level: int = None,
    lat_slice: slice = None,
    lon_slice: slice = None,
    downsample: int = 1,
) -> tuple:
    """
    Extract a variable from xarray dataset and convert it to numpy array.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert.
    variable : str
        Variable name to extract from the dataset.
    level : int, optional
        Level to extract from the dataset. If not specified, the first level is extracted.
    lat_slice : slice, optional
        Slice to extract from the latitude dimension of the dataset.
    lon_slice : slice, optional
        Slice to extract from the longitude dimension of the dataset.
    downsample : int, optional
        Factor to downsample the dataset in the lat and lon directions. Default is 1. Must be an integer greater than 0.

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the variable extracted from the dataset, with dimensions (time, lat, lon).
    attrs : dict
        Dictionary with the attributes of the variable.
    coords : dict
        Coordinates of the variable.
    dims : tuple
        Tuple with the dimensions of the variable.
    """

    try:
        if level is None:
            data = ds[variable].isel(level=0)
        else:
            data = ds[variable].sel(level=level)
        if lat_slice:
            data = data.sel(latitude=lat_slice)
        if lon_slice:
            data = data.sel(longitude=lon_slice)
        data = data.coarsen(
            latitude=downsample, longitude=downsample, boundary="trim"
        ).mean()
        attrs = data.attrs
        coords = dict(data.coords)
        dims = data.dims
        data = data.values
    except Exception as e:
        print(f"Error converting dataset to numpy array: {e}")
        return None, None, None, None

    return data, attrs, coords, dims


def array_to_dataarray(
    data: xr.DataArray, attrs: dict, coords: dict, dims: tuple
) -> xr.DataArray:
    """
    Convert numpy array to xarray DataArray.

    Parameters
    ----------
    data : numpy.ndarray
        Numpy array to convert.
    attrs : dict
        Dictionary with the attributes of the variable.
    coords : dict
        Coordinates of the variable.
    dims : tuple
        Tuple with the dimensions of the variable.

    Returns
    -------
    da : xarray.DataArray
        DataArray created from the numpy array.
    """

    try:
        da = xr.DataArray(data, coords=coords, dims=dims)
        da.attrs = attrs
    except Exception as e:
        print(f"Error converting numpy array to DataArray: {e}")
        return None

    return da


def datarray_to_zarr(da: xr.DataArray, file_name: str = "era5_dmd_forecast"):
    """
    Save DataArray to Zarr store.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to save.
    file_name : str
        Name of the file to save. Will be preceded by the time range of the data.
        Will be saved in the data/output directory.
    """

    time = da.time.values
    time_start = np.datetime_as_string(time[0], unit="D")
    time_end = np.datetime_as_string(time[-1], unit="D")

    path = os.path.join(
        here(), "data/output", f"{time_start}_{time_end}_{file_name}.zarr"
    )
    da.to_zarr(path, mode="w")
    print(f"DataArray saved to {path}")
