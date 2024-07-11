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
        Name of the file to load or relative/absolute path to the file.
        If only the name is provided, the file is assumed to be in the data/input directory.

    Returns
    -------
    ds : xarray.Dataset
        Dataset loaded from the file.
    """

    if os.path.sep in file_name:
        abs_path = os.path.join(here(), file_name)
        path = abs_path if os.path.exists(abs_path) else file_name
    else:
        path = os.path.join(here(), "data/input", file_name)

    try:
        if ".nc" in path:
            ds = xr.open_dataset(path)
        elif ".zarr" in path:
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


def array_to_datarray(
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
        da = da.expand_dims(
            level=[coords["level"].values]
        )  # for WB2, need to have the level coordinate as a dimension
        da.attrs = attrs
    except Exception as e:
        print(f"Error converting numpy array to DataArray: {e}")
        return None

    return da


def datarray_to_zarr(
    da: xr.DataArray,
    variable_name: str,
    file_name: str = "era5_dmd_forecast",
    prepend_time: bool = True,
):
    """
    Convert DataArray to Dataset and save it to a Zarr store.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to save.
    variable_name : str
        Name of the variable in the DataArray.
    file_name : str
        Name of the file to save. Will be saved in the data/output directory.
    prepend_time : bool
        If True, the start date of the DataArray will be prepended to the file name.
    """

    try:
        if prepend_time:
            time = da.time.values
            time_start = np.datetime_as_string(time[0], unit="D")
            path = os.path.join(here(), "data/output", f"{time_start}_{file_name}")
        else:
            path = os.path.join(here(), "data/output", f"{file_name}")

        # add ".zarr" extension if not present
        if ".zarr" not in path:
            path = f"{path}.zarr"

        ds = da.to_dataset(name=variable_name, promote_attrs=True)
        ds.to_zarr(path, mode="w-", consolidated=True)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving DataArray to Zarr: {e}")
        return None


def prepare_for_wb2(
    da: xr.DataArray,
    init_time: np.datetime64 = None,
) -> xr.DataArray:
    """
    Prepare DataArray for WeatherBench2 evaluation using the Init-time convention.
    See https://weatherbench2.readthedocs.io/en/latest/init-vs-valid-time.html#init-time-convention.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to prepare.
    init_time : np.datetime64, optional
        Initialization time of the forecast.
        In not provided, the first timestamp in the time coordinate minus the unique prediction time delta will be used.

    Returns
    -------
    da : xarray.DataArray
        DataArray prepared for WeatherBench2 evaluation using the Init-time convention.
        The time coordinate is renamed to "prediction_timedelta" and a new coordinate with name "time" is inserted.
        This new coordinate is the initialization time of the forecast.
    """

    try:
        time = da.time.values
        time_delta = np.diff(time)

        if init_time is None:
            if not np.all(time_delta == time_delta[0]):
                raise ValueError("Prediction time delta is not constant")
            time_delta = time_delta[0]
            init_time = time[0] - time_delta

        prediction_timedelta = time - init_time

        # update the time coordinate to be the prediction_timedelta
        da = da.assign_coords(time=("time", prediction_timedelta)).rename(
            time="prediction_timedelta"
        )

        # insert new coordinate with name "time" which is the init_time
        da = da.expand_dims(time=[init_time])
        return da
    except Exception as e:
        print(f"Error preparing data for WeatherBench2: {e}")
        return None
