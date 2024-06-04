import xarray as xr
import os
from pyprojroot import here


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
    downsample: int = 1,
    subregion: tuple = None,
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
    downsample : int, optional
        Factor to downsample the dataset in the lat and lon directions. Default is 1. Must be an integer greater than 0.
    subregion : tuple, optional
        Tuple with the coordinates of the subregion to extract. Must be in the format (lat_min, lat_max, lon_min, lon_max).

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the variable extracted from the dataset, with dimensions (time, lat, lon).
    attrs : dict
        Dictionary with the attributes of the variable.
    coords : DataArray Coordinates
        Coordinates of the variable.
    dims : tuple
        Tuple with the dimensions of the variable.
    """

    try:
        if level is None:
            data = ds[variable].isel(level=0)
        else:
            data = ds[variable].sel(level=level)
        if subregion:
            data = data.sel(
                latitude=slice(subregion[0], subregion[1]),
                longitude=slice(subregion[2], subregion[3]),
            )
        data = data.coarsen(
            latitude=downsample, longitude=downsample, boundary="trim"
        ).mean()
        attrs = data.attrs
        coords = data.coords
        dims = data.dims
        data = data.values
    except Exception as e:
        print(f"Error converting dataset to numpy array: {e}")
        return None, None, None, None

    return data, attrs, coords, dims


def array_to_dataarray(
    data: xr.DataArray, attrs: dict, coords: xr.DataArray, dims: tuple
) -> xr.DataArray:
    """
    Convert numpy array to xarray DataArray.

    Parameters
    ----------
    data : numpy.ndarray
        Numpy array to convert.
    attrs : dict
        Dictionary with the attributes of the variable.
    coords : DataArray Coordinates
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
