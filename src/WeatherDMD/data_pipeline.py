import xarray as xr
import os
from pyprojroot import here


def load_data(file_name: str):
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


def dataset_to_array(ds: xr.Dataset, var_name: str, level: int = None):
    """
    Convert xarray dataset to numpy array.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert.
    var_name : str
        Variable name to extract from the dataset.
    level : int, optional
        Level to extract from the dataset. If not specified, the first level is extracted.

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the data extracted from the dataset, with dimensions (time, lat, lon).
    attrs : dict
        Dictionary with the attributes of the variable.
    """

    try:
        if level is None:
            data = ds[var_name].isel(level=0)
        else:
            data = ds[var_name].sel(level=level)
        attrs = data.attrs
        data = data.values
    except Exception as e:
        print(f"Error converting dataset to array: {e}")
        return None, None

    return data, attrs
