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

    if path.endswith(".nc"):
        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
    elif path.endswith(".zarr"):
        try:
            ds = xr.open_zarr(path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
    else:
        raise ValueError("File format not supported")
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
    """

    try:
        if level is None:
            data = ds[var_name].isel(level=0).values
        else:
            data = ds[var_name].sel(level=level).values
        return data
    except Exception as e:
        print(f"Error converting dataset to array: {e}")
