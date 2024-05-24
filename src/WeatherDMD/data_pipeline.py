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
