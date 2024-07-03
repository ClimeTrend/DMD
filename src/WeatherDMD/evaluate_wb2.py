import xarray as xr
import os
import numpy as np
from pyprojroot import here
from weatherbench2 import config
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
from weatherbench2.metrics import RMSESqrtBeforeTimeAvg, SpatialMSE
from weatherbench2.regions import SliceRegion
from WeatherDMD.data_pipeline import load_data


def set_up_data_config(
    obs_path: str,
    forecast_path: str,
    output_dir: str,
    variables: list,
    levels: list,
    start_date: str,
    end_date: str,
) -> config.Data:

    output_file_prefix = os.path.basename(forecast_path)
    output_file_prefix, _ = os.path.splitext(output_file_prefix)

    paths_config = config.Paths(
        forecast=forecast_path,
        obs=obs_path,
        output_dir=output_dir,
        output_file_prefix=output_file_prefix,
    )

    selection_config = config.Selection(
        variables=variables,
        levels=levels,
        time_slice=slice(start_date, end_date),
    )

    return config.Data(
        selection=selection_config,
        paths=paths_config,
        by_init=False,  # we are following by-valid convention (see https://weatherbench2.readthedocs.io/en/latest/init-vs-valid-time.html)
    )


def set_up_eval_config(regions: dict = None) -> dict:

    if regions is None:
        regions = {
            "global": SliceRegion(),
        }
    else:
        regions = {
            name: SliceRegion(lat_slice=region[0], lon_slice=region[1])
            for name, region in regions.items()
        }
    return {
        "spatial": config.Eval(
            metrics={
                "spatial_mse": SpatialMSE(),
            },
            regions=regions,
        ),
        "non_spatial": config.Eval(
            metrics={
                "rmse": RMSESqrtBeforeTimeAvg(),
            },
            regions=regions,
        ),
    }


def evaluate_wb2(
    obs_path: str,
    forecast_path: str,
    output_dir: str = None,
    variables: list = None,
    levels: list = None,
    start_date: str = None,
    end_date: str = None,
    regions: dict = None,
    use_beam: bool = False,
) -> None:

    forecast = load_data(forecast_path)

    if output_dir is None:
        output_dir = os.path.join(here(), "data/weatherbench2")

    if variables is None:
        variables = [i for i in forecast.data_vars]
        variables = variables[0]

    if levels is None:
        levels = [i for i in forecast[variables].levels]
        levels = levels[0]

    if start_date is None:
        start_date = forecast.time.values[0]
        start_date = np.datetime_as_string(start_date, unit="D")

    if end_date is None:
        end_date = forecast.time.values[-1]
        end_date = np.datetime_as_string(end_date, unit="D")
