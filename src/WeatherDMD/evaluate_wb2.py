import os
import numpy as np
from pyprojroot import here
from weatherbench2 import config
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
from weatherbench2.metrics import RMSESqrtBeforeTimeAvg, SpatialMSE
from weatherbench2.regions import SliceRegion
from WeatherDMD.data_pipeline import load_data


def _set_up_data_config(
    obs_path: str,
    forecast_path: str,
    output_dir: str,
    variables: list,
    levels: list,
    start_date: str,
    end_date: str,
) -> config.Data:
    """
    Set up the configuration for the data to be evaluated by WeatherBench2.
    """

    # if obs_path is a path and not a file name, check if its a relative path or an absolute path
    # if it's a file name, assume it's in the data/input directory
    if os.path.sep in obs_path:
        abs_path = os.path.join(here(), obs_path)
        obs_path = abs_path if os.path.exists(abs_path) else obs_path
    else:
        obs_path = os.path.join(here(), "data/input", obs_path)

    # if forecast_path is a path and not a file name, check if its a relative path or an absolute path
    # if it's a file name, assume it's in the data/output directory
    if os.path.sep in forecast_path:
        abs_path = os.path.join(here(), forecast_path)
        forecast_path = abs_path if os.path.exists(abs_path) else forecast_path
    else:
        forecast_path = os.path.join(here(), "data/output", forecast_path)

    # Get the forecast file name without the path and extension, so
    # that it can be used as the prefix for the output file.
    output_file_prefix = os.path.basename(forecast_path)
    output_file_prefix, _ = os.path.splitext(output_file_prefix)
    output_file_prefix = f"{output_file_prefix}_"

    if output_dir is None:
        output_dir = os.path.join(here(), "data/weatherbench2")
    else:
        # check if output_dir is a relative path or an absolute path
        if os.path.exists(os.path.join(here(), output_dir)):
            output_dir = os.path.join(here(), output_dir)
        elif os.path.exists(output_dir):
            output_dir = output_dir
        else:
            raise FileNotFoundError(f"Directory {output_dir} does not exist.")

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
        by_init=True,  # we are following init-time convention (see https://weatherbench2.readthedocs.io/en/latest/init-vs-valid-time.html)
    )


def _set_up_eval_config(regions: dict = None) -> dict:
    """
    Set up the configuration for the evaluation of the data by WeatherBench2.
    """

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
                "rmse": RMSESqrtBeforeTimeAvg(),  # TODO: support other metrics?
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
    """
    Compute the evaluation metrics for the forecast using WeatherBench2.
    The results are saved in the data/weatherbench2 directory, prefixed with the forecast file name.

    Parameters
    ----------
    obs_path : str
        Path to the observation data. Can be a relative path, an absolute path, or a file name.
        If it's a file name, it's assumed to be in the data/input directory.
    forecast_path : str
        Path to the forecast data. Can be a relative path, an absolute path, or a file name.
        If it's a file name, it's assumed to be in the data/output directory.
    output_dir : str, optional
        Directory to save the evaluation results. Can be a relative path or an absolute path.
        Default is the data/weatherbench2 directory.
    variables : list, optional
        List of variables to evaluate. If None, all variables in the forecast data are evaluated.
    levels : list, optional
        List of levels to evaluate. If None, all levels in the forecast data are evaluated.
    start_date : str, optional
        Start date for the evaluation (in the format "YYYY-MM-DD"). If None, the first date in the forecast data is used.
    end_date : str, optional
        End date for the evaluation (in the format "YYYY-MM-DD"). If None, the last date in the forecast data is used.
    regions : dict, optional
        Dictionary of regions to evaluate the data in. The keys are the names of the regions and the values are tuples
        of the latitudinal and longitudinal slices for the region. If None, the global region is evaluated.
    use_beam : bool, optional
        Whether to use Apache Beam for the evaluation. If False, the evaluation is done in memory. Default is False.
    """

    forecast = load_data(forecast_path)

    if variables is None:
        variables = [i for i in forecast.data_vars]

    if levels is None:
        levels = list(forecast.level.values)

    if start_date is None:
        start_date = forecast.time.values[0]
        start_date = np.datetime_as_string(start_date, unit="D")

    if end_date is None:
        end_date = forecast.time.values[-1]
        end_date = np.datetime_as_string(end_date, unit="D")

    try:
        data_config = _set_up_data_config(
            obs_path=obs_path,
            forecast_path=forecast_path,
            output_dir=output_dir,
            variables=variables,
            levels=levels,
            start_date=start_date,
            end_date=end_date,
        )
        eval_config = _set_up_eval_config(regions=regions)
    except Exception as e:
        print(f"Error setting up configuration for WeatherBench2: {e}")
        raise

    if not use_beam:
        print("Evaluating WB2 in memory...")
        try:
            evaluate_in_memory(data_config, eval_config)
        except Exception as e:
            print(f"Error evaluating WB2 in memory: {e}")
            raise
    else:
        try:
            print("Evaluating WB2 with Beam...")
            evaluate_with_beam(
                data_config,
                eval_config,
                runner="DirectRunner",
                input_chunks={"time": 1},
                argv=[
                    "--direct_num_workers",
                    "0",
                    "--direct_running_mode",
                    "multi_threading",
                ],
            )
        except Exception as e:
            print(f"Error evaluating WB2 with Beam: {e}")
            print("Falling back to evaluating WB2 in memory...")
            try:
                evaluate_in_memory(data_config, eval_config)
            except Exception as e:
                print(f"Error evaluating WB2 in memory: {e}")
                raise
