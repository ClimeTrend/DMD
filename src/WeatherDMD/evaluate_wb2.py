import xarray as xr
from weatherbench2 import config
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
from weatherbench2.metrics import RMSESqrtBeforeTimeAvg, SpatialMSE
from weatherbench2.regions import SliceRegion


def set_up_data_config(
    obs_path: str,
    forecast_path: str,
    output_dir: str,
    variables: list,
    levels: list,
    start_date: str,
    end_date: str,
) -> config.Data:

    paths_config = config.Paths(
        forecast=forecast_path,
        obs=obs_path,
        output_dir=output_dir,
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
    output_dir: str,
    variables: list,
    levels: list,
    start_date: str,
    end_date: str,
    regions: dict = None,
    use_beam: bool = False,
) -> None:
    pass
