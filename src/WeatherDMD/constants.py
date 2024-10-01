# A file to capture and define constants used in the project

# Meteorological data variables and dimensions in weatherbench2
wb2_variables = [
    "temperature_2m",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
]

wb2_obs_dimensions = ["time", "latitude", "longitude","level"]

wb2_forecast_dimensions = ["time", "latitude", "longitude", "level","prediction_timedelta"]