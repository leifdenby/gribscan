from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import gribscan
from gribscan.magician import HarmonieMagician

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keep tests working without python-dotenv
    load_dotenv = None

HARMONIE_GRIB_LAMBERT_FP_ENV_VAR = "HARMONIE_LAMBERT_GRID_GRIB_FILE"
HARMONIE_GRIB_ROTOTATED_LATLON_FP_ENV_VAR = "HARMONIE_ROTATED_LATLON_GRID_GRIB_FILE"

HARMONIE_SF_LEVEL_TYPES = ['heightAboveGround', 'hybrid', 'heightAboveSea', 'surface', 'entireAtmosphere', 'isothermZero', 'isothermal', 'nominalTop', 'adiabaticCondensation', 'freeConvection', 'neutralBuoyancy', 'cloudTop']

if load_dotenv:
    load_dotenv()


@pytest.mark.slow
def test_harmonie_forecast_on_lambert_grid_can_be_read(tmp_path):
    """
    Ensure the HarmonieMagician builds usable references for the sample forecast.
    This means that:
    - the references are named by valid level types
    - the references can be opened as zarr datasets
    - at least one variable can be loaded from each dataset
    - lat/lon coordinates are present and vary along x/y dimensions
    """
    env_value = os.environ.get(HARMONIE_GRIB_LAMBERT_FP_ENV_VAR)
    if not env_value:
        warnings.warn(
            f"Set {HARMONIE_GRIB_LAMBERT_FP_ENV_VAR} in your environment or .env to run Harmonie tests.",
            UserWarning,
        )
        pytest.skip("Harmonie GRIB Lambert example path not provided")

    grib_path = Path(env_value).expanduser()
    if not grib_path.exists():
        pytest.skip("Harmonie sample GRIB is not available")

    index_path = tmp_path / "harmonie.index"
    gribscan.write_index(grib_path.as_posix(), index_path)

    magician = HarmonieMagician()
    refs = gribscan.grib_magic([index_path], magician=magician)
    
    # check that the keys of the references match valid level types
    invalid_level_types = set(refs.keys()) - set(HARMONIE_SF_LEVEL_TYPES)
    assert not invalid_level_types, f"Invalid level types found in references: {invalid_level_types}"
    
    for ref_name in refs.keys():
        ref_path = tmp_path / f"{ref_name}.json"

        with ref_path.open("w") as fh:
            json.dump(refs[ref_name], fh)

        ds = xr.open_zarr(f"reference::{ref_path}", consolidated=False)
        try:
            var_name = next(iter(ds.data_vars))
        except StopIteration:  # pragma: no cover - defensive guard
            pytest.fail("No variables found in Harmonie dataset")

        subset = ds[var_name]
        scalar_dims = {dim: 0 for dim in subset.dims if dim not in {"x", "y"}}
        subset = subset.isel(**scalar_dims)
        for dim in ("y", "x"):
            if dim in subset.dims:
                subset = subset.isel({dim: slice(0, 1)})

        values = np.asarray(subset.load().values)
        assert values.size > 0, "Expected to load at least one data point"
        assert np.isfinite(values).any(), "Loaded data should contain finite values"
        
        # check that lat/lon coordinates are present and that they span along x and y
        assert "lat" in ds.coords, "Latitude coordinate missing"
        assert "lon" in ds.coords, "Longitude coordinate missing"
        
        da_lat = ds["lat"]
        da_lon = ds["lon"]
        assert da_lat.shape == (ds.y.count(), ds.x.count()), "Latitude shape mismatch"
        assert da_lon.shape == (ds.y.count(), ds.x.count()), "Longitude shape mismatch"
        assert np.ptp(da_lat.values) > 0, "Latitude coordinate should vary along y dimension"
        assert np.ptp(da_lon.values) > 0, "Longitude coordinate should vary along x dimension"
        

    ds.close()


@pytest.mark.slow
def test_harmonie_analysis_on_rotated_latlon_grid_can_be_read(tmp_path):
    """
    Ensure the HarmonieMagician builds usable references for the sample analysis.
    This means that:
    - the references are named by valid level types
    - the references can be opened as zarr datasets
    - at least one variable can be loaded from each dataset
    - lat/lon coordinates are present and vary along x/y dimensions
    """
    env_value = os.environ.get(HARMONIE_GRIB_ROTOTATED_LATLON_FP_ENV_VAR)
    if not env_value:
        warnings.warn(
            f"Set {HARMONIE_GRIB_ROTOTATED_LATLON_FP_ENV_VAR} in your environment or .env to run Harmonie tests.",
            UserWarning,
        )
        pytest.skip("Harmonie GRIB rotated lat/lon example path not provided")

    grib_path = Path(env_value).expanduser()
    if not grib_path.exists():
        pytest.skip("Harmonie sample GRIB is not available")

    index_path = tmp_path / "harmonie_rotated.index"
    gribscan.write_index(grib_path.as_posix(), index_path)

    magician = HarmonieMagician()
    refs = gribscan.grib_magic([index_path], magician=magician)
    
    # check that the keys of the references match valid level types
    invalid_level_types = set(refs.keys()) - set(HARMONIE_SF_LEVEL_TYPES)
    assert not invalid_level_types, f"Invalid level types found in references: {invalid_level_types}"
    
    for ref_name in refs.keys():
        ref_path = tmp_path / f"{ref_name}.json"

        with ref_path.open("w") as fh:
            json.dump(refs[ref_name], fh)

        ds = xr.open_zarr(f"reference::{ref_path}", consolidated=False)
        try:
            var_name = next(iter(ds.data_vars))
        except StopIteration:  # pragma: no cover - defensive guard
            pytest.fail("No variables found in Harmonie dataset")

        subset = ds[var_name]
        scalar_dims = {dim: 0 for dim in subset.dims if dim not in {"x", "y"}}
        subset = subset.isel(**scalar_dims)
        for dim in ("y", "x"):
            if dim in subset.dims:
                subset = subset.isel({dim: slice(0, 1)})

        values = np.asarray(subset.load().values)
        
        assert values.size > 0, "Expected to load at least one data point"
        assert np.isfinite(values).any(), "Loaded data should contain finite values"
        # check that lat/lon coordinates are present and that they span along x and y
        assert "lat" in ds.coords, "Latitude coordinate missing"
        assert "lon" in ds.coords, "Longitude coordinate missing"
        da_lat = ds["lat"]
        da_lon = ds["lon"]
        assert da_lat.shape == (ds.y.count(), ds.x.count()), "Latitude shape mismatch"
        assert da_lon.shape == (ds.y.count(), ds.x.count()), "Longitude shape mismatch"
        assert np.ptp(da_lat.values) > 0, "Latitude coordinate should vary along y dimension"
        assert np.ptp(da_lon.values) > 0, "Longitude coordinate should vary along x dimension"
