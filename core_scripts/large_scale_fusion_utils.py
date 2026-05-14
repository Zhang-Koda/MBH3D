"""
Large-scale fusion module for MBH3D.

This script implements the large-scale component of the multi-scale Bayesian
hierarchical reconstruction framework used to reconstruct three-dimensional
ocean temperature anomalies from satellite-derived background fields and sparse
in situ observations.

Main workflow
-------------
For each target date and depth level, the script:
1. Estimates the background residual samples from the GTWR mapping residuals.
2. Estimates the coefficient-induced mapping uncertainty from GTWR-GSVC beta residuals.
3. Prepares in situ observations after removing the climatological seasonal cycle
   and the mesoscale first guess.
4. Builds the large-scale background ensemble and the coefficient-error ensemble.
5. Constructs observation and representativeness error variances.
6. Solves a localized Bayesian hierarchical EnOI analysis using sparse covariance
   matrices and a bilinear observation operator.

Notes
-----
- File names and variable names follow the conventions used in the MBH3D project.
- Paths are configurable through command-line arguments or by editing the
  FusionConfig dataclass.
- The algorithmic steps are kept consistent with the research version; only
  formatting, documentation, and public-release structure have been cleaned.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FusionConfig:
    """Configuration for large-scale HB-EnOI fusion.

    Parameters are intentionally explicit so that users can adapt this script to
    their own file organization without changing the core algorithm.
    """

    # Root directories. Replace these with local project paths before running.
    root_data_dir: str = "./data"
    output_dir: str = "./outputs/large_scale_fusion"
    satellite_input_dir: str = "./data/satellite_inputs"

    # Target region: [min_lon, max_lon, min_lat, max_lat]
    region: tuple[float, float, float, float] = (65.0, 120.0, -70.0, -40.0)

    # Spatial resolution settings. The input grid is assumed to be 0.25 deg and
    # is coarsened to the target large-scale grid resolution.
    input_resolution: float = 0.25
    target_resolution: float = 1.0

    # Depth levels used in the released reconstruction product.
    depth_levels: tuple[int, ...] = (5, 25, 50, 100, 150, 200, 300, 500, 700, 1000, 1500)

    # Temporal settings.
    start_date: str = "20201220"
    end_date: str = "20211231"
    year_range: tuple[int, int] = (1993, 2020)
    date_bandwidth_days: int = 60

    # Observation window relative to the analysis date:
    # use observations from [date - 30 days, date + 6 days].
    obs_bandwidth_days: tuple[int, int] = (60, 6)

    # Localization parameters for the two uncertainty components.
    length_scale_yy: tuple[float, float] = (5.0, 2.5)
    length_scale_bb: tuple[float, float] = (7.0, 3.5)
    localization_cutoff: float = 2.0
    search_pad: float = 0.15

    # Parallelization and chunking.
    n_workers: int = 24
    dates_per_job: int = 2

    # Model/data file-name stems.
    model_name: str = "xxx"
    gsvc_beta_stem: str = "gsvc_thetao_SO_50_140_m70_m30_4deg_largescale_bw30_4_3_2truncation_nobeta0_1993_2016"
    gtwr_beta_stem: str = "gtwr_thetao_SO_50_140_m70_m30_Bilateral_4deg_largescale_bw30_4_3_2truncation_nobeta0"

    # Internal path templates derived from root_data_dir.
    grid_ref_file: str = field(init=False)
    gtwr_glorys_input_stem: str = field(init=False)
    glorys_large_stem: str = field(init=False)
    gsvc_beta_file_stem: str = field(init=False)
    gtwr_beta_file_stem: str = field(init=False)
    depth_ref_file: str = field(init=False)
    climatology_stem: str = field(init=False)
    obs_stem: str = field(init=False)
    meso_first_guess_stem: str = field(init=False)
    large_first_guess_stem: str = field(init=False)
    meso_uncertainty_stem: str = field(init=False)
    large_deltaz_var_stem: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.root_data_dir)
        self.grid_ref_file = str(root / "thetaoa2004.nc")
        self.gtwr_glorys_input_stem = str(root / "02_glorys_as_input" / "SO_largescale_gtwr")
        self.glorys_large_stem = str(root / "04_glorys_data" / "depth{depth}" / "thetaoa_4deg_largescale")
        self.gsvc_beta_file_stem = str(root / "01_beta" / self.gsvc_beta_stem / "beta_gsvc")
        self.gtwr_beta_file_stem = str(root / "01_beta" / self.gtwr_beta_stem / "beta_gtwr")
        self.depth_ref_file = str(root / "data_19940101.nc")
        self.climatology_stem = str(root / "05_clim_data" / "clim_thetao")
        self.obs_stem = str(root / "06_cheng_obs" / "merged_obst")
        self.meso_first_guess_stem = str(root / "03_satel_as_input" / "SO_mesoscale_gsvc")
        self.large_first_guess_stem = str(root / "03_satel_as_input" / "SO_largescale_gsvc")
        self.meso_uncertainty_stem = str(root / "07_meso_uncertainty" / "meso_gsvcres_mse_2017_2020")
        self.large_deltaz_var_stem = str(root / "08_deltaz_var" / "var_large_mse")

    @property
    def coarsening_factor(self) -> int:
        """Integer coarsening factor from input grid to target grid."""
        return int(self.target_resolution / self.input_resolution)


# A module-level config is used so that multiprocessing workers can access it.
CFG: FusionConfig | None = None


# =============================================================================
# Numerical utilities
# =============================================================================


def svd_inverse(matrix: np.ndarray, pseudo: bool = False, threshold: float = 1e-10) -> np.ndarray:
    """Compute the inverse or pseudo-inverse using singular value decomposition."""
    u, s, vt = np.linalg.svd(matrix)

    if pseudo:
        s_inv = np.zeros_like(matrix.T, dtype=float)
        diag = np.where(s > threshold, 1.0 / s, 0.0)
        s_inv[: len(s), : len(s)] = np.diag(diag)
    else:
        if not (matrix.shape[0] == matrix.shape[1] and np.all(s > threshold)):
            raise ValueError("The matrix is not invertible. Use pseudo=True if needed.")
        s_inv = np.diag(1.0 / s)

    return vt.T @ s_inv @ u.T



def ensure_datetime_date(data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    """Convert a YYYYMMDD-like ``date`` coordinate to pandas datetime if needed."""
    if "date" in data.dims and not pd.api.types.is_datetime64_any_dtype(data["date"].dtype):
        data = data.assign_coords(date=pd.to_datetime(data["date"].values.astype(str), format="%Y%m%d"))
    return data



def date_range_yyyymmdd(start_date: str, end_date: str) -> list[str]:
    """Return a list of YYYYMMDD date strings between two inclusive endpoints."""
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    return [(start + pd.Timedelta(days=i)).strftime("%Y%m%d") for i in range((end - start).days + 1)]


# =============================================================================
# Sample construction
# =============================================================================


def residual_of_gtwr_model(
    region: Sequence[float],
    year_range: tuple[int, int],
    depth: int | float,
    current_date: str,
    date_bandwidth: int,
    sca_factor: int,
    gtwr_glorys_input_stem: str,
    glorys_stem_template: str,
) -> xr.DataArray:
    """Build residual samples between GTWR-mapped fields and GLORYS fields.

    These residuals are used to estimate the covariance of temperature
    variability that is not explained by the satellite-to-subsurface mapping.
    """
    llon1, llon2, llat1, llat2 = region
    month = int(current_date[4:6])
    day = int(current_date[6:8])
    if (month, day) == (2, 29):
        raise ValueError("February 29 is excluded from cross-year sampling.")

    glorys_stem = glorys_stem_template.format(depth=depth)

    def open_thetaoa_window(prefix: str, year: int, start: pd.Timestamp, end: pd.Timestamp) -> xr.DataArray:
        file_path = Path(f"{prefix}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
            if "depth" in da.dims or "depth" in da.coords:
                da = da.sel(depth=depth)
            da = ensure_datetime_date(da)
            da = da.sel(date=slice(start, end))
            da = da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()
            return da.load()

    diff_samples: list[xr.DataArray] = []
    for sample_year in range(year_range[0], year_range[1] + 1):
        center = pd.Timestamp(sample_year, month, day)
        start = center - pd.Timedelta(days=date_bandwidth)
        end = center + pd.Timedelta(days=date_bandwidth)

        gtwr_parts, glorys_parts = [], []
        for year in range(start.year, end.year + 1):
            year_start = max(start, pd.Timestamp(f"{year}-01-01"))
            year_end = min(end, pd.Timestamp(f"{year}-12-31"))
            try:
                gtwr_parts.append(open_thetaoa_window(gtwr_glorys_input_stem, year, year_start, year_end))
                glorys_parts.append(open_thetaoa_window(glorys_stem, year, year_start, year_end))
            except FileNotFoundError:
                continue

        if gtwr_parts and glorys_parts:
            gtwr_window = xr.concat(gtwr_parts, dim="date").sortby("date")
            glorys_window = xr.concat(glorys_parts, dim="date").sortby("date")
            diff_samples.append(gtwr_window - glorys_window)

    if not diff_samples:
        raise ValueError("No valid GTWR-GLORYS residual samples were found.")

    return xr.concat(diff_samples, dim="date")



def residual_of_coefficients(
    region: Sequence[float],
    year_range: tuple[int, int],
    depth: int | float,
    current_date: str,
    date_bandwidth: int,
    sca_factor: int,
    gsvc_beta_stem: str,
    gtwr_beta_stem: str,
) -> xr.DataArray:
    """Build coefficient-residual samples between GTWR and GSVC beta fields.

    Multiplying these residual coefficients by the target-date satellite inputs
    gives an ensemble of mapping-function uncertainty, i.e. ``XB``.
    """
    llon1, llon2, llat1, llat2 = region
    month = int(current_date[4:6])
    day = int(current_date[6:8])
    if (month, day) == (2, 29):
        raise ValueError("February 29 is excluded from cross-year sampling.")

    with xr.open_dataset(f"{gsvc_beta_stem}_{depth}.nc") as ds:
        gsvc_beta = ds["beta"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
        # GSVC beta is stored by calendar day; use MMDD to match GTWR samples.
        gsvc_beta = gsvc_beta.assign_coords(date=pd.Index(gsvc_beta["date"].values.astype(str)).str[-4:])
        gsvc_beta = gsvc_beta.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean().load()

    gsvc_dates = pd.Index(gsvc_beta["date"].values.astype(str))

    def open_gtwr_beta_window(year: int, start: pd.Timestamp, end: pd.Timestamp) -> xr.DataArray:
        file_path = Path(f"{gtwr_beta_stem}_y{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            da = ds["beta"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2), depth=depth)
            da = ensure_datetime_date(da)
            da = da.sel(date=slice(start, end))
            da = da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()
            return da.load()

    diff_samples: list[xr.DataArray] = []
    for sample_year in range(year_range[0], year_range[1] + 1):
        center = pd.Timestamp(sample_year, month, day)
        start = center - pd.Timedelta(days=date_bandwidth)
        end = center + pd.Timedelta(days=date_bandwidth)

        gtwr_parts = []
        for year in range(start.year, end.year + 1):
            year_start = max(start, pd.Timestamp(f"{year}-01-01"))
            year_end = min(end, pd.Timestamp(f"{year}-12-31"))
            try:
                gtwr_parts.append(open_gtwr_beta_window(year, year_start, year_end))
            except FileNotFoundError:
                continue

        if not gtwr_parts:
            continue

        gtwr_window = xr.concat(gtwr_parts, dim="date").sortby("date")
        date_mmdd = pd.Index(gtwr_window["date"].dt.strftime("%m%d").values)
        valid = (date_mmdd != "0229") & date_mmdd.isin(gsvc_dates)
        if not valid.any():
            continue

        gtwr_window = gtwr_window.isel(date=valid)
        gsvc_window = gsvc_beta.sel(date=date_mmdd[valid].tolist()).assign_coords(date=gtwr_window["date"])
        diff_samples.append(gtwr_window - gsvc_window)

    if not diff_samples:
        raise ValueError("No valid coefficient residual samples were found.")

    return xr.concat(diff_samples, dim="date")



def prepare_observations_crossyear(
    region: Sequence[float],
    depth: int | float,
    current_date: str,
    obs_bandwidth: tuple[int, int],
    depth_ref_file: str,
    climatology_stem: str,
    obs_stem: str,
    meso_first_guess_stem: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare observation anomalies used by the large-scale fusion.

    The returned observation value is:
        observed temperature - climatology - mesoscale first guess

    This leaves the observation in a large-scale anomaly space consistent with
    the large-scale background field.
    """
    llon1, llon2, llat1, llat2 = region
    center_date = pd.to_datetime(current_date, format="%Y%m%d")
    obs_dates = [center_date + timedelta(days=i) for i in range(-obs_bandwidth[0], obs_bandwidth[1] + 1)]

    with xr.open_dataset(depth_ref_file) as ds:
        depth_grid = ds["depth"].values
    depth_index = int(np.where(np.isclose(depth_grid, depth))[0][0])

    with xr.open_dataset(f"{climatology_stem}{depth_index}.nc") as ds:
        climatology = ds["clim_thetao"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)).load()

    @lru_cache(maxsize=None)
    def open_obs(year: int) -> xr.Dataset:
        file_path = Path(f"{obs_stem}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        with xr.open_dataset(file_path) as ds:
            out = ds.sel(depths=depth)
            out = ensure_datetime_date(out)
            return out.load()

    @lru_cache(maxsize=None)
    def open_meso(year: int) -> xr.DataArray:
        file_path = Path(f"{meso_first_guess_stem}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        with xr.open_dataset(file_path) as ds:
            out = ds["thetaoa"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2), depth=depth)
            out = ensure_datetime_date(out)
            return out.load()

    lon_obs, lat_obs, temp_obs, date_obs = [], [], [], []

    for obs_date in obs_dates:
        try:
            ds_day = open_obs(obs_date.year).sel(date=obs_date)
            meso_day = open_meso(obs_date.year).sel(date=obs_date)
        except (FileNotFoundError, KeyError):
            continue

        mask = (
            (ds_day.lon > llon1)
            & (ds_day.lon < llon2)
            & (ds_day.lat > llat1)
            & (ds_day.lat < llat2)
        )
        obs_index = np.where(mask.values)[0]
        if len(obs_index) == 0:
            continue

        ds_day = ds_day.isel(numobs=obs_index)
        lon = ds_day.lon.values
        lat = ds_day.lat.values
        temp = ds_day.temp.values

        clim_index = pd.Timestamp(year=2012, month=obs_date.month, day=obs_date.day).dayofyear - 1
        meso_at_obs = meso_day.interp(lon=("points", lon), lat=("points", lat)).values
        clim_at_obs = climatology.isel(time=clim_index).interp(lon=("points", lon), lat=("points", lat)).values

        lon_obs.append(lon)
        lat_obs.append(lat)
        temp_obs.append(temp - meso_at_obs - clim_at_obs)
        date_obs.append(np.full(len(lat), obs_date.strftime("%Y%m%d")))

    if not lon_obs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    lon_obs = np.concatenate(lon_obs)
    lat_obs = np.concatenate(lat_obs)
    temp_obs = np.concatenate(temp_obs)
    date_obs = np.concatenate(date_obs)

    valid = ~(np.isnan(lon_obs) | np.isnan(lat_obs) | np.isnan(temp_obs))
    return lon_obs[valid], lat_obs[valid], temp_obs[valid], date_obs[valid]



def build_residual_xb(
    coefficient_residuals: xr.DataArray,
    current_date: str,
    region: Sequence[float],
    sca_factor: int,
    satellite_input_dir: str,
) -> xr.DataArray:
    """Project coefficient residuals onto target-date satellite predictors."""
    llon1, llon2, llat1, llat2 = region
    current_ts = pd.to_datetime(current_date, format="%Y%m%d")

    @lru_cache(maxsize=None)
    def open_satellite_field(file_name: str, var_name: str, year: int) -> xr.DataArray:
        file_path = Path(satellite_input_dir) / f"{file_name}_{year}.nc"
        with xr.open_dataset(file_path) as ds:
            da = ds[var_name].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
            da = ensure_datetime_date(da)
            return da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean().load()

    # The coefficient dimension may be named differently across files.
    coef_dim = [dim for dim in coefficient_residuals.dims if dim not in ("date", "lat", "lon")][0]

    ssta = open_satellite_field("ssta_ostia_4deg_largescale", "ssta", current_ts.year).sel(date=current_ts)
    ssha = open_satellite_field("ssha_4deg_largescale", "ssha", current_ts.year).sel(date=current_ts)

    return coefficient_residuals.isel({coef_dim: 0}) * ssta + coefficient_residuals.isel({coef_dim: 1}) * ssha



def drop_nan_for_analysis(
    background_residuals: xr.DataArray,
    coefficient_error_samples: xr.DataArray,
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    theta_obs: np.ndarray,
    date_obs: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove invalid ensemble samples and observations before assimilation."""
    background_residuals, coefficient_error_samples = xr.align(
        background_residuals, coefficient_error_samples, join="inner"
    )

    sample_bad = (
        np.isnan(background_residuals.values).all(axis=(1, 2))
        | np.isnan(coefficient_error_samples.values).all(axis=(1, 2))
    )
    background_residuals = background_residuals.isel(date=~sample_bad)
    coefficient_error_samples = coefficient_error_samples.isel(date=~sample_bad)

    h_a = background_residuals.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values
    h_xb = coefficient_error_samples.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values

    obs_bad = np.isnan(h_a).any(axis=1) | np.isnan(h_xb).any(axis=1)
    return (
        background_residuals,
        coefficient_error_samples,
        lon_obs[~obs_bad],
        lat_obs[~obs_bad],
        theta_obs[~obs_bad],
        date_obs[~obs_bad],
    )



def build_assimilation_inputs_zeromean_crossyear(
    background_residuals: xr.DataArray,
    coefficient_error_samples: xr.DataArray,
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    date_obs: np.ndarray,
    current_date: str,
    depth: int | float,
    region: Sequence[float],
    sca_factor: int,
    large_first_guess_stem: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct state-space and observation-space arrays for zero-mean fusion.

    ``A`` represents the background residual ensemble. ``XB`` represents the
    coefficient-induced mapping-error ensemble. The first guess ``Y_r`` provides
    the prior mean for the target date.
    """
    llon1, llon2, llat1, llat2 = region
    current_ts = pd.to_datetime(current_date, format="%Y%m%d")
    date_obs_ts = pd.to_datetime(date_obs.astype(str), format="%Y%m%d")
    date_gap_day = (date_obs_ts - current_ts).days.astype(int)

    @lru_cache(maxsize=None)
    def open_large_first_guess(year: int) -> xr.DataArray:
        file_path = Path(f"{large_first_guess_stem}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2), depth=depth)
            da = ensure_datetime_date(da)
            da = da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()
            return da.load()

    a = background_residuals.transpose("date", "lat", "lon").values.reshape(background_residuals.sizes["date"], -1).T
    xb = coefficient_error_samples.transpose("date", "lat", "lon").values.reshape(coefficient_error_samples.sizes["date"], -1).T

    index_nan = np.isnan(a).any(axis=1) | np.isnan(xb).any(axis=1)
    a[index_nan, :] = 0.0
    xb[index_nan, :] = 0.0

    h_a = background_residuals.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values
    h_xb = coefficient_error_samples.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values

    large_now = open_large_first_guess(current_ts.year)
    y_r = large_now.sel(date=current_ts).values.reshape(-1)

    # For cross-year observations, use the same target month-day in the year of
    # each observation. This keeps the background mean consistent with the
    # zero-mean cross-year sample construction.
    def interp_large_at_target_month_day(obs_date: pd.Timestamp, lon: float, lat: float) -> float:
        try:
            target_date = pd.Timestamp(year=obs_date.year, month=current_ts.month, day=current_ts.day)
            return open_large_first_guess(obs_date.year).sel(date=target_date).interp(lon=lon, lat=lat).values.item()
        except Exception:
            return np.nan

    h_y_r = np.array([
        interp_large_at_target_month_day(obs_date, lon, lat)
        for lon, lat, obs_date in zip(lon_obs, lat_obs, date_obs_ts)
    ])

    return a, xb, h_a, h_xb, y_r, h_y_r, index_nan, date_gap_day



def build_error_variances(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    date_gap_day: np.ndarray,
    depth: int | float,
    region: Sequence[float],
    meso_uncertainty_stem: str,
    large_deltaz_var_stem: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build observation-space uncertainty variances.

    ``uncertainty_var`` represents mesoscale uncertainty at observation points.
    ``Cdzdz`` accounts for temporal representativeness error as a function of
    observation-date offset from the analysis date.
    """
    llon1, llon2, llat1, llat2 = region

    with xr.open_dataset(f"{meso_uncertainty_stem}_{depth}m.nc") as ds:
        meso_var = ds["var"].sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2)).load()

    uncertainty_var = meso_var.interp(lon=("points", lon_obs), lat=("points", lat_obs)).values

    with xr.open_dataset(f"{large_deltaz_var_stem}_{depth}.nc") as ds:
        deltaz_var = ds["variance"].load()

    cdzdz = np.array([
        0.0 if gap == 0 else deltaz_var.sel(lag_days=abs(gap)).interp(lon=lon, lat=lat).values
        for lon, lat, gap in zip(lon_obs, lat_obs, date_gap_day)
    ])

    return uncertainty_var, cdzdz



def filter_obs_arrays(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    theta_obs: np.ndarray,
    date_gap_day: np.ndarray,
    h_y_r: np.ndarray,
    cdzdz: np.ndarray,
    uncertainty_var: np.ndarray,
    h_a: np.ndarray,
    h_xb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove observations with missing values in any required term."""
    obs_bad = (
        np.isnan(theta_obs)
        | np.isnan(h_y_r)
        | np.isnan(cdzdz)
        | np.isnan(uncertainty_var)
        | np.isnan(h_a).any(axis=1)
        | np.isnan(h_xb).any(axis=1)
    )

    return (
        lon_obs[~obs_bad],
        lat_obs[~obs_bad],
        theta_obs[~obs_bad],
        date_gap_day[~obs_bad],
        h_y_r[~obs_bad],
        cdzdz[~obs_bad],
        uncertainty_var[~obs_bad],
        h_a[~obs_bad],
        h_xb[~obs_bad],
    )


# =============================================================================
# Sparse localized HB-EnOI solver
# =============================================================================


def gaspari_cohn(r: np.ndarray) -> np.ndarray:
    """Gaspari-Cohn fifth-order compact localization function.

    The input distance ``r`` is nondimensional. The function has compact support
    over [0, 2], which is consistent with the localization cutoff used below.
    """
    r = np.asarray(r, dtype=float)
    w = np.zeros_like(r)

    mask1 = (r >= 0) & (r <= 1)
    rr = r[mask1]
    w[mask1] = -0.25 * rr**5 + 0.5 * rr**4 + 0.625 * rr**3 - (5.0 / 3.0) * rr**2 + 1.0

    mask2 = (r > 1) & (r <= 2)
    rr = r[mask2]
    w[mask2] = (
        (1.0 / 12.0) * rr**5
        - 0.5 * rr**4
        + 0.625 * rr**3
        + (5.0 / 3.0) * rr**2
        - 5.0 * rr
        + 4.0
        - (2.0 / 3.0) / rr
    )

    w[r > 2] = 0.0
    return w



def build_obs_state_bilinear(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    sparse_output: bool = True,
    clip_to_domain: bool = True,
) -> sparse.csr_matrix | np.ndarray:
    """Build a bilinear interpolation matrix from grid space to observation space."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_obs = np.asarray(lon_obs, dtype=float)
    lat_obs = np.asarray(lat_obs, dtype=float)

    n_obs = len(lon_obs)
    n_lon = len(lon)
    n_lat = len(lat)
    n_state = n_lon * n_lat

    if lon_obs.shape != lat_obs.shape:
        raise ValueError("lon_obs and lat_obs must have the same shape.")
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("lon and lat must each have at least two grid points.")

    def get_bracketing_indices(arr: np.ndarray, x: float) -> tuple[int, int, float, float, float]:
        arr_min = min(arr[0], arr[-1])
        arr_max = max(arr[0], arr[-1])
        if clip_to_domain:
            x = float(np.clip(x, arr_min, arr_max))

        if arr[0] < arr[-1]:
            idx_hi = int(np.clip(np.searchsorted(arr, x), 1, len(arr) - 1))
            i0, i1 = idx_hi - 1, idx_hi
            x0, x1 = arr[i0], arr[i1]
        else:
            arr_rev = arr[::-1]
            idx_hi_rev = int(np.clip(np.searchsorted(arr_rev, x), 1, len(arr_rev) - 1))
            j0, j1 = idx_hi_rev - 1, idx_hi_rev
            i0, i1 = len(arr) - 1 - j0, len(arr) - 1 - j1
            x0, x1 = arr[i0], arr[i1]
            if x0 > x1:
                i0, i1 = i1, i0
                x0, x1 = x1, x0
        return i0, i1, float(x0), float(x1), x

    rows = np.empty(4 * n_obs, dtype=int)
    cols = np.empty(4 * n_obs, dtype=int)
    vals = np.empty(4 * n_obs, dtype=float)

    for k in range(n_obs):
        j0, j1, lon0, lon1, x = get_bracketing_indices(lon, lon_obs[k])
        i0, i1, lat0, lat1, y = get_bracketing_indices(lat, lat_obs[k])

        dx = 0.0 if lon1 == lon0 else (x - lon0) / (lon1 - lon0)
        dy = 0.0 if lat1 == lat0 else (y - lat0) / (lat1 - lat0)

        weights = [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ]
        indices = [i0 * n_lon + j0, i0 * n_lon + j1, i1 * n_lon + j0, i1 * n_lon + j1]

        base = 4 * k
        rows[base : base + 4] = k
        cols[base : base + 4] = indices
        vals[base : base + 4] = weights

    h_matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_state)).tocsr()
    return h_matrix if sparse_output else h_matrix.toarray()



def build_sparse_localized_covariance(
    samples: np.ndarray,
    lon_flat: np.ndarray,
    lat_flat: np.ndarray,
    length_scale_lon: float,
    length_scale_lat: float,
    cutoff: float = 2.0,
    search_pad: float = 0.15,
) -> sparse.csr_matrix:
    """Estimate a sparse localized covariance matrix from ensemble samples."""
    samples = np.asarray(samples, dtype=float)
    lon_flat = np.asarray(lon_flat, dtype=float)
    lat_flat = np.asarray(lat_flat, dtype=float)

    n_state, n_samples = samples.shape
    if n_samples < 2:
        raise ValueError(f"At least two samples are required; got {n_samples}.")

    # KD-tree is built in scaled coordinate space so that the query radius is
    # nondimensional and consistent with the GC localization radius.
    coords = np.column_stack([lon_flat / length_scale_lon, lat_flat / length_scale_lat])
    tree = cKDTree(coords)
    search_radius = cutoff + search_pad

    rows, cols, vals = [], [], []
    for i in range(n_state):
        neighbors = tree.query_ball_point(coords[i], r=search_radius)
        neighbors = np.asarray([j for j in neighbors if j >= i], dtype=int)
        if neighbors.size == 0:
            continue

        dlon = (lon_flat[neighbors] - lon_flat[i]) / length_scale_lon
        dlat = (lat_flat[neighbors] - lat_flat[i]) / length_scale_lat
        r = np.sqrt(dlon**2 + dlat**2)
        weights = gaspari_cohn(r)

        keep = (r <= cutoff) & (weights > 0.0)
        if not np.any(keep):
            continue

        neighbors = neighbors[keep]
        weights = weights[keep]
        cov_i = (samples[neighbors, :] @ samples[i, :]) / (n_samples - 1)
        values = cov_i * weights

        rows.extend([i] * len(neighbors))
        cols.extend(neighbors.tolist())
        vals.extend(values.tolist())

    upper = sparse.coo_matrix((vals, (rows, cols)), shape=(n_state, n_state))
    diag = sparse.diags(upper.diagonal())
    return (upper + upper.T - diag).tocsr()



def posterior_variance_diag_sparse(
    m_col: sparse.csr_matrix,
    b_inverse: np.ndarray,
    m_diag: np.ndarray,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Compute the posterior variance diagonal without forming dense state matrices."""
    n_state = m_col.shape[0]
    output = np.empty(n_state, dtype=float)

    for i0 in range(0, n_state, chunk_size):
        i1 = min(i0 + chunk_size, n_state)
        block = m_col[i0:i1, :].toarray()
        output[i0:i1] = m_diag[i0:i1] - np.einsum("ij,jk,ik->i", block, b_inverse, block)

    return np.maximum(output, 0.0)



def solve_analysis_localized_sparse(
    a: np.ndarray,
    xb: np.ndarray,
    y_r: np.ndarray,
    h_y_r: np.ndarray,
    theta_obs: np.ndarray,
    uncertainty_var: np.ndarray,
    cdzdz: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    index_nan: np.ndarray,
    out_shape: tuple[int, int],
    length_scale_yy: tuple[float, float],
    length_scale_bb: tuple[float, float],
    cutoff: float,
    search_pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the localized Bayesian hierarchical EnOI analysis.

    The total prior covariance is represented as the sum of two localized sparse
    covariance matrices:
        M = Cov(A) + Cov(XB)

    where ``A`` captures residual temperature variability and ``XB`` captures
    uncertainty induced by the satellite-to-subsurface mapping coefficients.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = lon_grid.reshape(-1)
    lat_flat = lat_grid.reshape(-1)

    h_matrix = build_obs_state_bilinear(lon, lat, lon_obs, lat_obs, sparse_output=True)

    m_yy = build_sparse_localized_covariance(
        a,
        lon_flat,
        lat_flat,
        length_scale_lon=length_scale_yy[0],
        length_scale_lat=length_scale_yy[1],
        cutoff=cutoff,
        search_pad=search_pad,
    )
    m_bb = build_sparse_localized_covariance(
        xb,
        lon_flat,
        lat_flat,
        length_scale_lon=length_scale_bb[0],
        length_scale_lat=length_scale_bb[1],
        cutoff=cutoff,
        search_pad=search_pad,
    )
    m_total = (m_yy + m_bb).tocsr()

    # Observation-space covariance: B = H M H^T + R.
    b_matrix = (h_matrix @ m_total @ h_matrix.T).toarray()
    b_matrix = 0.5 * (b_matrix + b_matrix.T)
    b_matrix[np.diag_indices_from(b_matrix)] += uncertainty_var + cdzdz

    # A small jitter stabilizes the pseudo-inverse for nearly singular cases.
    jitter = max(1e-8, 1e-6 * np.trace(b_matrix) / b_matrix.shape[0])
    b_matrix[np.diag_indices_from(b_matrix)] += jitter

    b_inverse = svd_inverse(b_matrix, pseudo=True, threshold=1e-6)
    innovation = theta_obs - h_y_r
    m_col = (m_total @ h_matrix.T).tocsr()

    # Posterior mean.
    analysis = y_r + np.asarray(m_col @ (b_inverse @ innovation)).ravel()
    analysis[index_nan] = np.nan
    analysis = analysis.reshape(out_shape)

    # Posterior uncertainty: diagonal of M - M H^T B^-1 H M.
    m_diag = m_total.diagonal()
    error_diag = posterior_variance_diag_sparse(m_col, b_inverse, m_diag, chunk_size=2000)
    error_diag[index_nan] = np.nan
    error = error_diag.reshape(out_shape)

    return analysis, error


# =============================================================================
# End-to-end processing
# =============================================================================


def run_large_scale_fusion_for_depth(args: tuple[str, int | float]) -> tuple[np.ndarray, np.ndarray]:
    """Run large-scale fusion for one date-depth pair."""
    if CFG is None:
        raise RuntimeError("Global configuration CFG has not been initialized.")

    current_date, depth = args
    cfg = CFG
    region = cfg.region
    sca_factor = cfg.coarsening_factor

    print(f"Processing date={current_date}, depth={depth} m")

    # 1. Background residual covariance samples.
    background_residuals = residual_of_gtwr_model(
        region=region,
        year_range=cfg.year_range,
        depth=depth,
        current_date=current_date,
        date_bandwidth=cfg.date_bandwidth_days,
        sca_factor=sca_factor,
        gtwr_glorys_input_stem=cfg.gtwr_glorys_input_stem,
        glorys_stem_template=cfg.glorys_large_stem,
    )

    # 2. Coefficient residual samples.
    coefficient_residuals = residual_of_coefficients(
        region=region,
        year_range=cfg.year_range,
        depth=depth,
        current_date=current_date,
        date_bandwidth=cfg.date_bandwidth_days,
        sca_factor=sca_factor,
        gsvc_beta_stem=cfg.gsvc_beta_file_stem,
        gtwr_beta_stem=cfg.gtwr_beta_file_stem,
    )

    # 3. Observation anomalies in large-scale anomaly space.
    lon_obs, lat_obs, theta_obs, date_obs = prepare_observations_crossyear(
        region=region,
        depth=depth,
        current_date=current_date,
        obs_bandwidth=cfg.obs_bandwidth_days,
        depth_ref_file=cfg.depth_ref_file,
        climatology_stem=cfg.climatology_stem,
        obs_stem=cfg.obs_stem,
        meso_first_guess_stem=cfg.meso_first_guess_stem,
    )

    # 4. Mapping-coefficient uncertainty projected onto satellite predictors.
    coefficient_error_samples = build_residual_xb(
        coefficient_residuals=coefficient_residuals,
        current_date=current_date,
        region=region,
        sca_factor=sca_factor,
        satellite_input_dir=cfg.satellite_input_dir,
    )

    # 5. Remove invalid samples/observations before building matrices.
    background_residuals, coefficient_error_samples, lon_obs, lat_obs, theta_obs, date_obs = drop_nan_for_analysis(
        background_residuals,
        coefficient_error_samples,
        lon_obs,
        lat_obs,
        theta_obs,
        date_obs,
    )

    # 6. Build state-space and observation-space arrays.
    a, xb, h_a, h_xb, y_r, h_y_r, index_nan, date_gap_day = build_assimilation_inputs_zeromean_crossyear(
        background_residuals=background_residuals,
        coefficient_error_samples=coefficient_error_samples,
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        date_obs=date_obs,
        current_date=current_date,
        depth=depth,
        region=region,
        sca_factor=sca_factor,
        large_first_guess_stem=cfg.large_first_guess_stem,
    )

    # 7. Observation and representativeness error variances.
    uncertainty_var, cdzdz = build_error_variances(
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        date_gap_day=date_gap_day,
        depth=depth,
        region=region,
        meso_uncertainty_stem=cfg.meso_uncertainty_stem,
        large_deltaz_var_stem=cfg.large_deltaz_var_stem,
    )

    # The original experiment inflates near-surface uncertainty to avoid
    # overfitting dense or strongly variable near-surface observations.
    if depth in (5, 25):
        uncertainty_var = 16.0 * uncertainty_var

    # 8. Final observation filtering.
    lon_obs, lat_obs, theta_obs, date_gap_day, h_y_r, cdzdz, uncertainty_var, h_a, h_xb = filter_obs_arrays(
        lon_obs,
        lat_obs,
        theta_obs,
        date_gap_day,
        h_y_r,
        cdzdz,
        uncertainty_var,
        h_a,
        h_xb,
    )

    # 9. Sparse localized Bayesian hierarchical EnOI analysis.
    t0 = time.time()
    analysis, error = solve_analysis_localized_sparse(
        a=a,
        xb=xb,
        y_r=y_r,
        h_y_r=h_y_r,
        theta_obs=theta_obs,
        uncertainty_var=uncertainty_var,
        cdzdz=cdzdz,
        lon=background_residuals.lon.values,
        lat=background_residuals.lat.values,
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        index_nan=index_nan,
        out_shape=(background_residuals.sizes["lat"], background_residuals.sizes["lon"]),
        length_scale_yy=cfg.length_scale_yy,
        length_scale_bb=cfg.length_scale_bb,
        cutoff=cfg.localization_cutoff,
        search_pad=cfg.search_pad,
    )
    print(f"Finished date={current_date}, depth={depth} m in {time.time() - t0:.1f} s")

    return analysis, error



def initialize_worker(config: FusionConfig) -> None:
    """Initialize multiprocessing worker state."""
    global CFG
    CFG = config



def get_output_grid(config: FusionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Read the reference grid and return the coarsened output coordinates."""
    llon1, llon2, llat1, llat2 = config.region
    with xr.open_dataset(config.grid_ref_file) as ds:
        grid = ds.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
        grid_coarse = grid.coarsen(lon=config.coarsening_factor, lat=config.coarsening_factor, boundary="trim").mean()
        return grid_coarse.lon.values, grid_coarse.lat.values



def save_daily_result(
    current_date: str,
    depth_levels: Sequence[int | float],
    lon: np.ndarray,
    lat: np.ndarray,
    results: Sequence[tuple[np.ndarray, np.ndarray]],
    config: FusionConfig,
) -> Path:
    """Save one daily multi-depth analysis as a NetCDF file."""
    theta = np.array([item[0] for item in results])
    err = np.array([item[1] for item in results])

    ds_out = xr.Dataset(
        data_vars={
            "thetaoa": (["depth", "lat", "lon"], theta),
            "err": (["depth", "lat", "lon"], err),
        },
        coords={"depth": list(depth_levels), "lat": lat, "lon": lon},
        attrs={
            "title": "Large-scale reconstructed three-dimensional ocean temperature anomaly",
            "framework": "MBH3D large-scale Bayesian hierarchical EnOI fusion",
            "date": current_date,
        },
    )

    save_dir = Path(config.output_dir) / config.model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"large_{current_date}.nc"
    ds_out.to_netcdf(save_file)
    return save_file
