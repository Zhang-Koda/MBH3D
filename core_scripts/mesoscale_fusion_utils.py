"""
Mesoscale fusion utilities for MBH3D.

This module implements the mesoscale component of the MBH3D reconstruction
framework. It is intended for public release: the command-line entry point is
kept in ``run_mesoscale_fusion.py``, while the scientific and numerical routines
are collected here.

Workflow
--------
For each analysis date and depth level, the mesoscale fusion step:
1. Estimates mesoscale background residual samples from GTWR-mapped fields and GLORYS.
2. Estimates coefficient-induced residual samples from GTWR-GSVC beta differences.
3. Removes the large-scale analysis and climatology from in situ observations.
4. Builds zero-mean cross-year assimilation inputs for the mesoscale background.
5. Combines large-scale uncertainty, temporal representativeness error, and
   localized ensemble covariances in a sparse HB-EnOI solver.
6. Saves the mesoscale temperature anomaly and its posterior error variance.

Only formatting, documentation, configurable paths, and public-facing names have
been cleaned; the numerical algorithm follows the research version.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Sequence

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
class MesoscaleFusionConfig:
    """Configuration for the mesoscale HB-EnOI fusion step.

    The defaults are public-release placeholders. Before running the script,
    replace the root directories with paths matching your local data layout.
    """

    root_data_dir: str = "./data"
    output_dir: str = "./outputs/mesoscale_fusion"
    satellite_input_dir: str = "./data/satellite_inputs"
    large_scale_dir: str = "./outputs/large_scale_fusion"

    # Target region: [min_lon, max_lon, min_lat, max_lat]
    region: tuple[float, float, float, float] = (65.0, 120.0, -70.0, -42.0)

    # Mesoscale fusion is performed on the 0.25-degree grid in this version.
    input_resolution: float = 0.25
    target_resolution: float = 0.25

    depth_levels: tuple[int, ...] = (5, 25, 50, 100, 150, 200, 300, 500, 700, 1000, 1500)

    start_date: str = "20210101"
    end_date: str = "20211231"
    year_range: tuple[int, int] = (1993, 2020)
    date_bandwidth_days: int = 60
    obs_bandwidth_days: tuple[int, int] = (10, 6)

    # Localization parameters used by the sparse localized solver.
    length_scale_yy: tuple[float, float] = (1.5, 1.0)
    length_scale_bb: tuple[float, float] = (3.5, 2.5)
    localization_cutoff: float = 2.0
    search_pad: float = 0.15

    n_workers: int = 12
    dates_per_job: int = 1

    # Public, simple output naming. Files are saved as mesoscale_YYYYMMDD.nc.
    output_prefix: str = "mesoscale"

    # File-name stems used by the MBH3D processing pipeline.
    gsvc_beta_stem: str = "gsvc_thetao_SO_50_140_m70_m30_4deg_largescale_bw30_4_3_2truncation_nobeta0_1993_2016"
    gtwr_beta_stem: str = "gtwr_thetao_SO_50_140_m70_m30_Bilateral_4deg_largescale_bw30_4_3_2truncation_nobeta0"

    grid_ref_file: str = field(init=False)
    gtwr_glorys_input_stem: str = field(init=False)
    glorys_meso_stem: str = field(init=False)
    gsvc_beta_file_stem: str = field(init=False)
    gtwr_beta_file_stem: str = field(init=False)
    depth_ref_file: str = field(init=False)
    climatology_stem: str = field(init=False)
    obs_stem: str = field(init=False)
    meso_first_guess_stem: str = field(init=False)
    large_uncertainty_dir: str = field(init=False)
    meso_deltaz_var_stem: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.root_data_dir)
        self.grid_ref_file = str(root / "thetaoa2004.nc")
        self.gtwr_glorys_input_stem = str(root / "02_glorys_as_input" / "SO_mesoscale_gtwr")
        self.glorys_meso_stem = str(root / "04_glorys_data" / "depth{depth}" / "thetaoa_4deg_mesoscale")
        self.gsvc_beta_file_stem = str(root / "01_beta" / self.gsvc_beta_stem / "beta_gsvc")
        self.gtwr_beta_file_stem = str(root / "01_beta" / self.gtwr_beta_stem / "beta_gtwr")
        self.depth_ref_file = str(root / "data_19940101.nc")
        self.climatology_stem = str(root / "05_clim_data" / "clim_thetao")
        self.obs_stem = str(root / "06_cheng_obs" / "merged_obst")
        self.meso_first_guess_stem = str(root / "03_satel_as_input" / "SO_mesoscale_gsvc")
        self.large_uncertainty_dir = self.large_scale_dir
        self.meso_deltaz_var_stem = str(root / "08_deltaz_var" / "var_meso_mse")


_CONFIG: MesoscaleFusionConfig | None = None


def initialize_worker(config: MesoscaleFusionConfig) -> None:
    """Store configuration once in each multiprocessing worker."""
    global _CONFIG
    _CONFIG = config


def date_range_yyyymmdd(start_date: str, end_date: str) -> list[str]:
    """Return all dates between start_date and end_date, inclusive."""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    if end < start:
        raise ValueError("end_date must not be earlier than start_date.")
    return [
        (start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((end - start).days + 1)
    ]


def get_output_grid(config: MesoscaleFusionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Read the target output grid after applying the configured region/coarsening."""
    llon1, llon2, llat1, llat2 = config.region
    sca_factor = int(config.target_resolution / config.input_resolution)
    with xr.open_dataset(config.grid_ref_file) as ds:
        ds_region = ds.sel(lon=slice(llon1, llon2), lat=slice(llat1, llat2))
        ds_out = ds_region.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()
        return ds_out.lon.values.astype(np.float32), ds_out.lat.values.astype(np.float32)

def smooth2d(data, nx, ny):
    """Apply a simple two-dimensional moving-window smoother with NaN handling."""
    N = [nx, ny]
    row, col = data.shape
    extended_data = np.hstack((data[:, -nx:], data, data[:, :nx]))
    extended_data = np.vstack((np.flipud(extended_data[:ny, :]), extended_data, np.flipud(extended_data[-ny:, :])))
    el = sparse.spdiags(np.ones((row + 2 * ny, 2 * N[0] + 1)).transpose(), np.arange(-N[0], N[0] + 1, 1), row + 2 * ny, row + 2 * ny)
    er = sparse.spdiags(np.ones((col + 2 * nx, 2 * N[1] + 1)).transpose(), np.arange(-N[1], N[1] + 1, 1), col + 2 * nx, col + 2 * nx)
    pl = np.isnan(extended_data)
    extended_data[pl] = 0
    nrmlize = el @ ~pl @ er
    nrmlize[pl] = np.nan
    new = el @ extended_data @ er
    new = new / nrmlize
    smoothed_data = new[ny:-ny, nx:-nx]
    return smoothed_data

def svd_inverse(A, pseudo=False, threshold=1e-10):
    """Compute a matrix inverse or pseudo-inverse using singular value decomposition."""
    U, S, Vt = np.linalg.svd(A)
    
    if pseudo:
        S_inv = np.zeros_like(A.T, dtype=float)
        diag = np.where(S > threshold, 1/S, 0.0)
        S_inv[:len(S), :len(S)] = np.diag(diag)
    else:
        if not (A.shape[0] == A.shape[1] and np.all(S > threshold)):
            raise ValueError("The matrix is not invertible!")
        S_inv = np.diag(1.0 / S)
    
    return Vt.T @ S_inv @ U.T

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def residual_of_GTWR_model(
    region,
    year_range,
    depth,
    current_date,
    date_bandwidth,
    sca_factor,
    gtwr_glorys_as_input_name,
    glorys_name,
):
    llon1, llon2, llat1, llat2 = region
    month = int(current_date[4:6])
    day = int(current_date[6:8])

    if (month, day) == (2, 29):
        raise ValueError("current_date = '0229' is not used.")

    def open_thetaoa_window(prefix, year, start_date, end_date):
        file_path = Path(f"{prefix}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2)
            )

            if "depth" in da.dims or "depth" in da.coords:
                da = da.sel(depth=depth)

            if "date" not in da.dims:
                raise ValueError(f"{file_path} has no 'date' dimension, dims={da.dims}")

            if not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
                da = da.assign_coords(
                    date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
                )

            da = da.sel(date=slice(start_date, end_date))

            da = da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()
            return da.load()

    diff_list = []

    for sample_year in range(year_range[0], year_range[1] + 1):
        center_date = pd.Timestamp(sample_year, month, day)
        start_date = center_date - pd.Timedelta(days=date_bandwidth)
        end_date = center_date + pd.Timedelta(days=date_bandwidth)

        gtwr_parts, glorys_parts = [], []

        for y in range(start_date.year, end_date.year + 1):
            y_start = max(start_date, pd.Timestamp(f"{y}-01-01"))
            y_end = min(end_date, pd.Timestamp(f"{y}-12-31"))

            try:
                gtwr_parts.append(open_thetaoa_window(gtwr_glorys_as_input_name, y, y_start, y_end))
                glorys_parts.append(open_thetaoa_window(glorys_name, y, y_start, y_end))
            except FileNotFoundError:
                continue

        if not gtwr_parts or not glorys_parts:
            continue

        gtwr_window = xr.concat(gtwr_parts, dim="date").sortby("date")
        glorys_window = xr.concat(glorys_parts, dim="date").sortby("date")
        diff_list.append(gtwr_window - glorys_window)

    if not diff_list:
        raise ValueError("No valid GTWR-GLORYS residual samples found.")

    return xr.concat(diff_list, dim="date")


def residual_of_coefficients(
    region,
    year_range,
    depth,
    current_date,
    date_bandwidth,
    sca_factor,
    gsvc_beta_name,
    gtwr_beta_name,
):
    llon1, llon2, llat1, llat2 = region
    month = int(current_date[4:6])
    day = int(current_date[6:8])

    if (month, day) == (2, 29):
        raise ValueError("0229 samples are not used.")

    with xr.open_dataset(f"{gsvc_beta_name}_{depth}.nc") as ds:
        gsvc_beta = ds["beta"].sel(
            lon=slice(llon1, llon2),
            lat=slice(llat1, llat2)
        )
        gsvc_beta = gsvc_beta.assign_coords(
            date=pd.Index(gsvc_beta["date"].values.astype(str)).str[-4:]
        )
        gsvc_beta = gsvc_beta.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean().load()

    gsvc_dates = pd.Index(gsvc_beta["date"].values.astype(str))

    def open_gtwr_beta_window(year, start_date, end_date):
        file_path = Path(f"{gtwr_beta_name}_y{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            da = ds["beta"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )

            if "date" not in da.dims:
                raise ValueError(f"{file_path} has no 'date' dimension, dims={da.dims}")

            if not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
                da = da.assign_coords(
                    date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
                )

            da = da.sel(date=slice(start_date, end_date))

            da = da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()

            return da.load()

    diff_list = []

    for sample_year in range(year_range[0], year_range[1] + 1):
        center_date = pd.Timestamp(sample_year, month, day)
        start_date = center_date - pd.Timedelta(days=date_bandwidth)
        end_date = center_date + pd.Timedelta(days=date_bandwidth)

        gtwr_parts = []
        for y in range(start_date.year, end_date.year + 1):
            y_start = max(start_date, pd.Timestamp(f"{y}-01-01"))
            y_end = min(end_date, pd.Timestamp(f"{y}-12-31"))

            try:
                gtwr_parts.append(open_gtwr_beta_window(y, y_start, y_end))
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

        diff_list.append(gtwr_window - gsvc_window)

    if not diff_list:
        raise ValueError("No valid coefficient residual samples found.")
    return xr.concat(diff_list, dim="date")







def prepare_obs_meso(
    region,
    depth,
    current_date,
    obs_bandwidth,
    depth_ref_name,
    clim_name,
    obs_path,
    large_scale_enoi_name,
):
    llon1, llon2, llat1, llat2 = region
    center_date = pd.to_datetime(current_date, format="%Y%m%d")
    obs_dates = [center_date + timedelta(days=i) for i in range(-obs_bandwidth[0], obs_bandwidth[1] + 1)]

    with xr.open_dataset(depth_ref_name) as ds:
        depth_g = ds["depth"].values
    dep_ind = np.where(np.isclose(depth_g, depth))[0][0]

    with xr.open_dataset(f"{clim_name}{dep_ind}.nc") as ds:
        ds_clim = ds["clim_thetao"].sel(
            lon=slice(llon1, llon2),
            lat=slice(llat1, llat2)
        ).load()

    @lru_cache(maxsize=None)
    def open_obs(year):
        file_path = Path(f"{obs_path}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with xr.open_dataset(file_path) as ds:
            da = ds.sel(depths=depth)
            if not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
                da = da.assign_coords(
                    date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
                )
            return da.load()

    @lru_cache(maxsize=None)
    def open_large_enoi_one_day(date_str):
        file_path = Path(f"{large_scale_enoi_name}/large_{date_str}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )
            return da.load()

    lon_obs, lat_obs, thetao_obs, date_obs = [], [], [], []

    for d in obs_dates:
        date_str = d.strftime("%Y%m%d")
        import os
        try:
            ds_day = open_obs(d.year).sel(date=d)
            ds_large = open_large_enoi_one_day(date_str)
        except (FileNotFoundError, KeyError):
            continue

        mask = (
            (ds_day.lon > llon1) & (ds_day.lon < llon2) &
            (ds_day.lat > llat1) & (ds_day.lat < llat2)
        )
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            continue

        ds_day = ds_day.isel(numobs=idx)
        lon1 = ds_day.lon.values
        lat1 = ds_day.lat.values
        temp1 = ds_day.temp.values

        clim_idx = d.replace(year=2012).dayofyear - 1
        temp_large1 = ds_large.interp(lon=("points", lon1), lat=("points", lat1)).values
        temp_clim1 = ds_clim.isel(time=clim_idx).interp(lon=("points", lon1), lat=("points", lat1)).values

        lon_obs.append(lon1)
        lat_obs.append(lat1)
        thetao_obs.append(temp1 - temp_large1 - temp_clim1)
        date_obs.append(np.full(len(lat1), date_str))

    if not lon_obs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    lon_obs = np.concatenate(lon_obs)
    lat_obs = np.concatenate(lat_obs)
    thetao_obs = np.concatenate(thetao_obs)
    date_obs = np.concatenate(date_obs)

    valid = ~(np.isnan(lon_obs) | np.isnan(lat_obs) | np.isnan(thetao_obs))
    return lon_obs[valid], lat_obs[valid], thetao_obs[valid], date_obs[valid]

def build_residual_XB(
    res_Lbeta,
    current_date,
    region,
    sca_factor,
    satellite_input_path,
):
    llon1, llon2, llat1, llat2 = region
    current_ts = pd.to_datetime(current_date, format="%Y%m%d")

    def _to_datetime_date(da):
        if "date" in da.dims and not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
            da = da.assign_coords(
                date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
            )
        return da

    @lru_cache(maxsize=None)
    def open_satellite_field(file_name, var_name, year):
        file_path = Path(f"{satellite_input_path}/{file_name}_{year}.nc")
        with xr.open_dataset(file_path) as ds:
            da = ds[var_name].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2)
            )
            da = _to_datetime_date(da)
            return da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean().load()

    coef_dim = [d for d in res_Lbeta.dims if d not in ("date", "lat", "lon")][0]

    ssta_ostia_large = open_satellite_field(
        "ssta_ostia_4deg_largescale", "ssta", current_ts.year
    ).sel(date=current_ts)

    ssha_aviso_large = open_satellite_field(
        "ssha_4deg_largescale", "ssha", current_ts.year
    ).sel(date=current_ts)

    residual_XB = (
        res_Lbeta.isel({coef_dim: 0}) * ssta_ostia_large +
        res_Lbeta.isel({coef_dim: 1}) * ssha_aviso_large
    )

    return residual_XB

from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr


def build_error_variances_meso(
    lon_obs,
    lat_obs,
    date_obs,
    date_gap_day,
    depth,
    region,
    large_uncertainty_name,
    meso_deltaz_var_name,
):
    llon1, llon2, llat1, llat2 = region
    date_obs_ts = pd.to_datetime(date_obs.astype(str), format="%Y%m%d")

    @lru_cache(maxsize=None)
    def open_large_uncertainty_one_day(date_str):
        file_path = Path(f"{large_uncertainty_name}/large_{date_str}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with xr.open_dataset(file_path) as ds:
            da = ds["err"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )
            return da.load()

    uncertainty_var = np.array([
        open_large_uncertainty_one_day(d.strftime("%Y%m%d")).interp(lon=lon, lat=lat).values.item()
        for lon, lat, d in zip(lon_obs, lat_obs, date_obs_ts)
    ])

    with xr.open_dataset(f"{meso_deltaz_var_name}_{depth}.nc") as ds:
        deltaz_var = ds["variance"].load()

    Cdzdz = np.array([
        0.0 if gap == 0 else deltaz_var.sel(lag_days=abs(gap)).interp(lon=lon, lat=lat).values
        for lon, lat, gap in zip(lon_obs, lat_obs, date_gap_day)
    ])

    return uncertainty_var, Cdzdz

from pathlib import Path
from functools import lru_cache
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr


def prepare_obs_meso_crossyear(
    region,
    depth,
    current_date,
    obs_bandwidth,
    depth_ref_name,
    clim_name,
    obs_path,
    large_scale_enoi_name,
):
    llon1, llon2, llat1, llat2 = region
    center_date = pd.to_datetime(current_date, format="%Y%m%d")
    obs_dates = [center_date + timedelta(days=i) for i in range(-obs_bandwidth[0], obs_bandwidth[1] + 1)]

    with xr.open_dataset(depth_ref_name) as ds:
        depth_g = ds["depth"].values
    dep_ind = np.where(np.isclose(depth_g, depth))[0][0]

    with xr.open_dataset(f"{clim_name}{dep_ind}.nc") as ds:
        ds_clim = ds["clim_thetao"].sel(
            lon=slice(llon1, llon2),
            lat=slice(llat1, llat2)
        ).load()

    def _to_datetime_date(da):
        if "date" in da.dims and not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
            da = da.assign_coords(
                date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
            )
        return da

    @lru_cache(maxsize=None)
    def open_obs(year):
        file_path = Path(f"{obs_path}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with xr.open_dataset(file_path) as ds:
            da = ds.sel(depths=depth)
            da = _to_datetime_date(da)
            return da.load()

    @lru_cache(maxsize=None)
    def open_large_enoi_one_day(date_str):
        file_path = Path(f"{large_scale_enoi_name}/large_{date_str}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )
            return da.load()

    lon_obs, lat_obs, thetao_obs, date_obs = [], [], [], []

    for d in obs_dates:
        date_str = d.strftime("%Y%m%d")

        try:
            ds_obs_year = open_obs(d.year)
            ds_day = ds_obs_year.sel(date=d)
            ds_large = open_large_enoi_one_day(date_str)
        except (FileNotFoundError, KeyError):
            continue

        mask = (
            (ds_day.lon > llon1) & (ds_day.lon < llon2) &
            (ds_day.lat > llat1) & (ds_day.lat < llat2)
        )
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            continue

        ds_day = ds_day.isel(numobs=idx)
        lon1 = ds_day.lon.values
        lat1 = ds_day.lat.values
        temp1 = ds_day.temp.values

        # 按月日映射到 2012 年 climatology，跨年也没问题
        clim_idx = pd.Timestamp(year=2012, month=d.month, day=d.day).dayofyear - 1

        temp_large1 = ds_large.interp(
            lon=("points", lon1),
            lat=("points", lat1)
        ).values

        temp_clim1 = ds_clim.isel(time=clim_idx).interp(
            lon=("points", lon1),
            lat=("points", lat1)
        ).values

        lon_obs.append(lon1)
        lat_obs.append(lat1)
        thetao_obs.append(temp1 - temp_large1 - temp_clim1)
        date_obs.append(np.full(len(lat1), date_str))

    if not lon_obs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    lon_obs = np.concatenate(lon_obs)
    lat_obs = np.concatenate(lat_obs)
    thetao_obs = np.concatenate(thetao_obs)
    date_obs = np.concatenate(date_obs)

    valid = ~(np.isnan(lon_obs) | np.isnan(lat_obs) | np.isnan(thetao_obs))
    return lon_obs[valid], lat_obs[valid], thetao_obs[valid], date_obs[valid]


def drop_nan_for_analysis(
    res_Ldata,
    residual_XB,
    lon_obs,
    lat_obs,
    thetao_obs,
    date_obs,
):
    res_Ldata, residual_XB = xr.align(res_Ldata, residual_XB, join="inner")

    sample_bad = (
        np.isnan(res_Ldata.values).all(axis=(1, 2)) |
        np.isnan(residual_XB.values).all(axis=(1, 2))
    )
    res_Ldata = res_Ldata.isel(date=~sample_bad)
    residual_XB = residual_XB.isel(date=~sample_bad)

    HA = res_Ldata.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values
    HXB = residual_XB.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values

    obs_bad = np.isnan(HA).any(axis=1) | np.isnan(HXB).any(axis=1)

    lon_obs = lon_obs[~obs_bad]
    lat_obs = lat_obs[~obs_bad]
    thetao_obs = thetao_obs[~obs_bad]
    date_obs = date_obs[~obs_bad]

    return res_Ldata, residual_XB, lon_obs, lat_obs, thetao_obs, date_obs

def build_assimilation_inputs(
    res_Ldata,
    residual_XB,
    lon_obs,
    lat_obs,
    date_obs,
    current_date,
    depth,
    region,
    sca_factor,
    large_first_guess_name,
):
    llon1, llon2, llat1, llat2 = region
    current_ts = pd.to_datetime(current_date, format="%Y%m%d")
    date_obs_ts = pd.to_datetime(date_obs.astype(str), format="%Y%m%d")
    date_gap_day = (date_obs_ts - current_ts).days.astype(int)

    def _to_datetime_date(da):
        if "date" in da.dims and not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
            da = da.assign_coords(
                date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
            )
        return da

    @lru_cache(maxsize=None)
    def open_large_first_guess(year):
        file_path = Path(f"{large_first_guess_name}_{year}.nc")
        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )
            da = _to_datetime_date(da)
            return da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean().load()

    A = res_Ldata.transpose("date", "lat", "lon").values.reshape(res_Ldata.sizes["date"], -1).T
    XB = residual_XB.transpose("date", "lat", "lon").values.reshape(residual_XB.sizes["date"], -1).T

    index_nan = np.isnan(A).any(axis=1) | np.isnan(XB).any(axis=1)
    A[index_nan, :] = 0
    XB[index_nan, :] = 0

    HA = res_Ldata.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values
    HXB = residual_XB.interp(lon=("points", lon_obs), lat=("points", lat_obs)).transpose("points", "date").values

    ds_large_now = open_large_first_guess(current_ts.year)
    Y_r = ds_large_now.sel(date=current_ts).values.reshape(-1)

    def interp_large_at(obs_date, lon, lat):
        try:
            return open_large_first_guess(obs_date.year).sel(date=obs_date).interp(lon=lon, lat=lat).values.item()
        except Exception:
            return np.nan

    H_Y_r = np.array([
        interp_large_at(d, lon, lat)
        for lon, lat, d in zip(lon_obs, lat_obs, date_obs_ts)
    ])

    return A, XB, HA, HXB, Y_r, H_Y_r, index_nan, date_gap_day



from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr


def build_assimilation_inputs_zeromean(
    res_Ldata,
    residual_XB,
    lon_obs,
    lat_obs,
    date_obs,
    current_date,
    depth,
    region,
    sca_factor,
    large_first_guess_name,
):
    llon1, llon2, llat1, llat2 = region
    current_ts = pd.to_datetime(current_date, format="%Y%m%d")
    date_obs_ts = pd.to_datetime(date_obs.astype(str), format="%Y%m%d")
    date_gap_day = (date_obs_ts - current_ts).days.astype(int)

    def _to_datetime_date(da):
        if "date" in da.dims and not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
            da = da.assign_coords(
                date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
            )
        return da

    @lru_cache(maxsize=None)
    def open_large_first_guess(year):
        file_path = Path(f"{large_first_guess_name}_{year}.nc")
        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )
            da = _to_datetime_date(da)
            return da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean().load()

    A = res_Ldata.transpose("date", "lat", "lon").values.reshape(res_Ldata.sizes["date"], -1).T
    XB = residual_XB.transpose("date", "lat", "lon").values.reshape(residual_XB.sizes["date"], -1).T

    index_nan = np.isnan(A).any(axis=1) | np.isnan(XB).any(axis=1)
    A[index_nan, :] = 0
    XB[index_nan, :] = 0

    HA = res_Ldata.interp(
        lon=("points", lon_obs),
        lat=("points", lat_obs)
    ).transpose("points", "date").values

    HXB = residual_XB.interp(
        lon=("points", lon_obs),
        lat=("points", lat_obs)
    ).transpose("points", "date").values

    ds_large_now = open_large_first_guess(current_ts.year)
    Y_r = ds_large_now.sel(date=current_ts).values.reshape(-1)

    def interp_large_at_currentdate(obs_date, lon, lat):
        try:
            return open_large_first_guess(obs_date.year).sel(date=current_ts).interp(
                lon=lon, lat=lat
            ).values.item()
        except Exception:
            return np.nan

    H_Y_r = np.array([
        interp_large_at_currentdate(d, lon, lat)
        for lon, lat, d in zip(lon_obs, lat_obs, date_obs_ts)
    ])

    return A, XB, HA, HXB, Y_r, H_Y_r, index_nan, date_gap_day

def filter_obs_arrays(
    lon_obs, lat_obs, thetao_obs, date_gap_day,
    H_Y_r, Cdzdz, uncertainty_var, HA, HXB,
):
    obs_bad = (
        np.isnan(thetao_obs) |
        np.isnan(H_Y_r) |
        np.isnan(Cdzdz) |
        np.isnan(uncertainty_var) |
        np.isnan(HA).any(axis=1) |
        np.isnan(HXB).any(axis=1)
    )

    return (
        lon_obs[~obs_bad],
        lat_obs[~obs_bad],
        thetao_obs[~obs_bad],
        date_gap_day[~obs_bad],
        H_Y_r[~obs_bad],
        Cdzdz[~obs_bad],
        uncertainty_var[~obs_bad],
        HA[~obs_bad],
        HXB[~obs_bad],
    )
    
    
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr


def build_assimilation_inputs_zeromean_crossyear(
    res_Ldata,
    residual_XB,
    lon_obs,
    lat_obs,
    date_obs,
    current_date,
    depth,
    region,
    sca_factor,
    large_first_guess_name,
):
    llon1, llon2, llat1, llat2 = region
    current_ts = pd.to_datetime(current_date, format="%Y%m%d")
    date_obs_ts = pd.to_datetime(date_obs.astype(str), format="%Y%m%d")
    date_gap_day = (date_obs_ts - current_ts).days.astype(int)

    def _to_datetime_date(da):
        if "date" in da.dims and not pd.api.types.is_datetime64_any_dtype(da["date"].dtype):
            da = da.assign_coords(
                date=pd.to_datetime(da["date"].values.astype(str), format="%Y%m%d")
            )
        return da

    @lru_cache(maxsize=None)
    def open_large_first_guess(year):
        file_path = Path(f"{large_first_guess_name}_{year}.nc")
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with xr.open_dataset(file_path) as ds:
            da = ds["thetaoa"].sel(
                lon=slice(llon1, llon2),
                lat=slice(llat1, llat2),
                depth=depth
            )
            da = _to_datetime_date(da)
            da = da.coarsen(lon=sca_factor, lat=sca_factor, boundary="trim").mean()
            return da.load()

    A = res_Ldata.transpose("date", "lat", "lon").values.reshape(res_Ldata.sizes["date"], -1).T
    XB = residual_XB.transpose("date", "lat", "lon").values.reshape(residual_XB.sizes["date"], -1).T

    index_nan = np.isnan(A).any(axis=1) | np.isnan(XB).any(axis=1)
    A[index_nan, :] = 0
    XB[index_nan, :] = 0

    HA = res_Ldata.interp(
        lon=("points", lon_obs),
        lat=("points", lat_obs)
    ).transpose("points", "date").values

    HXB = residual_XB.interp(
        lon=("points", lon_obs),
        lat=("points", lat_obs)
    ).transpose("points", "date").values

    ds_large_now = open_large_first_guess(current_ts.year)
    Y_r = ds_large_now.sel(date=current_ts).values.reshape(-1)

    def interp_large_at_target_md(obs_date, lon, lat):
        try:
            target_date = pd.Timestamp(
                year=obs_date.year,
                month=current_ts.month,
                day=current_ts.day
            )
            return open_large_first_guess(obs_date.year).sel(date=target_date).interp(
                lon=lon, lat=lat
            ).values.item()
        except Exception:
            return np.nan

    H_Y_r = np.array([
        interp_large_at_target_md(d, lon, lat)
        for lon, lat, d in zip(lon_obs, lat_obs, date_obs_ts)
    ])

    return A, XB, HA, HXB, Y_r, H_Y_r, index_nan, date_gap_day
    

def gaspari_cohn(r):
    """
    Gaspari-Cohn localization kernel.
    Compact support on [0, 2].
    """
    r = np.asarray(r, dtype=float)
    w = np.zeros_like(r)

    mask1 = (r >= 0) & (r <= 1)
    rr = r[mask1]
    w[mask1] = (
        -0.25 * rr**5
        + 0.5 * rr**4
        + 0.625 * rr**3
        - (5.0 / 3.0) * rr**2
        + 1.0
    )

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


import numpy as np
from scipy import sparse


def build_obs_state_bilinear(lon, lat, lon_obs, lat_obs, sparse_output=True, clip_to_domain=True):
    """
    Build bilinear interpolation operator H_rec.

    Parameters
    ----------
    lon, lat : 1D arrays
        Model grid coordinates.
    lon_obs, lat_obs : 1D arrays
        Observation coordinates.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_obs = np.asarray(lon_obs, dtype=float)
    lat_obs = np.asarray(lat_obs, dtype=float)

    n_obs = len(lon_obs)
    n_lon = len(lon)
    n_lat = len(lat)
    n_state = n_lon * n_lat

    if lon_obs.shape != lat_obs.shape:
        raise ValueError("lon_obs and lat_obs must have the same shape")
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("lon and lat must each have at least 2 grid points")

    def get_bracketing_indices(arr, x):
        arr_min = min(arr[0], arr[-1])
        arr_max = max(arr[0], arr[-1])

        if clip_to_domain:
            x = np.clip(x, arr_min, arr_max)

        if arr[0] < arr[-1]:  # ascending
            idx_hi = np.searchsorted(arr, x)
            idx_hi = np.clip(idx_hi, 1, len(arr) - 1)
            i0 = idx_hi - 1
            i1 = idx_hi
            x0 = arr[i0]
            x1 = arr[i1]
        else:  # descending
            arr_rev = arr[::-1]
            idx_hi_rev = np.searchsorted(arr_rev, x)
            idx_hi_rev = np.clip(idx_hi_rev, 1, len(arr_rev) - 1)

            j0 = idx_hi_rev - 1
            j1 = idx_hi_rev

            i0 = len(arr) - 1 - j0
            i1 = len(arr) - 1 - j1
            x0 = arr[i0]
            x1 = arr[i1]

            if x0 > x1:
                i0, i1 = i1, i0
                x0, x1 = x1, x0

        return i0, i1, x0, x1, x

    rows = np.empty(4 * n_obs, dtype=int)
    cols = np.empty(4 * n_obs, dtype=int)
    vals = np.empty(4 * n_obs, dtype=float)

    for k in range(n_obs):
        j0, j1, lon0, lon1, x = get_bracketing_indices(lon, lon_obs[k])
        i0, i1, lat0, lat1, y = get_bracketing_indices(lat, lat_obs[k])

        dx = 0.0 if lon1 == lon0 else (x - lon0) / (lon1 - lon0)
        dy = 0.0 if lat1 == lat0 else (y - lat0) / (lat1 - lat0)

        w_ll = (1.0 - dx) * (1.0 - dy)
        w_lr = dx * (1.0 - dy)
        w_ul = (1.0 - dx) * dy
        w_ur = dx * dy

        idx_ll = i0 * n_lon + j0
        idx_lr = i0 * n_lon + j1
        idx_ul = i1 * n_lon + j0
        idx_ur = i1 * n_lon + j1

        base = 4 * k
        rows[base:base + 4] = k
        cols[base:base + 4] = [idx_ll, idx_lr, idx_ul, idx_ur]
        vals[base:base + 4] = [w_ll, w_lr, w_ul, w_ur]

    H_rec = sparse.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_state)).tocsr()

    if sparse_output:
        return H_rec
    return H_rec.toarray()


import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def build_sparse_localized_covariance(
    X,
    lon_flat,
    lat_flat,
    length_scale_lon,
    length_scale_lat,
    cutoff=2.0,
    search_pad=0.15,
):
    """
    Sparse localized covariance using Gaspari-Cohn kernel.

    r = sqrt((dlon/Llon)^2 + (dlat/Llat)^2)
    GC support is on [0, 2].
    """
    X = np.asarray(X, dtype=float)
    lon_flat = np.asarray(lon_flat, dtype=float)
    lat_flat = np.asarray(lat_flat, dtype=float)

    nstate, N = X.shape
    if N < 2:
        raise ValueError(f"Not enough samples: N={N}")

    coords = np.column_stack([
        lon_flat / length_scale_lon,
        lat_flat / length_scale_lat,
    ])

    tree = cKDTree(coords)
    search_radius = cutoff + search_pad

    rows = []
    cols = []
    vals = []

    for i in range(nstate):
        nbrs = tree.query_ball_point(coords[i], r=search_radius)
        nbrs = np.asarray([j for j in nbrs if j >= i], dtype=int)
        if nbrs.size == 0:
            continue

        dlon = (lon_flat[nbrs] - lon_flat[i]) / length_scale_lon
        dlat = (lat_flat[nbrs] - lat_flat[i]) / length_scale_lat
        r = np.sqrt(dlon**2 + dlat**2)

        w = gaspari_cohn(r)

        keep = (r <= cutoff) & (w > 0.0)
        if not np.any(keep):
            continue

        nbrs = nbrs[keep]
        w = w[keep]

        cov_i = (X[nbrs, :] @ X[i, :]) / (N - 1)
        v = cov_i * w

        rows.extend([i] * len(nbrs))
        cols.extend(nbrs.tolist())
        vals.extend(v.tolist())

    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    vals = np.asarray(vals, dtype=float)

    upper = sparse.coo_matrix((vals, (rows, cols)), shape=(nstate, nstate))
    diag = sparse.diags(upper.diagonal())
    M_loc = (upper + upper.T - diag).tocsr()

    return M_loc


import numpy as np


def posterior_variance_diag_sparse(M_col, B_inverse, M_diag, chunk_size=2000):
    """
    Compute posterior variance diagonal:
        diag(M - M H^T B^{-1} H M)

    Here M_col = M H^T, shape (nstate, nobs), sparse.
    """
    nstate = M_col.shape[0]
    out = np.empty(nstate, dtype=float)

    for i0 in range(0, nstate, chunk_size):
        i1 = min(i0 + chunk_size, nstate)
        E = M_col[i0:i1, :].toarray()
        out[i0:i1] = M_diag[i0:i1] - np.einsum("ij,jk,ik->i", E, B_inverse, E)

    return np.maximum(out, 0.0)




import numpy as np


def solve_analysis_localized_sparse(
    A,
    XB,
    Y_r,
    H_Y_r,
    thetao_obs,
    uncertainty_var,
    Cdzdz,
    lon,
    lat,
    lon_obs,
    lat_obs,
    index_nan,
    out_shape,
    length_scale_yy=(1.5, 1.0),
    length_scale_bb=(3.5, 2.5),
    cutoff=2.0,
    search_pad=0.15,
    svd_inverse_func=None,
):
    """
    HBEnOI sparse localized solver:
    - GC localization
    - bilinear observation operator
    - separate localization scales for A and XB
    """
    if svd_inverse_func is None:
        raise ValueError("Please pass svd_inverse_func=svd_inverse")

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_flat = lon_grid.reshape(-1)
    lat_flat = lat_grid.reshape(-1)

    # bilinear H
    H_rec = build_obs_state_bilinear(
        lon=lon,
        lat=lat,
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        sparse_output=True,
    )

    # localized covariances
    M_yy_loc = build_sparse_localized_covariance(
        A,
        lon_flat,
        lat_flat,
        length_scale_lon=length_scale_yy[0],
        length_scale_lat=length_scale_yy[1],
        cutoff=cutoff,
        search_pad=search_pad,
    )

    M_bb_loc = build_sparse_localized_covariance(
        XB,
        lon_flat,
        lat_flat,
        length_scale_lon=length_scale_bb[0],
        length_scale_lat=length_scale_bb[1],
        cutoff=cutoff,
        search_pad=search_pad,
    )

    M_loc = (M_yy_loc + M_bb_loc).tocsr()

    # B = H M H^T + R
    B = (H_rec @ M_loc @ H_rec.T).toarray()
    B = 0.5 * (B + B.T)
    B[np.diag_indices_from(B)] += uncertainty_var + Cdzdz

    # numerical stabilization
    jitter = max(1e-8, 1e-6 * np.trace(B) / B.shape[0])
    B[np.diag_indices_from(B)] += jitter

    B_inverse = svd_inverse_func(B, pseudo=True, threshold=1e-6)
    innovation = thetao_obs - H_Y_r

    # M H^T
    M_col = (M_loc @ H_rec.T).tocsr()

    # posterior mean
    Y = Y_r + np.asarray(M_col @ (B_inverse @ innovation)).ravel()
    Y[index_nan] = np.nan
    Y = Y.reshape(out_shape)

    # posterior variance (diagonal only)
    M_diag = M_loc.diagonal()
    C_err_diag = posterior_variance_diag_sparse(M_col, B_inverse, M_diag, chunk_size=2000)
    C_err_diag[index_nan] = np.nan
    C_err = C_err_diag.reshape(out_shape)

    return Y, C_err

# =============================================================================
# High-level orchestration
# =============================================================================


def run_mesoscale_fusion_for_depth(args: tuple[str, int | float]) -> tuple[np.ndarray, np.ndarray]:
    """Run mesoscale HB-EnOI fusion for one date-depth pair.

    Parameters
    ----------
    args : tuple
        ``(current_date, depth)`` where current_date is formatted as YYYYMMDD.

    Returns
    -------
    tuple of ndarray
        Posterior mesoscale temperature anomaly and posterior error variance.
    """
    if _CONFIG is None:
        raise RuntimeError("Worker configuration has not been initialized.")

    config = _CONFIG
    current_date, depth = args
    depth = float(depth)
    region = config.region
    sca_factor = int(config.target_resolution / config.input_resolution)

    print(f"Processing mesoscale fusion: date={current_date}, depth={depth:g} m")

    t1 = time.time()
    glorys_meso_stem = config.glorys_meso_stem.format(depth=depth)
    res_Ldata = residual_of_GTWR_model(
        region=region,
        year_range=config.year_range,
        depth=depth,
        current_date=current_date,
        date_bandwidth=config.date_bandwidth_days,
        sca_factor=sca_factor,
        gtwr_glorys_as_input_name=config.gtwr_glorys_input_stem,
        glorys_name=glorys_meso_stem,
    )
    print(f"  01 GTWR-GLORYS residual samples: {(time.time() - t1) / 60:.2f} min")

    t1 = time.time()
    res_Lbeta = residual_of_coefficients(
        region=region,
        year_range=config.year_range,
        depth=depth,
        current_date=current_date,
        date_bandwidth=config.date_bandwidth_days,
        sca_factor=sca_factor,
        gsvc_beta_name=config.gsvc_beta_file_stem,
        gtwr_beta_name=config.gtwr_beta_file_stem,
    )
    print(f"  02 coefficient residual samples: {(time.time() - t1) / 60:.2f} min")

    t1 = time.time()
    lon_obs, lat_obs, thetao_obs, date_obs = prepare_obs_meso_crossyear(
        region=region,
        depth=depth,
        current_date=current_date,
        obs_bandwidth=config.obs_bandwidth_days,
        depth_ref_name=config.depth_ref_file,
        clim_name=config.climatology_stem,
        obs_path=config.obs_stem,
        large_scale_enoi_name=config.large_scale_dir,
    )
    print(f"  03 observation preprocessing: {(time.time() - t1) / 60:.2f} min; nobs={len(lon_obs)}")

    t1 = time.time()
    residual_XB = build_residual_XB(
        res_Lbeta=res_Lbeta,
        current_date=current_date,
        region=region,
        sca_factor=sca_factor,
        satellite_input_path=config.satellite_input_dir,
    )
    print(f"  04 coefficient-error ensemble: {(time.time() - t1) / 60:.2f} min")

    t1 = time.time()
    res_Ldata, residual_XB, lon_obs, lat_obs, thetao_obs, date_obs = drop_nan_for_analysis(
        res_Ldata, residual_XB, lon_obs, lat_obs, thetao_obs, date_obs
    )
    print(f"  05 NaN filtering: {(time.time() - t1) / 60:.2f} min; nobs={len(lon_obs)}")

    t1 = time.time()
    A, XB, HA, HXB, Y_r, H_Y_r, index_nan, date_gap_day = build_assimilation_inputs_zeromean_crossyear(
        res_Ldata=res_Ldata,
        residual_XB=residual_XB,
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        date_obs=date_obs,
        current_date=current_date,
        depth=depth,
        region=region,
        sca_factor=sca_factor,
        large_first_guess_name=config.meso_first_guess_stem,
    )
    print(f"  06 assimilation inputs: {(time.time() - t1) / 60:.2f} min")

    t1 = time.time()
    uncertainty_var, Cdzdz = build_error_variances_meso(
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        date_obs=date_obs,
        date_gap_day=date_gap_day,
        depth=depth,
        region=region,
        large_uncertainty_name=config.large_uncertainty_dir,
        meso_deltaz_var_name=config.meso_deltaz_var_stem,
    )
    print(f"  07 error variances: {(time.time() - t1) / 60:.2f} min")

    t1 = time.time()
    lon_obs, lat_obs, thetao_obs, date_gap_day, H_Y_r, Cdzdz, uncertainty_var, HA, HXB = filter_obs_arrays(
        lon_obs, lat_obs, thetao_obs, date_gap_day, H_Y_r, Cdzdz, uncertainty_var, HA, HXB
    )
    print(f"  08 observation filtering: {(time.time() - t1) / 60:.2f} min; nobs={len(lon_obs)}")

    t1 = time.time()
    Y, C_err = solve_analysis_localized_sparse(
        A=A,
        XB=XB,
        Y_r=Y_r,
        H_Y_r=H_Y_r,
        thetao_obs=thetao_obs,
        uncertainty_var=uncertainty_var,
        Cdzdz=Cdzdz,
        lon=res_Ldata.lon.values,
        lat=res_Ldata.lat.values,
        lon_obs=lon_obs,
        lat_obs=lat_obs,
        index_nan=index_nan,
        out_shape=(res_Ldata.sizes["lat"], res_Ldata.sizes["lon"]),
        length_scale_yy=config.length_scale_yy,
        length_scale_bb=config.length_scale_bb,
        cutoff=config.localization_cutoff,
        search_pad=config.search_pad,
        svd_inverse_func=svd_inverse,
    )
    print(f"  09 localized sparse analysis: {(time.time() - t1) / 60:.2f} min")

    return Y, C_err


def save_daily_result(
    current_date: str,
    depth_levels: Sequence[int | float],
    lon: np.ndarray,
    lat: np.ndarray,
    results: Sequence[tuple[np.ndarray, np.ndarray]],
    config: MesoscaleFusionConfig,
) -> Path:
    """Save one daily mesoscale fusion result to NetCDF."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_file = output_dir / f"{config.output_prefix}_{current_date}.nc"
    if save_file.exists():
        print(f"Existing file found; overwriting: {save_file}")

    mean_results = np.asarray([item[0] for item in results])
    err_results = np.asarray([item[1] for item in results])

    ds_save = xr.Dataset(
        data_vars={
            "thetaoa": (["depth", "lat", "lon"], mean_results),
            "err": (["depth", "lat", "lon"], err_results),
        },
        coords={
            "depth": np.asarray(depth_levels),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "MBH3D mesoscale temperature anomaly fusion result",
            "description": "Mesoscale component generated by the HB-EnOI fusion framework.",
        },
    )
    ds_save.to_netcdf(save_file)
    return save_file
