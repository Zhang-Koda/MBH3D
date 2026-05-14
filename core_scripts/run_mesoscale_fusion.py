"""
Command-line entry point for MBH3D mesoscale fusion.

The main function is intentionally lightweight. All data preparation,
assimilation, and numerical routines are implemented in
``mesoscale_fusion_utils.py``.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

from mesoscale_fusion_utils import (
    MesoscaleFusionConfig,
    date_range_yyyymmdd,
    get_output_grid,
    initialize_worker,
    run_mesoscale_fusion_for_depth,
    save_daily_result,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mesoscale fusion."""
    parser = argparse.ArgumentParser(
        description="Run the MBH3D mesoscale Bayesian hierarchical EnOI fusion."
    )
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default="./data",
        help="Root directory containing the required MBH3D input data.",
    )
    parser.add_argument(
        "--satellite_input_dir",
        type=str,
        default="./data/satellite_inputs",
        help="Directory containing satellite-derived predictor fields.",
    )
    parser.add_argument(
        "--large_scale_dir",
        type=str,
        default="./outputs/large_scale_fusion",
        help="Directory containing daily large-scale fusion files, e.g. large_YYYYMMDD.nc.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/mesoscale_fusion",
        help="Directory where daily mesoscale fusion files will be saved.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="20210201",
        help="Start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="20210228",
        help="End date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--idate",
        type=int,
        default=0,
        help="Date-block index for job-array or batch submission.",
    )
    parser.add_argument(
        "--dates_per_job",
        type=int,
        default=1,
        help="Number of analysis dates processed by this job.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=12,
        help="Number of multiprocessing workers used over depth levels.",
    )
    return parser.parse_args()


def main() -> None:
    """Run mesoscale fusion for the selected block of dates."""
    args = parse_args()

    config = MesoscaleFusionConfig(
        root_data_dir=args.root_data_dir,
        satellite_input_dir=args.satellite_input_dir,
        large_scale_dir=args.large_scale_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        dates_per_job=args.dates_per_job,
        n_workers=args.n_workers,
    )

    all_dates = date_range_yyyymmdd(config.start_date, config.end_date)
    date_block = all_dates[
        args.idate * config.dates_per_job : (args.idate + 1) * config.dates_per_job
    ]
    if not date_block:
        raise ValueError(f"No dates selected for idate={args.idate}.")

    lon_out, lat_out = get_output_grid(config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    with Pool(config.n_workers, initializer=initialize_worker, initargs=(config,)) as pool:
        for current_date in date_block:
            save_path = Path(config.output_dir) / f"{config.output_prefix}_{current_date}.nc"
            if save_path.exists():
                print(f"Skip existing file: {save_path}")
                continue

            depth_tasks = [(current_date, depth) for depth in config.depth_levels]
            results = pool.map(run_mesoscale_fusion_for_depth, depth_tasks)
            saved_file = save_daily_result(
                current_date=current_date,
                depth_levels=config.depth_levels,
                lon=lon_out,
                lat=lat_out,
                results=results,
                config=config,
            )
            print(f"Saved: {saved_file}")


if __name__ == "__main__":
    main()
