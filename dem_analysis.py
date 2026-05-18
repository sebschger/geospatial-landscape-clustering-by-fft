from __future__ import annotations

import pyfftw
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, transform_bounds, reproject
from pyproj import Geod, CRS, Transformer
from rasterio.enums import Resampling as ResampleEnum
import dask.array as da

# Project specific imports
import common
from settings import INTERNAL_SETTINGS


def _create_fft_tiles(dem: AugmentedDEM):
    """
    This creates an 2D FFT magnitude map for each individual tile of the map.
    """
    fft_input_array = pyfftw.empty_aligned(
        (
            len(dem.tiles_resampled),
            dem.settings["fft"]["tile_size_px"],
            dem.settings["fft"]["tile_size_px"],
        ),
        dtype="complex64",
    )

    fft_output_array = pyfftw.empty_aligned(
        (
            len(dem.tiles_resampled),
            dem.settings["fft"]["tile_size_px"],
            dem.settings["fft"]["tile_size_px"],
        ),
        dtype="complex64",
    )

    fft_execution_plan = pyfftw.FFTW(
        fft_input_array,
        fft_output_array,
        axes=(1, 2),
        direction="FFTW_FORWARD",
        flags=("FFTW_MEASURE",),
    )

    fft_input_array[:] = (
        dem.tiles_resampled
    )  # Important. [:] ensures the reserved empty array is used, and no new one is created! It will not work without that special slicing.

    fft_execution_plan.execute()

    fft_magnitude_spectra_unshifted = np.log(np.abs(fft_output_array) + 1)
    dem.fft_footprint = np.fft.fftshift(
        fft_magnitude_spectra_unshifted, axes=(1, 2)
    )  # Centered FFT

    # Very very rough get a mask excluding the sea
    dem.sealevel_mask = (
        dem.tiles_resampled.max(axis=(1, 2)) > dem.settings["fft"]["sealevel_threshold"]
    )

    dem.tiles_variance = np.var(dem.tiles_resampled, axis=(1, 2))

    # Free memory. The resampled tiles are no longer needed
    del dem.tiles_resampled


# /////// # /////// # /////// # /////// # /////// # /////// # /////// # /////// # ///////


def _resample_original_dem(dem: AugmentedDEM):
    """Takes the original dem and reprojects it to azimuthal equidistant.
    This might lead to inaccuracies at the bounds. However those inaccuracies
    are negligible, that the areas labeled will fall into the right categories."""

    with rasterio.open(dem.dem_path) as src:

        tile_size_px = int(dem.settings["fft"]["tile_size_px"])

        intermediary_geo_bounds = dem.geo_bounds.in_another_crs(
            dem.geo_bounds.intermediary_aeqd_crs
        )
        dst_crs = intermediary_geo_bounds.crs

        target_res = (dem.settings["fft"]["tile_size_km"] * 1000) / dem.settings["fft"][
            "tile_size_px"
        ]
        # Metres per pixel

        (
            scaled_transform,
            projected_dem_width_pixels,
            projected_dem_height_pixels,
        ) = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_res,
            densify_pts=101,
        )

        projected_dem = np.empty(
            (src.count, projected_dem_height_pixels, projected_dem_width_pixels),
            dtype=np.float32,
        )

        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=projected_dem[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=scaled_transform,
                dst_crs=dst_crs,
                resampling=ResampleEnum.bilinear,
            )

        # Set the number of tiles, the projected DEM map will be split into
        tile_multiplier = INTERNAL_SETTINGS["fft"]["tile_overlap_multi"]

        num_tiles_x = int(
            (projected_dem_width_pixels // tile_size_px) * tile_multiplier
        )
        num_tiles_y = int(
            (projected_dem_height_pixels // tile_size_px) * tile_multiplier
        )

        # Set the starting positions of each tile
        start_left = (projected_dem_width_pixels % tile_size_px) // 2
        end_right = projected_dem_width_pixels - start_left - tile_size_px

        start_top = (projected_dem_height_pixels % tile_size_px) // 2
        end_bottom = projected_dem_height_pixels - start_top - tile_size_px

        tile_starts_x = np.linspace(start_left, end_right, num_tiles_x, endpoint=False)
        tile_starts_y = np.linspace(start_top, end_bottom, num_tiles_y, endpoint=False)

        tile_starts_x = tile_starts_x.astype(int)
        tile_starts_y = tile_starts_y.astype(int)

        tile_starts_grid_x, tile_starts_grid_y = np.meshgrid(
            tile_starts_x, tile_starts_y
        )

        tile_starts_grid_xy = np.stack(
            [tile_starts_grid_x.reshape(-1), tile_starts_grid_y.reshape(-1)], axis=1
        )

        # Calculate the total number of tiles
        num_tiles_total = num_tiles_x * num_tiles_y

        # Create a list for the sample positions to later grid-interpolate from them
        dem.tile_centers_orig = []

        # Create the array for the resampled and reprojected tiles that later will be analyzed
        dem.tiles_resampled = np.zeros((num_tiles_total, tile_size_px, tile_size_px))

        intermediary_transformer = Transformer.from_crs(
            dst_crs, dem.geo_bounds.crs, always_xy=True
        )

        # Flip the projected DEM map vertically
        projected_dem[0] = np.flip(projected_dem[0], axis=0)

        # Create a random generator for scattering the sample points a bit
        # This prevents the map from looking very "pixelated" and rather natural
        # It scatters the sample points, not the results, so the results are still accurate
        rng = np.random.default_rng()

        with common.SimpleTimer("Calculating the center positions"):
            for i in range(num_tiles_total):

                # Create randomness as mentioned above for x and y coordinates
                random_shift = rng.uniform(-tile_size_px / 2, tile_size_px / 2, 2)

                # Set up the horizontal bounds of the current tile
                x_base = tile_starts_grid_xy[i, 0] + random_shift[0]
                x_base = common.clamp(x_base, start_left, end_right)
                x_start = x_base
                x_end = x_base + tile_size_px
                x_center = (x_start + x_end) // 2

                # Set up the vertical bounds of the current tile
                y_base = tile_starts_grid_xy[i, 1] + random_shift[1]
                y_base = common.clamp(y_base, start_top, end_bottom)
                y_start = y_base
                y_end = y_base + tile_size_px
                y_center = (y_start + y_end) // 2

                # Extract the tiles
                dem.tiles_resampled[i] = projected_dem[0][y_start:y_end, x_start:x_end]

                # Save the metric centerpoints relative to the center
                # to later infer the geographic position of the samples
                projected_x_center = (
                    x_center - (projected_dem_width_pixels / 2)
                ) * target_res
                projected_y_center = (
                    y_center - (projected_dem_height_pixels / 2)
                ) * target_res

                # Append the geographic coordinates (relative to the source coordinate system)
                # to our list of the tile centers (which are the sample locations)
                # Optimization idea: this could be parallelized
                dem.tile_centers_orig.append(
                    common.GeographicCoordinate(
                        *intermediary_transformer.transform(
                            projected_x_center, projected_y_center
                        ),
                        dem.geo_bounds.crs,
                    )
                )


def _bin_and_average_fft_tiles(dem: AugmentedDEM, circle_masks):
    """
    This splits the FFT magnitude map for each tile into parts that average in a given distance.
    This is essentially the "FFT Footprint".
    The result will be a numpy array in the shape of (number_of_tiles, number_of_fft_levels)
    """

    # Set up Dask Arrays for faster computation of the weighted averages

    # The general dimensions are:
    # 1. Number of tiles (total)
    # 2. Height of a tile
    # 3. Width of a tile
    # 4. Number of levels (circle filters)

    # To multiply and divide all the dimensions have to be in the same order for all arrays

    # Weights – here go the circles
    # Weights are still (#4, #2, #3), change them to #2, #3, #4

    with common.SimpleTimer("Creating the weighted averages per mask"):

        da_weights = da.from_array(
            # The height becomes axis 0, width becomes axis 1, and different masks become axis 2
            np.transpose(circle_masks, (1, 2, 0))
        )

        # First dimension (#1) is missing, add it
        # Just a note: the tiles themselves are not x-y adressed, but the get a continuous index (1D)
        da_weights = da_weights[np.newaxis, ...]
        # Now the form of the weights is (#1, #2, #3, #4) like explained above

        # Take the results of the fft and place them also in a dask array
        # of the wanted form (see above)
        da_magnitude_spectra = da.from_array(dem.fft_footprint)[..., np.newaxis]

        # Sum the values of the areas covered by the circles, to later have something to divide by
        # to generate weighted sums
        da_weights_sum = da.sum(da_weights, axis=(1, 2))

        # Sum the results of the FFTs multiplied by the weights of the different levels (circle filters)
        da_sum_of_spectra = da.sum(da_magnitude_spectra * da_weights, axis=(1, 2))

        # Calculate the weighted average
        da_weighted_averages = da_sum_of_spectra / da_weights_sum

        # The result is now in the shape of #1, #4 -> Number of Tiles, Number of Levels
        dem.fft_magnitude_all_levels = da_weighted_averages.compute()

        del dem.fft_footprint
