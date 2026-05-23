from __future__ import annotations

import pyfftw
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, transform_bounds, reproject
from pyproj import Geod, CRS, Transformer
from rasterio.enums import Resampling as ResampleEnum
import dask.array as da
from shapely.ops import transform as shapely_transform
from shapely.geometry import Polygon
from rasterio.mask import geometry_mask

# Projektspezifische Importe
import common
from settings import INTERNAL_SETTINGS

import cv2


def densified_box(left, bottom, right, top, points_per_side=20):
    """Erstellt ein Rechteck mit extra Punkten an den Seitenkanten."""

    top_edge = list(
        zip(np.linspace(left, right, points_per_side), [top] * points_per_side)
    )
    right_edge = list(
        zip([right] * points_per_side, np.linspace(top, bottom, points_per_side))
    )
    bottom_edge = list(
        zip(np.linspace(right, left, points_per_side), [bottom] * points_per_side)
    )
    left_edge = list(
        zip([left] * points_per_side, np.linspace(bottom, top, points_per_side))
    )

    return Polygon(top_edge + right_edge + bottom_edge + left_edge)


def _create_fft_tiles(dem: AugmentedDEM):
    """
    Erstellt eine 2D-FFT-Magnitudenkarte für jede einzelne Kachel der Karte.
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

    fft_input_array[:] = dem.tiles_resampled - dem.tiles_resampled.mean(
        axis=(1, 2), keepdims=True
    )

    # Wichtig: [:] stellt sicher, dass das reservierte Array verwendet und kein neues erstellt wird! Ohne dieses spezielle Slicing funktioniert es nicht.

    fft_execution_plan.execute()

    fft_magnitude_spectra_unshifted = np.log(np.abs(fft_output_array) + 1)
    dem.fft_footprint = np.fft.fftshift(
        fft_magnitude_spectra_unshifted, axes=(1, 2)
    )  # Centered FFT

    # Sehr grobe Maske, die Meeresflächen ausschließt
    dem.sealevel_mask = (
        dem.tiles_resampled.max(axis=(1, 2)) > dem.settings["fft"]["sealevel_threshold"]
    )

    dem.tiles_variance = np.var(dem.tiles_resampled, axis=(1, 2))

    # Speicher freigeben. Die neu abgetasteten Kacheln werden nicht mehr benötigt
    del dem.tiles_resampled


# /////// # /////// # /////// # /////// # /////// # /////// # /////// # /////// # ///////


def _resample_original_dem(dem: AugmentedDEM):
    """Reprojiziiert das originale DEM auf azimutale Äquidistanz.
    Dies kann zu Ungenauigkeiten an den Rändern führen. Diese Ungenauigkeiten sind
    jedoch vernachlässigbar, sodass die bezeichneten Flächen in die richtigen Kategorien fallen."""

    with rasterio.open(dem.dem_path) as src:

        tile_size_px = int(dem.settings["fft"]["tile_size_px"])

        intermediary_geo_bounds = dem.geo_bounds.in_another_crs(
            dem.geo_bounds.intermediary_aeqd_crs
        )
        dem.aeqd_crs = intermediary_geo_bounds.crs

        dem.aeqd_target_res_m = (
            dem.settings["fft"]["tile_size_km"] * 1000
        ) / dem.settings["fft"]["tile_size_px"]
        # Meter pro Pixel

        (
            dem.aeqd_transform,
            dem.aeqd_width_px,
            dem.aeqd_height_px,
        ) = calculate_default_transform(
            src.crs,
            dem.aeqd_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=dem.aeqd_target_res_m,
            densify_pts=101,
        )

        dem.projected_dem = np.empty(
            (src.count, dem.aeqd_height_px, dem.aeqd_width_px),
            dtype=np.float32,
        )

        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=dem.projected_dem[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dem.aeqd_transform,
                dst_crs=dem.aeqd_crs,
                resampling=ResampleEnum.bilinear,
            )

        # Anzahl der Kacheln festlegen, in die die projizierte DEM-Karte aufgeteilt wird
        tile_multiplier = INTERNAL_SETTINGS["fft"]["tile_overlap_multi"]

        num_tiles_x = int((dem.aeqd_width_px // tile_size_px) * tile_multiplier)
        num_tiles_y = int((dem.aeqd_height_px // tile_size_px) * tile_multiplier)

        # Startpositionen jeder Kachel festlegen
        start_left = (dem.aeqd_width_px % tile_size_px) // 2
        end_right = dem.aeqd_width_px - start_left - tile_size_px

        start_top = (dem.aeqd_height_px % tile_size_px) // 2
        end_bottom = dem.aeqd_height_px - start_top - tile_size_px

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

        # Gesamtanzahl der Kacheln berechnen
        num_tiles_total = num_tiles_x * num_tiles_y

        # Liste der Stichprobenpositionen erstellen, um später daraus zu interpolieren
        dem.tile_centers_orig = []

        intermediary_transformer = Transformer.from_crs(
            dem.aeqd_crs, dem.geo_bounds.crs, always_xy=True
        )

        # Projizierte DEM-Karte vertikal spiegeln
        dem.projected_dem[0] = np.flip(dem.projected_dem[0], axis=0)

        # Gültigkeitsmaske erstellen: True für Pixel im gültigen Bereich, False an den Rändern
        # (wo die Reprojektion Daten „erfunden" hat)

        src_bbox = densified_box(
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            points_per_side=101,
        )

        transformer = Transformer.from_crs(src.crs, dem.aeqd_crs, always_xy=True)
        dem.bbox_aeqd = shapely_transform(transformer.transform, src_bbox)

        # Bounding-Box schrumpfen, damit Modus-Ränder korrigiert werden
        mode_filter_radius_m = (
            INTERNAL_SETTINGS["output"]["label_mode_filter_radius_km"] * 1000 * 2.65
        )

        dem.bbox_aeqd_safe = dem.bbox_aeqd.buffer(-mode_filter_radius_m)

        dem.validity_mask_output = geometry_mask(
            [dem.bbox_aeqd_safe],
            out_shape=(dem.aeqd_height_px, dem.aeqd_width_px),
            transform=dem.aeqd_transform,
            invert=True,  # True = valid, False = invalid
        )

        with common.SimpleTimer("Calculating the center positions"):
            # Tile-Zentren als Pixelposition (im geflippten Array)
            x_centers = tile_starts_grid_xy[:, 0] + tile_size_px // 2
            y_centers = tile_starts_grid_xy[:, 1] + tile_size_px // 2

            # Tiles aus dem geflippten Array schneiden
            dem.tiles_resampled = np.stack(
                [
                    dem.projected_dem[0][y : y + tile_size_px, x : x + tile_size_px]
                    for x, y in zip(
                        tile_starts_grid_xy[:, 0], tile_starts_grid_xy[:, 1]
                    )
                ]
            )

            # ECHTE AEQD-Koordinaten aus dem Transform (bezogen aufs Projektionszentrum)
            # x: kein Flip. y: Flip rückgängig, da y_centers im geflippten Array liegen.
            res = dem.aeqd_target_res_m

            projected_x_centers = dem.aeqd_transform.c + (x_centers + 0.5) * res
            projected_y_centers = (
                dem.aeqd_transform.f - (dem.aeqd_height_px - 0.5 - y_centers) * res
            )

            # Diese echten Koordinaten gehen in BEIDE Listen
            dem.tile_centers_aeqd = list(zip(projected_y_centers, projected_x_centers))

            # Ein einziger Batch-Transform-Aufruf statt num_tiles einzelner Aufrufe
            xs_orig, ys_orig = intermediary_transformer.transform(
                projected_x_centers, projected_y_centers
            )

            dem.tile_centers_orig = [
                common.GeographicCoordinate(x, y, dem.geo_bounds.crs)
                for x, y in zip(xs_orig, ys_orig)
            ]


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
