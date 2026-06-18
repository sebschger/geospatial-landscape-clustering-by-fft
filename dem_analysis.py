from __future__ import annotations

import pyfftw
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, transform_bounds, reproject
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.transform import from_origin
from rasterio.features import geometry_mask as rio_geometry_mask
from pyproj import Geod, CRS, Transformer
from rasterio.enums import Resampling as ResampleEnum
import dask.array as da
from shapely.ops import unary_union, transform as shapely_transform
from shapely.ops import transform as shapely_transform_v2
from shapely.geometry import Polygon, shape
from shapely import contains_xy
from rasterio.mask import geometry_mask
import geopandas as gpd
import pandas as pd
import os
import hashlib
import time
import pickle

from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

import umap
from hdbscan import HDBSCAN

import plotly.graph_objects as go

# Projektspezifische Importe
import common
from common import GeographicBounds
from settings import INTERNAL_SETTINGS
from geo_colors import custom_colors_medium

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

    pyfftw.config.NUM_THREADS = 10


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
        flags=("FFTW_MEASURE","FFTW_DESTROY_INPUT"),
    )

    # Neu: Hanning Filter für FFT
    hanning_1d =  np.hanning(dem.settings["fft"]["tile_size_px"])
    hanning_2d = np.outer(hanning_1d,hanning_1d)

    fft_input_array[:] = dem.tiles_resampled - dem.tiles_resampled.mean(
        axis=(1, 2), keepdims=True
    ) * hanning_2d

    # Wichtig: [:] stellt sicher, dass das reservierte Array verwendet und kein neues erstellt wird! 
    # Ohne dieses spezielle Slicing funktioniert es nicht.

    fft_execution_plan.execute()

    print("FFT Plan executed.")

    fft_magnitude_spectra_unshifted = np.log(np.abs(fft_output_array) + 1)
    
    # Speicher Freigeben
    del fft_output_array
    del fft_input_array
    del fft_execution_plan

    dem.fft_footprint = np.fft.fftshift(
        fft_magnitude_spectra_unshifted, axes=(1, 2)
    )  # Centered FFT

    # Sehr grobe Maske, die Meeresflächen vom FFT ausschließt
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


class AugmentedDEM:
    """Container für alle Informationen zu einem einzelnen DEM-GeoTIFF.

    Koordiniert den gesamten Verarbeitungspfad: Reprojektion → FFT → Binning → Clustering → Ausgabe.
    """

    # Wird vor der Verarbeitung gesetzt (gemeinsam für alle Instanzen)
    settings = None

    def __init__(self, dem_path: str):
        """Liest die wichtigsten Metadaten aus dem GeoTIFF (Bounds, Dimensionen, CRS).

        Die eigentlichen Höhendaten werden erst beim Verarbeiten geladen.
        """
        with rasterio.open(dem_path) as dem_file:
            self.src_crs = dem_file.crs

            self.geo_bounds = GeographicBounds(
                dem_file.bounds.left,
                dem_file.bounds.bottom,
                dem_file.bounds.right,
                dem_file.bounds.top,
                dem_file.crs,
                dem_file.transform,
            )

            self.dem_width_px = dem_file.width
            self.dem_height_px = dem_file.height
            self.tile_centers_aeqd = []

        self.dem_path = dem_path

    def process_dem_fast(self, circle_masks: np.ndarray):
        """Führt die Verarbeitungspipeline aus: Resampling → FFT → Binning.

        Args:
            circle_masks: Ringmasken-Array aus common.RingImageSeriesLog, wird für das Binning benötigt.
        """
        with common.SimpleTimer("Resampling the original DEM to azimuthal equidistant"):
            _resample_original_dem(self)

        with common.SimpleTimer("Creating FFT footprints"):
            _create_fft_tiles(self)

        with common.SimpleTimer("Binning and averaging FFT footprints"):
            _bin_and_average_fft_tiles(self, circle_masks)

    @classmethod
    def create_labels(cls, instances: list[AugmentedDEM]):
        """Führt K-Means-Clustering über alle DEM-Instanzen durch und schreibt Labels zurück.

        Das Clustering berücksichtigt Land- und Meerestiles getrennt, um eine ausgewogene
        Stichprobenverteilung zu gewährleisten.
        """
        all_magnitudes_np = np.concatenate(
            [d.fft_magnitude_all_levels.astype(np.float32) for d in instances], axis=0
        )
        all_sealevel_mask = np.concatenate([d.sealevel_mask for d in instances], axis=0)

        # Originale Tile-Anzahl merken, um Labels hinterher wieder aufzuteilen
        original_tile_count = [d.fft_magnitude_all_levels.shape[0] for d in instances]
        original_indices_for_splitting = np.cumsum(original_tile_count[:-1])

        n_clusters = cls.settings["output"]["label_count"]
        k_means_clusterer = KMeans(n_clusters=n_clusters, random_state=550)

        with common.SimpleTimer("Clustering"):
            all_magnitudes_np = np.nan_to_num(all_magnitudes_np, nan=0)

            all_magnitudes_land = all_magnitudes_np[all_sealevel_mask]
            all_magnitudes_sea = all_magnitudes_np[~all_sealevel_mask]

            print(f"Number of all magnitudes: {all_magnitudes_np.shape[0]}")

            max_samples_land = 90000
            max_samples_sea = 9000

            random_sample_size_land = min(int(all_magnitudes_land.shape[0] * 0.25), max_samples_land)
            random_magnitude_ids_land = np.random.choice(
                all_magnitudes_land.shape[0], size=random_sample_size_land, replace=False
            )

            random_sample_size_sea = min(int(all_magnitudes_sea.shape[0] * 0.01), max_samples_sea)
            random_magnitude_ids_sea = np.random.choice(
                all_magnitudes_sea.shape[0], size=random_sample_size_sea, replace=False
            )

            print(f"Using {random_sample_size_land} land samples...")

            with common.SimpleTimer("K-Means explicitly"):
                # Training nur auf Landtiles, Vorhersage auf alle
                k_means_clusterer.fit(all_magnitudes_land)
                all_dem_labels = k_means_clusterer.predict(all_magnitudes_np)

        with common.SimpleTimer("Sorting labels by amplitude of their contents"):
            # Labels nach mittlerer FFT-Amplitude sortieren (niedrig = flach, hoch = rau)
            variances_per_label = {
                label: np.mean(all_magnitudes_np[np.where(all_dem_labels == label)])
                for label in np.unique(all_dem_labels)
            }
            sorted_keys = sorted(variances_per_label, key=lambda k: variances_per_label[k])
            mapping = {key: rank for rank, key in enumerate(sorted_keys)}
            lookup = np.array([mapping[i] for i in range(max(mapping.keys()) + 1)])
            all_dem_labels = lookup[all_dem_labels]

        # Labels auf die einzelnen DEM-Instanzen verteilen
        single_dem_labels = np.split(all_dem_labels, original_indices_for_splitting)
        for i, labels in enumerate(single_dem_labels):
            instances[i].labels = labels

    def create_image(self):
        """Erzeugt ein eingefärbtes Rasterbild aus den geclusterten Labels.

        Labels werden per Nearest-Neighbor-Interpolation auf das AEQD-Pixelraster projiziert
        und danach mit einem Modus-Filter geglättet.
        """
        res = self.aeqd_target_res_m
        grid_x = self.aeqd_transform.c + (np.arange(self.aeqd_width_px) + 0.5) * res
        grid_y = self.aeqd_transform.f - (np.arange(self.aeqd_height_px) + 0.5) * res

        # indexing="ij": erste Achse = Zeilen (y), zweite Achse = Spalten (x)
        tmp_grid_y, tmp_grid_x = np.meshgrid(grid_y, grid_x, indexing="ij")

        output_image = griddata(
            self.tile_centers_aeqd,
            self.labels,
            (tmp_grid_y, tmp_grid_x),
            method="nearest",
        )

        with common.SimpleTimer("Filtering of the mapped labels"):
            filter_radius_px = (
                self.settings["output"]["label_mode_filter_radius_km"]
                * 1000
                / self.aeqd_target_res_m
            )
            self.labels_mode_filtered = common.mode_filter(output_image, filter_radius_px)

        # Bereiche außerhalb des gültigen DEM-Bereichs auf -1 setzen
        self.labels_mode_filtered[~self.validity_mask_output.astype(bool)] = -1

        color_array = common.color_to_numpy(custom_colors_medium)
        output_image_rgb = color_array[common.constrain_labels(self.labels_mode_filtered)]
        self.image_rgb = np.transpose(output_image_rgb, (2, 0, 1)).astype("uint8")

    def create_image_unfiltered(self):
        """Erzeugt ein eingefärbtes Rasterbild ohne Modus-Filter (für Vergleiche nützlich)."""
        res = self.aeqd_target_res_m
        grid_x = self.aeqd_transform.c + (np.arange(self.aeqd_width_px) + 0.5) * res
        grid_y = self.aeqd_transform.f - (np.arange(self.aeqd_height_px) + 0.5) * res

        tmp_grid_y, tmp_grid_x = np.meshgrid(grid_y, grid_x, indexing="ij")

        output_image = griddata(
            self.tile_centers_aeqd,
            self.labels,
            (tmp_grid_y, tmp_grid_x),
            method="nearest",
        )

        self.output_image_unfiltered = output_image
        self.output_image_unfiltered[~self.validity_mask_output.astype(bool)] = -1

        color_array = common.color_to_numpy(custom_colors_medium)
        output_image_rgb = color_array[common.constrain_labels(self.output_image_unfiltered)]
        self.image_rgb = np.transpose(output_image_rgb, (2, 0, 1)).astype("uint8")

    def write_image(self):
        """Schreibt das eingefärbte AEQD-Rasterbild als GeoTIFF auf die Festplatte.

        Der Dateiname enthält die wichtigsten Settings, damit Ergebnisse unterschiedlicher
        Läufe leicht auseinanderzuhalten sind.
        """
        # Kurzer Hash zur eindeutigen Kennzeichnung dieses Laufs
        h = hashlib.sha1(str(time.time()).encode()).hexdigest()[:4]

        self.transform_scaled = self.aeqd_transform * Affine.scale(1)

        filename_string = (
            os.path.basename(self.dem_path) + " "
            f"{h}  "
            f"tlszkm {self.settings['fft']['tile_size_km']:.1f}  "
            f"tlszpx {self.settings['fft']['tile_size_px']:.0f}  "
            f"fftlvls {self.settings['fft']['fft_levels']:.0f}  "
            f"{self.settings['fft']['tile_overlap_multi']:.1f}x "
            f"-ovrlp-pct {self.settings['fft']['tile_overlap_percent']:.0f} "
            f"fltrrds {self.settings['output']['label_mode_filter_radius_km']:.0f} "
            f"lblct {self.settings['output']['label_count']:.0f}"
            f".tif"
        )

        filepath = os.path.join(self.settings["output"]["folder_name"], filename_string)

        with rasterio.open(
            filepath,
            "w",
            driver="GTiff",
            height=self.aeqd_height_px,
            width=self.aeqd_width_px,
            count=3,
            dtype="uint8",
            crs=self.aeqd_crs,
            transform=self.transform_scaled,
        ) as dst:
            dst.write(self.image_rgb[0], 1)
            dst.write(self.image_rgb[1], 2)
            dst.write(self.image_rgb[2], 3)

    @classmethod
    def find_most_representative_tiles(cls, instances: list[AugmentedDEM]):
        """Findet geografisch diverse, typische Tile-Positionen pro Cluster.

        Ablauf je Label:
          1. Die FFT-median-nächsten Tiles auswählen (typische Vertreter)
          2. Per HDBSCAN geografisch clustern (um Wiederholungen zu vermeiden)
          3. Den Medoid jeder Geo-Gruppe als Repräsentanten wählen
          4. Die geografisch am weitesten verteilten Kandidaten bevorzugen

        Returns:
            (positions_per_label, variances_per_label): je ein dict label -> np.ndarray
        """
        all_tiles_variance = np.concatenate([d.tiles_variance for d in instances])
        all_sealevel_mask = np.concatenate([d.sealevel_mask for d in instances], axis=0)

        land_tiles_ids = np.where(all_sealevel_mask)[0]
        sea_tiles_ids = np.where(~all_sealevel_mask)[0]

        max_samples_land = 3000000
        random_sample_size_land = min(max_samples_land, len(land_tiles_ids))
        print(f"Calculating using {random_sample_size_land} land samples")

        sampled_land_ids = np.random.choice(land_tiles_ids, size=random_sample_size_land, replace=False)
        # Meerestiles werden in dieser Phase nicht berücksichtigt
        sampled_combined_ids = sampled_land_ids

        all_labels = np.concatenate([d.labels for d in instances], axis=0)
        all_magnitudes = np.concatenate(
            [d.fft_magnitude_all_levels.astype(np.float32) for d in instances], axis=0
        )
        all_tile_centers = np.concatenate([d.tile_centers_orig for d in instances], axis=0)
        all_tile_positions = np.stack([c.to_numpy() for c in all_tile_centers])

        # Auf die Stichprobe reduzieren
        all_labels = all_labels[sampled_combined_ids]
        all_magnitudes = all_magnitudes[sampled_combined_ids]
        all_tile_centers = all_tile_centers[sampled_combined_ids]
        all_tile_positions = all_tile_positions[sampled_combined_ids]
        all_tiles_variance = all_tiles_variance[sampled_combined_ids]

        unique_labels = np.unique(all_labels)
        representative_position_per_label = {}
        representative_variance_per_label = {}

        for label in unique_labels:
            if label == -1:
                continue

            label_indices = np.where(all_labels == label)[0]
            label_count = len(label_indices)

            # Top-1% der FFT-median-nächsten Tiles als Kandidaten
            n_candidates = int(0.01 * label_count)
            candidate_ids_in_label = common.closest_to_median_indices(
                all_magnitudes[label_indices]
            )[:n_candidates]

            candidate_ids_global = label_indices[candidate_ids_in_label]
            candidate_positions = all_tile_positions[candidate_ids_global]
            candidate_variances = all_tiles_variance[candidate_ids_global]

            # Geografisches Clustering der Kandidaten (Haversine-Distanz)
            hdb = HDBSCAN(min_cluster_size=10, metric="haversine")
            candidate_geo_groups = hdb.fit_predict(
                np.radians(np.flip(candidate_positions, axis=1))
            )

            geo_groups, geo_group_sizes = np.unique(candidate_geo_groups, return_counts=True)
            valid = geo_groups != -1
            valid_geo_groups = geo_groups[valid]
            valid_geo_group_sizes = geo_group_sizes[valid]

            if len(valid_geo_groups) == 0:
                continue

            top_geo_groups = valid_geo_groups[np.argsort(valid_geo_group_sizes)[::-1]]
            print(f"We have {len(top_geo_groups)} representative candidates")

            how_many_wide_spread = 3
            representative_positions = []
            representative_variances = []

            for top_geo_group in top_geo_groups:
                top_group_ids = np.where(candidate_geo_groups == top_geo_group)[0]
                top_group_positions = candidate_positions[top_group_ids]
                top_group_variances = candidate_variances[top_group_ids]

                # Medoid = zentralster Punkt der Gruppe
                medoid_id = common.medoid_indices(top_group_positions)[0]
                representative_positions.append(top_group_positions[medoid_id])
                representative_variances.append(top_group_variances[medoid_id])

            # Die geografisch am weitesten voneinander entfernten bevorzugen
            furthest_ids = common.sort_by_furthest(representative_positions)
            representative_positions = np.array(representative_positions)
            representative_variances = np.array(representative_variances)

            representative_position_per_label[label] = representative_positions[furthest_ids[:how_many_wide_spread]]
            representative_variance_per_label[label] = representative_variances[furthest_ids[:how_many_wide_spread]]

        return representative_position_per_label, representative_variance_per_label

    @classmethod
    def visualize_embeddings(cls, instances: list[AugmentedDEM]):
        """Visualisiert die FFT-Embeddings aller Tiles interaktiv in 3D via UMAP."""
        fig = go.Figure()
        fig.update_layout(height=1000, width=1000)

        all_magnitudes_np = np.concatenate(
            [d.fft_magnitude_all_levels.astype(np.float32) for d in instances], axis=0
        )

        vis_umap = umap.UMAP(n_components=3, n_neighbors=25)
        umap_result = vis_umap.fit_transform(all_magnitudes_np)

        all_labels_np = np.concatenate(
            [d.labels.ravel().astype(np.float32) for d in instances], axis=0
        )

        fig.add_scatter3d(
            x=umap_result[:, 0],
            y=umap_result[:, 1],
            z=umap_result[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.03, color=all_labels_np),
            name="Interpolated Labels",
        )

        fig.show()


# /////// # /////// # /////// # /////// # /////// # /////// # /////// # /////// # ///////


def return_combined_shapes(filtered_labels, combined_transform, combined_aeqd_crs,
                           output_path="output/labels_geojson/combined_map_labels.geojson"):
    """
    Erstellt aus der zusammengefassten Label-Map (combined_map) GeoJSON-Polygone
    pro Label und speichert sie als GeoJSON in WGS84.
    """

    all_map_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # filtered_labels in einen für rasterio.features.shapes passenden Typ bringen
    label_array = filtered_labels.astype("int16")

    for label in np.unique(label_array):
        # -1 ist der ungültige Außenbereich, den überspringen wir
        if label == -1:
            continue

        # Binäre Maske für das aktuelle Label
        labelmask = (label_array == label).astype("int8")

        # Polygone aus der Maske extrahieren (in der combined-AEQD)
        polygons = [
            shape(geom)
            for geom, value in shapes(labelmask, transform=combined_transform)
            if value == 1
        ]

        if len(polygons) > 0:
            intermediate_gdf = gpd.GeoDataFrame(
                geometry=polygons, crs=combined_aeqd_crs
            )
            intermediate_gdf = intermediate_gdf.to_crs("EPSG:4326")
        else:
            intermediate_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        intermediate_gdf["label"] = int(label)
        all_map_gdf = pd.concat(
            [all_map_gdf, intermediate_gdf], ignore_index=True
        )

    # Defekte Geometrien reparieren und ungültige rauswerfen
    all_map_gdf.geometry = all_map_gdf.geometry.buffer(0)
    all_map_gdf = all_map_gdf[all_map_gdf.geometry.is_valid]

    # Pro Label vereinigen
    all_map_gdf = all_map_gdf.dissolve(by="label")

    # Nach Dissolve nochmal reparieren
    all_map_gdf.geometry = all_map_gdf.geometry.buffer(0)
    all_map_gdf = all_map_gdf[all_map_gdf.geometry.is_valid]

    # Verzeichnis sicherstellen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_map_gdf.to_file(output_path)

    print(f"GeoJSON gespeichert: {output_path}")
    return all_map_gdf


def create_combined_map(
    augmented_dems: list[AugmentedDEM],
    settings: dict,
    oversampling: float = 12.0,
) -> tuple[np.ndarray, Affine, CRS]:
    """Fügt alle DEMs zu einer gemeinsamen azimuthal-äquidistanten Karte zusammen.

    Ablauf:
      1. Gemeinsames AEQD-CRS über den Mittelpunkt aller DEM-Bounds bestimmen
      2. Valide Tile-Zentren + Labels aller DEMs einsammeln
      3. Zentren in die gemeinsame AEQD umrechnen
      4. Pixelraster aufspannen (Auflösung = tile_size_km / oversampling)
      5. Labels per Nearest-Neighbor-Interpolation auf das Raster legen
      6. Modus-Filter zur Glättung
      7. Bereiche ohne DEM-Abdeckung auf -1 setzen
      8. Eingefärbtes GeoTIFF speichern

    Args:
        augmented_dems: Liste aller verarbeiteten AugmentedDEM-Instanzen.
        settings: INTERNAL_SETTINGS-Dictionary.
        oversampling: Verhältnis Pixelgröße zu Tile-Größe (höher = feiner).

    Returns:
        (filtered_labels, combined_transform, combined_aeqd_crs)
    """

    # --- 1. Gemeinsame AEQD-Projektion über das Zentrum aller DEMs ---
    all_xmins = [d.geo_bounds.xmin for d in augmented_dems]
    all_xmaxs = [d.geo_bounds.xmax for d in augmented_dems]
    all_ymins = [d.geo_bounds.ymin for d in augmented_dems]
    all_ymaxs = [d.geo_bounds.ymax for d in augmented_dems]

    center_lon = (min(all_xmins) + max(all_xmaxs)) / 2
    center_lat = (min(all_ymins) + max(all_ymaxs)) / 2

    combined_aeqd_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs"
    )
    print(f"Gemeinsames AEQD-Zentrum: lat={center_lat:.3f}, lon={center_lon:.3f}")

    wgs84_to_combined_aeqd = Transformer.from_crs("EPSG:4326", combined_aeqd_crs, always_xy=True)

    # --- 2. Valide Tile-Zentren + Labels aller DEMs einsammeln ---
    all_centers_wgs84 = []
    all_labels = []

    for dem in augmented_dems:
        tile_y_aeqd = np.array([c[0] for c in dem.tile_centers_aeqd])
        tile_x_aeqd = np.array([c[1] for c in dem.tile_centers_aeqd])

        # Nur Tiles innerhalb der sicheren Bounding-Box berücksichtigen
        validity = contains_xy(dem.bbox_aeqd_safe, tile_x_aeqd, tile_y_aeqd)
        valid_indices = np.where(validity)[0]

        for idx in valid_indices:
            center = dem.tile_centers_orig[idx]
            all_centers_wgs84.append((center.x, center.y))
            all_labels.append(dem.labels[idx])

    all_centers_wgs84 = np.array(all_centers_wgs84)
    all_labels = np.array(all_labels)
    print(f"Anzahl gesammelter valider Tiles: {len(all_labels)}")

    # --- 3. Zentren in die gemeinsame AEQD umrechnen ---
    centers_aeqd_x, centers_aeqd_y = wgs84_to_combined_aeqd.transform(
        all_centers_wgs84[:, 0], all_centers_wgs84[:, 1]
    )

    # --- 4. Pixelraster aufspannen ---
    combined_xmin, combined_xmax = min(all_xmins), max(all_xmaxs)
    combined_ymin, combined_ymax = min(all_ymins), max(all_ymaxs)

    # Ecken und Kantenmittelpunkte transformieren, damit gekrümmte Ränder nicht abgeschnitten werden
    corner_lons = [combined_xmin, combined_xmax, combined_xmax, combined_xmin]
    corner_lats = [combined_ymin, combined_ymin, combined_ymax, combined_ymax]
    edge_lons = [(combined_xmin + combined_xmax) / 2, combined_xmax,
                 (combined_xmin + combined_xmax) / 2, combined_xmin]
    edge_lats = [combined_ymin, (combined_ymin + combined_ymax) / 2,
                 combined_ymax, (combined_ymin + combined_ymax) / 2]

    corners_x, corners_y = wgs84_to_combined_aeqd.transform(corner_lons, corner_lats)
    edges_x, edges_y = wgs84_to_combined_aeqd.transform(edge_lons, edge_lats)

    all_x_m = np.concatenate([corners_x, edges_x, centers_aeqd_x])
    all_y_m = np.concatenate([corners_y, edges_y, centers_aeqd_y])

    img_xmin_m, img_xmax_m = float(np.min(all_x_m)), float(np.max(all_x_m))
    img_ymin_m, img_ymax_m = float(np.min(all_y_m)), float(np.max(all_y_m))

    tile_size_m = settings["fft"]["tile_size_km"] * 1000
    pixel_size_m = tile_size_m / oversampling

    img_width_px = int(np.ceil((img_xmax_m - img_xmin_m) / pixel_size_m))
    img_height_px = int(np.ceil((img_ymax_m - img_ymin_m) / pixel_size_m))

    # Affiner Transform: Ursprung oben-links
    combined_transform = from_origin(img_xmin_m, img_ymax_m, pixel_size_m, pixel_size_m)
    print(f"Bildmaße: {img_width_px} x {img_height_px} Pixel (Auflösung: {pixel_size_m:.0f} m/Pixel)")

    # --- 5. Nearest-Neighbor-Interpolation ---
    grid_x_m = img_xmin_m + (np.arange(img_width_px) + 0.5) * pixel_size_m
    grid_y_m = img_ymax_m - (np.arange(img_height_px) + 0.5) * pixel_size_m
    grid_xx_m, grid_yy_m = np.meshgrid(grid_x_m, grid_y_m, indexing="xy")

    with common.SimpleTimer("Nearest-Interpolation auf das gemeinsame Raster"):
        interpolated_labels = griddata(
            points=np.column_stack([centers_aeqd_x, centers_aeqd_y]),
            values=all_labels,
            xi=(grid_xx_m, grid_yy_m),
            method="nearest",
        )

    # --- 6. Modus-Filter ---
    mode_filter_radius_km = settings["output"]["label_mode_filter_radius_km"]
    mode_filter_radius_px = int(round(mode_filter_radius_km * 1000 / pixel_size_m))
    print(f"Modus-Filter-Radius: {mode_filter_radius_px} Pixel (= {mode_filter_radius_km} km)")

    with common.SimpleTimer("Modus-Filter auf der gemeinsamen Karte"):
        filtered_labels = common.mode_filter(interpolated_labels, radius=mode_filter_radius_px)

    # --- 7. Außenbereich auf -1 setzen ---
    combined_validity_polygons = []
    for dem in augmented_dems:
        aeqd_to_wgs84 = Transformer.from_crs(dem.aeqd_crs, "EPSG:4326", always_xy=True)
        bbox_wgs84 = shapely_transform_v2(aeqd_to_wgs84.transform, dem.bbox_aeqd_safe)
        bbox_combined = shapely_transform_v2(wgs84_to_combined_aeqd.transform, bbox_wgs84)
        combined_validity_polygons.append(bbox_combined)

    combined_validity_union = unary_union(combined_validity_polygons)
    combined_validity_mask = rio_geometry_mask(
        [combined_validity_union],
        out_shape=(img_height_px, img_width_px),
        transform=combined_transform,
        invert=True,  # True = innerhalb der Union (gültig)
    )

    filtered_labels = filtered_labels.astype(np.int16)
    filtered_labels[~combined_validity_mask] = -1

    # --- 8. Einfärben und als GeoTIFF speichern ---
    # Label -1 (Außenbereich) wird schwarz, daher Farbpalette um Schwarz vorne ergänzen
    shifted_labels = filtered_labels + 1
    color_palette = ["#000000"] + list(custom_colors_medium)
    color_array = common.color_to_numpy(color_palette)

    max_label = int(shifted_labels.max())
    if max_label >= len(color_array):
        raise ValueError(
            f"Es gibt mehr Labels ({max_label + 1}) als Farben ({len(color_array)}). "
            "Bitte label_count in den Settings oder die Farbpalette anpassen."
        )

    output_image_rgb = color_array[shifted_labels]
    combined_image_rgb = np.transpose(output_image_rgb, (2, 0, 1)).astype("uint8")

    filename = (
        f"combined_map  "
        f"tlszkm {settings['fft']['tile_size_km']:.1f}  "
        f"tlszpx {settings['fft']['tile_size_px']:.0f}  "
        f"fftlvls {settings['fft']['fft_levels']:.0f}  "
        f"ovrlp-pct {settings['fft']['tile_overlap_percent']:.0f}  "
        f"fltrkm {settings['output']['label_mode_filter_radius_km']:.1f}  "
        f"lblct {settings['output']['label_count']:.0f}  "
        f"oversampling {oversampling:.2f}"
        f".tif"
    )
    filepath = os.path.join(settings["output"]["folder_name"], filename)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=img_height_px,
        width=img_width_px,
        count=3,
        dtype="uint8",
        crs=combined_aeqd_crs,
        transform=combined_transform,
    ) as dst:
        dst.write(combined_image_rgb[0], 1)
        dst.write(combined_image_rgb[1], 2)
        dst.write(combined_image_rgb[2], 3)

    print(f"Combined Map gespeichert: {filepath}")
    return filtered_labels, combined_transform, combined_aeqd_crs
