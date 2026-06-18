# Projektbezogen
from pyproj import Geod, CRS, Transformer
from rasterio.warp import transform_bounds
import time
import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances, haversine_distances
import random
import re
from skimage.filters.rank import modal
from skimage.morphology import disk

geod = Geod(ellps="WGS84")


class GeographicCoordinate:
    """Stores coordinates and their respective coordinate system."""

    def __init__(self, x, y, crs):
        self.x = x
        self.y = y
        self.crs = crs

    def in_another_crs(self, dst_crs):
        """Reproject the same coordinates into another coordinate system"""

        intermediary_transformer = Transformer.from_crs(
            self.crs, dst_crs, always_xy=True
        )
        projected_bounds = intermediary_transformer.transform(self.x, self.y)

        return GeographicCoordinate(*projected_bounds, dst_crs)

    def __str__(self):
        return f"X: {self.x:.1f}, Y: {self.y:.1f}"

    def to_numpy(self):
        return np.array((self.x, self.y))


class GeographicBounds:
    """Speichert geografische oder projizierte Begrenzungsrahmen (West/Süd/Ost/Nord)
    inklusive Koordinatensystem und optionalem Affin-Transform."""

    def __init__(self, xmin, ymin, xmax, ymax, crs, transform=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.crs = crs
        self.transform = transform

    def in_another_crs(self, dst_crs):
        """Projiziert die Begrenzungsrahmen in ein anderes Koordinatensystem."""
        # densify_pts verhindert, dass Wölbungen bei der Reprojektion abgeschnitten werden
        projected_bounds = transform_bounds(self.crs, dst_crs, *self.bounds, densify_pts=10)
        return GeographicBounds(*projected_bounds, dst_crs)

    @property
    def bounds(self):
        """Gibt die Grenzen als Tupel (xmin, ymin, xmax, ymax) zurück."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def center_x(self):
        """Mittelpunkt entlang der X-Achse (Einheit je nach CRS)."""
        return (self.xmin + self.xmax) / 2

    @property
    def center_y(self):
        """Mittelpunkt entlang der Y-Achse (Einheit je nach CRS)."""
        return (self.ymin + self.ymax) / 2

    @property
    def intermediary_aeqd_crs(self):
        """Azimuthal-äquidistantes CRS zentriert auf den Mittelpunkt dieser Bounds.
        Wird als Zwischenprojektion für die DEM-Kachelung verwendet."""
        geo = self.in_another_crs(CRS("EPSG:4326"))
        return CRS.from_proj4(
            f"+proj=aeqd +lat_0={geo.center_y} +lon_0={geo.center_x} "
            "x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"
        )

    def get_bound_length(self, side: str) -> float:
        """Gibt die Länge einer Seite des Begrenzungsrahmens in Metern zurück.

        Args:
            side: 'left', 'top', 'right' oder 'bottom'
        """
        if self.crs.is_geographic:
            coords = {
                "left":   (self.xmin, self.ymin, self.xmin, self.ymax),
                "top":    (self.xmin, self.ymax, self.xmax, self.ymax),
                "right":  (self.xmax, self.ymin, self.xmax, self.ymax),
                "bottom": (self.xmin, self.ymin, self.xmax, self.ymin),
            }
            if side not in coords:
                raise ValueError(f"Ungültige Seite: '{side}'. Erlaubt: left, top, right, bottom.")
            _, _, distance = geod.inv(*coords[side])
        elif self.crs.is_projected:
            if side in ("left", "right"):
                distance = abs(self.ymax - self.ymin)
            elif side in ("top", "bottom"):
                distance = abs(self.xmax - self.xmin)
            else:
                raise ValueError(f"Ungültige Seite: '{side}'. Erlaubt: left, top, right, bottom.")
        else:
            raise ValueError("CRS ist weder geografisch noch projiziert.")
        return distance

    def get_projected_extent(self):
        """Gibt Breite und Höhe in Metern als (x, y)-Tupel zurück.

        Bei geografischen Bounds wird zunächst in die AEQD-Projektion umgerechnet.
        """
        if self.crs.is_geographic:
            # Rekursiv: erst in AEQD projizieren, dann messen
            return self.in_another_crs(self.intermediary_aeqd_crs).get_projected_extent()
        elif self.crs.is_projected:
            return (abs(self.xmax - self.xmin), abs(self.ymax - self.ymin))
        else:
            raise ValueError("CRS scheint ungültig. Weder geografisch noch projiziert.")

    def as_list(self):
        """Gibt die Grenzen als Liste [xmin, ymin, xmax, ymax] zurück."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def __str__(self):
        lines = [
            "GeographicBounds (gerundet):",
            f"  X Min: {self.xmin:>10.2f}",
            f"  Y Min: {self.ymin:>10.2f}",
            f"  X Max: {self.xmax:>10.2f}",
            f"  Y Max: {self.ymax:>10.2f}",
            f"  Links: {self.get_bound_length('left'):.2f} m",
            f"  Unten: {self.get_bound_length('bottom'):.2f} m",
            f"  Zentrum: {self.center_y:.3f}°N, {self.center_x:.3f}°E",
        ]
        return "\n".join(lines)


class SimpleTimer:
    """Diese Klasse ist ein einfacher Timer zur Leistungsmessung."""

    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.timer = time.perf_counter()
        print(f"Starting: {self.description}…")
        return self

    def __exit__(self, type, value, traceback):
        self.time_needed = time.perf_counter() - self.timer
        print(f"{self.description} took {self.time_needed:.1f} Seconds")


class VerboseInfoTimer:
    """
    Diese Klasse kapselt Prozesse und gibt Informationen aus, um nachzuvollziehen, was gerade ausgeführt wird.
    """

    def __init__(
        self,
        process_description: str,
        current_index: int = 1,
        total_count: int = 1,
        single_description: str | None = None,
    ):
        self.sd = single_description
        self.pd = process_description
        self.curr_it = current_index + 1
        self.total_count = total_count
        self.rest = total_count - self.curr_it

    def __enter__(self):
        self.timer = time.perf_counter()

        if self.total_count > 1:
            if self.curr_it == 1:
                print(f"\n{self.pd} starting…\n")
            print(f"{self.sd} {self.curr_it} of {self.total_count}…")
        else:
            print(f"{self.pd} starting…")

    def __exit__(self, type, value, traceback):
        self.time_needed = time.perf_counter() - self.timer

        if self.total_count > 1:
            print(f"{self.sd} completed. This one took {self.time_needed:.1f} seconds.")

            if self.rest > 0:
                print(f"{self.rest} more to go.")
            else:
                print(f"{self.pd} ({self.total_count}) completed.")
        else:
            print(f"{self.pd} completed. This group {self.time_needed:.1f} seconds.")

        print("")


# Hilfsfunktionen


def closest_to_median_indices(haystack):
    """Findet die IDs der Punkte, die dem Median einer Menge am nächsten liegen."""
    medianpoint = np.median(haystack, axis=0)
    distsances_squared = np.sum((haystack - medianpoint) ** 2, axis=1)

    closest_points_ids = np.argsort(distsances_squared)
    return closest_points_ids


def sort_by_furthest(haystack):
    distances = haversine_distances(np.radians(np.flip(haystack, axis=1)))
    np.fill_diagonal(distances, np.inf)
    closest_distances = np.min(distances, axis=0)
    return np.argsort(closest_distances)[::-1]


# Findet die Medoid-IDs eines Arrays und gibt sie in aufsteigender Reihenfolge zurück
def medoid_indices(haystack):
    distances = pairwise_distances(haystack)
    distances_to_all_others = distances.sum(axis=1)

    medoid_ranking = np.argsort(distances_to_all_others)
    return medoid_ranking


# Eine einfache Begrenzungsfunktion für ganze Zahlen
def clamp(x, min, max) -> int:
    return int(sorted([min, x, max])[1])


def positions_to_kml(positions_per_label, output_path, lon_first=True):
    """
    positions_per_label: dict label -> Liste von np.array([x, y]) in WGS84
    lon_first: True falls Position = [lon, lat], False falls [lat, lon]
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
    ]

    # Für jedes Label einen Style mit Zufallsfarbe
    label_colors = {}
    for label in positions_per_label.keys():
        # KML Farben: aabbggrr (Alpha, Blue, Green, Red)
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        color = f"ff{b:02x}{g:02x}{r:02x}"
        label_colors[label] = color

        lines.extend(
            [
                f'  <StyleMap id="m_label_{label}">',
                f"    <Pair><key>normal</key><styleUrl>#s_label_{label}</styleUrl></Pair>",
                f"    <Pair><key>highlight</key><styleUrl>#s_label_{label}_hl</styleUrl></Pair>",
                "  </StyleMap>",
                f'  <Style id="s_label_{label}">',
                "    <IconStyle>",
                "      <scale>5</scale>",
                f"      <color>{color}</color>",
                "      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/donut.png</href></Icon>",
                "    </IconStyle>",
                "    <LabelStyle><color>fffcffff</color></LabelStyle>",
                "  </Style>",
                f'  <Style id="s_label_{label}_hl">',
                "    <IconStyle>",
                "      <scale>5.90909</scale>",
                f"      <color>{color}</color>",
                "      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/donut.png</href></Icon>",
                "    </IconStyle>",
                "    <LabelStyle><color>fffcffff</color></LabelStyle>",
                "  </Style>",
            ]
        )

    lines.extend(
        [
            "  <Folder>",
            "    <name>Representative Tiles</name>",
            "    <open>1</open>",
        ]
    )

    for label, positions in positions_per_label.items():
        for idx, position in enumerate(positions):
            if lon_first:
                lon, lat = float(position[0]), float(position[1])
            else:
                lat, lon = float(position[0]), float(position[1])

            lines.extend(
                [
                    "    <Placemark>",
                    f"      <name>{label}_{idx}</name>",
                    f"      <styleUrl>#m_label_{label}</styleUrl>",
                    "      <LookAt>",
                    f"        <longitude>{lon}</longitude>",
                    f"        <latitude>{lat}</latitude>",
                    "        <range>5000</range>",
                    "        <tilt>62</tilt>",
                    "        <heading>0</heading>",
                    "      </LookAt>",
                    "      <Point>",
                    f"        <coordinates>{lon},{lat},0</coordinates>",
                    "      </Point>",
                    "    </Placemark>",
                ]
            )

    lines.extend(
        [
            "  </Folder>",
            "</Document>",
            "</kml>",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def mode_filter(label_map, radius=10) -> np.ndarray:
    """Modus-Filter glättet die Labels."""
    # modal braucht uint8 oder uint16 – offset für -1
    offset = 1
    shifted = (label_map + offset).astype(np.uint16)

    result = modal(shifted, disk(radius))

    return result.astype(np.int16) - offset


def constrain_labels(input) -> list:
    """
    Nimmt eine Liste von ganzzahligen Labels in einem beliebigen Bereich und konvertiert sie zu 1, 2, 3…
    """

    unique_labels = np.sort(np.unique(input))
    new_labels = np.arange(0, len(unique_labels), 1)
    from_to = dict(zip(unique_labels, new_labels))

    constrained_labels = np.vectorize(from_to.get)(input)

    return constrained_labels


def color_to_numpy(stringlist) -> np.ndarray:
    """
    Konvertiert eine Liste von Hex-Farbcodes als Zeichenketten in RGB-Werte in einem NumPy-Array.
    """

    colorlist = np.empty((len(stringlist), 3))

    for i, string in enumerate(stringlist):
        removed_hashtag = re.search(r"(?i)([a-f0-9]+)", string).group(0)
        colorlist[i] = np.array(
            (
                int(removed_hashtag[0:2], 16),
                int(removed_hashtag[2:4], 16),
                int(removed_hashtag[4:6], 16),
            )
        )

    return colorlist


def euclidean_distance(y_a, y_b, x_a, x_b) -> float:
    """
    Berechnet den euklidischen Abstand zwischen zwei Punkten.
    """

    return math.sqrt((y_a - y_b) ** 2 + (x_a - x_b) ** 2)


def threshold(input, threshold, bandwidth=1) -> float:
    """
    Erzeugt weiche Kanten in den Kreismasken.
    Werte unterhalb des Schwellwerts werden 0, Werte oberhalb 1.
    Werte im Bereich Schwellwert ± halbe Bandbreite werden weich übergeblendet.
    """

    # Im Wesentlichen Antialiasing ohne Unterabtastung.

    result = np.interp(
        input, [threshold - (bandwidth / 2), threshold + (bandwidth / 2)], [0, 1]
    )
    return result.astype(np.float32)


def sort_labels(labels):
    """Label-IDs nach Gruppengröße neu anordnen."""
    unique, counts = np.unique(labels, return_counts=True)
    size_order = np.argsort(-counts)

    lookup = np.empty_like(unique)
    lookup[size_order] = np.arange(len(size_order))

    relabeled = lookup[labels]
    return relabeled


def CircleImage(height, width, radius, inverted=False, bandwidth=1) -> np.ndarray:
    """
    Gibt ein kantengeglättetes Bild eines Kreises mit gegebenem Radius als NumPy-Array zurück (für Masken).
    """

    if radius == 0:
        circle_image = np.zeros((height, width), dtype=np.float32)
        return (1 - circle_image) if inverted else circle_image

    # Koordinaten relativ zum Mittelpunkt, mit 0.5-Versatz bei ungerader Größe
    ys = np.arange(height, dtype=np.float32) + (0.5 if height % 2 else 0.0) - height / 2
    xs = np.arange(width, dtype=np.float32) + (0.5 if width % 2 else 0.0) - width / 2

    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    circle_image = np.linalg.norm(np.stack([yy, xx], axis=-1), axis=-1)

    result = threshold(circle_image, radius, bandwidth=bandwidth)
    return 1 - result if inverted else result


def RingImage(height, width, inner_radius, outer_radius, bandwidth) -> np.ndarray:
    """
    Verwendet zwei Kreise, um eine Ringmaske mit weichen Kanten zu erstellen.
    Für eine ausführliche Beschreibung siehe CircleImage.
    """

    if outer_radius < inner_radius:
        raise ValueError("The inner radius must be smaller than the outer radius.")

    # Kombiniert zwei Kreise zu einer Ringmaske
    outercircle = CircleImage(
        height, width, outer_radius, inverted=True, bandwidth=bandwidth
    )
    innercircle = CircleImage(
        height, width, inner_radius, inverted=True, bandwidth=bandwidth
    )

    ringimg = outercircle - innercircle

    return (ringimg - ringimg.min()) / (ringimg.max() - ringimg.min())


def RingImageSeries(height, width, steps, bandwidth, ref_size=23) -> np.ndarray:
    """
    Erstellt ein 3D-NumPy-Array der Form
    (Masken, einzelne Höhe, einzelne Breite).
    Wird verwendet, um FFT-Magnituden zu summieren und zu mitteln.
    """

    smallest_side = min(height, width)

    # Verteilung immer relativ zu ref_size berechnen, dann skalieren
    outer_radii = np.logspace(0, np.log2(ref_size / 2), steps, base=2) * (
        smallest_side / ref_size
    )

    inner_radii = np.append(0, outer_radii[:-1])

    all_masks = np.zeros((steps, height, width), dtype=np.float32)

    for i in range(steps):
        all_masks[i] = RingImage(
            height, width, inner_radii[i], outer_radii[i], bandwidth=bandwidth
        )

    return all_masks


def RingImageSeriesLog(height, width, steps, bandwidth) -> np.ndarray:
    """
    Erstellt ein 3D-NumPy-Array der Form
    (Masken, einzelne Höhe, einzelne Breite).
    Wird verwendet, um FFT-Magnituden zu summieren und zu mitteln.
    Dies ist die zweite Version, die die Radien logaritmisch abstuft
    """

    max_radius = (height + width ) / 4

    radii = np.logspace(0, np.log10(max_radius), steps + 1)

    all_masks = np.zeros((steps, height, width), dtype=np.float32)

    for i in range(steps):
        all_masks[i] = RingImage(
            height, width, radii[i], radii[i + 1], bandwidth=bandwidth
        )

    return all_masks



def diameter_series(end, steps, finesteps=None):
    # "steps" Schritte inkl. 1 und "end"

    if finesteps is None: finesteps = steps

    factor = 10 **( (np.log10(end)) / (steps-1))

    series = np.logspace(0, np.log10(end), finesteps)

    return (series, factor)


def RingImageSeriesLogFine(
    height, width, steps, animation_steps=100, bandwidth=1
) -> np.ndarray:
    """
    Erstellt ein 3D-NumPy-Array der Form
    (Masken, einzelne Höhe, einzelne Breite).
    Wird verwendet, um FFT-Magnituden zu summieren und zu mitteln.
    Dies ist die zweite Version, die die Radien logaritmisch abstuft
    """

    avg_radius = (width + height) / 4

    half_diagonal = np.linalg.norm((height, width), 2) / 2  # Halbdiagonale

    factor = 10 ** (np.log10(half_diagonal) / (steps - 1))

    outer_max = avg_radius * np.sqrt(factor)
    outer_min = factor**2

    inner_max = avg_radius / np.sqrt(factor)
    inner_min = factor

    outer_radii = np.logspace(np.log10(outer_max), np.log10(outer_min), animation_steps)

    inner_radii = np.logspace(np.log10(inner_max), np.log10(inner_min), animation_steps)

    all_masks = np.zeros((animation_steps, height, width), dtype=np.float32)

    for i in range(animation_steps):
        all_masks[i] = RingImage(
            height, width, inner_radii[i], outer_radii[i], bandwidth=bandwidth
        )

    return all_masks






def replace_umlaute(input):
    input = re.sub(r"ä", "ae", input)
    input = re.sub(r"ö", "oe", input)
    input = re.sub(r"ü", "ue", input)
    input = re.sub(r"ß", "ss", input)
    return input
