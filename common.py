# Project related
from pyproj import Geod, CRS, Transformer
import time
import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances, haversine_distances
import random
import re
from skimage.filters.rank import modal
from skimage.morphology import disk


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



class SimpleTimer:
    '''This class is just a timer for checking the performance'''



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
    '''
    This class wraps processes to help identifying what is done by printing info about them.
    '''

    def __init__(self, process_description: str, current_index: int = 1, total_count: int = 1, single_description: str | None = None):
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


# Helper functions

def closest_to_median_indices(haystack):
    """This finds the points’ ids closest to the median of a set"""
    medianpoint = np.median(haystack, axis=0)
    distsances_squared = np.sum((haystack - medianpoint) ** 2, axis=1)

    closest_points_ids = np.argsort(distsances_squared)
    return closest_points_ids


def sort_by_furthest(haystack):
    distances = haversine_distances(np.radians(np.flip(haystack, axis=1)))
    np.fill_diagonal(distances, np.inf)
    closest_distances = np.min(distances, axis=0)
    return np.argsort(closest_distances)[::-1]


# Find the medoid IDs of an array and return them in ascending order
def medoid_indices(haystack):
    distances = pairwise_distances(haystack)
    distances_to_all_others = distances.sum(axis=1)

    medoid_ranking = np.argsort(distances_to_all_others)
    return medoid_ranking


# A simple limit function for integers
def clamp(x, min, max) -> int:
    return int(sorted([min, x, max])[1])


def positions_to_kml(positions_per_label, output_path, lon_first=True):
    """
    positions_per_label: dict label -> list of np.array([x, y]) in WGS84
    lon_first: True falls position = [lon, lat], False falls [lat, lon]
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
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

        lines.extend([
            f'  <StyleMap id="m_label_{label}">',
            f'    <Pair><key>normal</key><styleUrl>#s_label_{label}</styleUrl></Pair>',
            f'    <Pair><key>highlight</key><styleUrl>#s_label_{label}_hl</styleUrl></Pair>',
            '  </StyleMap>',
            f'  <Style id="s_label_{label}">',
            '    <IconStyle>',
            '      <scale>5</scale>',
            f'      <color>{color}</color>',
            '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/donut.png</href></Icon>',
            '    </IconStyle>',
            '    <LabelStyle><color>fffcffff</color></LabelStyle>',
            '  </Style>',
            f'  <Style id="s_label_{label}_hl">',
            '    <IconStyle>',
            '      <scale>5.90909</scale>',
            f'      <color>{color}</color>',
            '      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/donut.png</href></Icon>',
            '    </IconStyle>',
            '    <LabelStyle><color>fffcffff</color></LabelStyle>',
            '  </Style>',
        ])

    lines.extend([
        '  <Folder>',
        '    <name>Representative Tiles</name>',
        '    <open>1</open>',
    ])

    for label, positions in positions_per_label.items():
        for idx, position in enumerate(positions):
            if lon_first:
                lon, lat = float(position[0]), float(position[1])
            else:
                lat, lon = float(position[0]), float(position[1])

            lines.extend([
                '    <Placemark>',
                f'      <name>{label}_{idx}</name>',
                f'      <styleUrl>#m_label_{label}</styleUrl>',
                '      <LookAt>',
                f'        <longitude>{lon}</longitude>',
                f'        <latitude>{lat}</latitude>',
                '        <range>5000</range>',
                '        <tilt>62</tilt>',
                '        <heading>0</heading>',
                '      </LookAt>',
                '      <Point>',
                f'        <coordinates>{lon},{lat},0</coordinates>',
                '      </Point>',
                '    </Placemark>',
            ])

    lines.extend([
        '  </Folder>',
        '</Document>',
        '</kml>',
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def mode_filter(label_map, radius=10) -> np.ndarray:
    """Mode-Filter cleans the labels to make them more smooth"""
    # modal braucht uint8 oder uint16 – offset für -1
    offset = 1
    shifted = (label_map + offset).astype(np.uint16)
    
    result = modal(shifted, disk(radius))
    
    return result.astype(np.int16) - offset



def constrain_labels(input) -> list:
    """
    Takes a list of integer labels in any range and converts them to 1, 2, 3…
    """

    unique_labels = np.sort(np.unique(input))
    new_labels = np.arange(0, len(unique_labels), 1)
    from_to = dict(zip(unique_labels, new_labels))

    constrained_labels = np.vectorize(from_to.get)(input)

    return constrained_labels


def color_to_numpy(stringlist) -> np.ndarray:
    """
    This converts a list of string hex color codes to actual rgb values in a numpy array.
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
    Calculates the pythagorean distance between two points.
    """

    return math.sqrt((y_a - y_b) ** 2 + (x_a - x_b) ** 2)


def threshold(input, threshold, bandwidth=1) -> float:
    """
    Creates smooth edges in the circle masks.
    Values below threshold become 0, values above 1.
    Values at the threshold ± half bandwidth are faded.
    """

    # This is essentially anti aliasing without subsampling.

    result = np.interp(
        input, [threshold - (bandwidth / 2), threshold + (bandwidth / 2)], [0, 1]
    )
    return result.astype(np.float32)


def sort_labels(labels):
    """Rearrange label ids by their group size"""
    unique, counts = np.unique(labels, return_counts=True)
    size_order = np.argsort(-counts)

    lookup = np.empty_like(unique)
    lookup[size_order] = np.arange(len(size_order))

    relabeled = lookup[labels]
    return relabeled


def CircleImage(height, width, radius, inverted=False, bandwidth=1) -> np.ndarray:
    """
    Returns an antialiased image of a circle with a given radius as a numpy array for masking.
    """

    # Height, width defines the shape of the 'image'
    # Inverted flips colors
    # The bandwidth defines the width of a smooth edge of the circle (for anti aliasing)

    circle_image = np.zeros((height, width), dtype=np.float32)

    if radius == 0:
        return (1 - circle_image) if inverted else (circle_image)
    else:
        for x in range(width):
            for y in range(height):
                circle_image[y, x] = euclidean_distance(
                    y + (0.5 if height % 2 == 1 else 0),
                    height / 2,
                    x + (0.5 if width % 2 == 1 else 0),
                    width / 2,
                )

    if inverted:
        return 1 - threshold(circle_image, radius, 1)
    else:
        return threshold(circle_image, radius, 1)


def RingImage(height, width, inner_radius, outer_radius, bandwidth) -> np.ndarray:
    """
    Utilizes two circles to create a ring mask with smooth edges.
    Please look into CircleImage for in-depth definition.
    """

    if outer_radius < inner_radius:
        raise ValueError("The inner radius must be smaller than the outer radius.")

    # This combines two circles to form a ring mask
    outercircle = CircleImage(
        height, width, outer_radius, inverted=True, bandwidth=bandwidth
    )
    innercircle = CircleImage(
        height, width, inner_radius, inverted=True, bandwidth=bandwidth
    )

    ringimg = outercircle - innercircle

    return (ringimg - ringimg.min()) / (ringimg.max() - ringimg.min())


def RingImageSeries(height, width, steps, bandwidth) -> np.ndarray:
    """
    This creates a 3D numpy array of the shape
    (Masks, individual Height, individual Width)
    This is used to sum and average the FFT magnitudes
    """

    # The diameter gets bigger logarithmically, starting with a diameter of 1
    smallest_side = min(height, width)

    outer_radii = np.logspace(
        0, np.log2(smallest_side / 2), steps, base=2
    )  # 0 means it starts with 1 (log)
    inner_radii = np.append(0, outer_radii[:-1])

    all_masks = np.zeros((steps, height, width), dtype=np.float32)

    for i in range(steps):
        all_masks[i] = RingImage(
            height, width, inner_radii[i], outer_radii[i], bandwidth=bandwidth
        )

    return all_masks


def replace_umlaute(input):
    input = re.sub(r"ä","ae",input)
    input = re.sub(r"ö","oe",input)
    input = re.sub(r"ü","ue",input)
    input = re.sub(r"ß","ss",input)
    return input