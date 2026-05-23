"""
DEM (16-bit Graustufen-TIF) -> 8-bit Graustufen-TIF mit echten Höhenlinien.

Echte Konturen via Marching Squares (skimage.measure.find_contours),
keine Schwellwert-Färbung. Plateaus erzeugen daher Linien an ihren Rändern,
keine flächigen Kleckse.

Optional: Index-Linien durchgezogen, Zwischenlinien gestrichelt.
Strichlinien-Breite kann per Verhältnis (Float) gesetzt werden;
sub-pixel Breiten werden per Supersampling gerendert.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage import measure
from skimage.transform import resize
import cv2


SUPERSAMPLE = 4  # Faktor für sub-pixel Strichbreiten


def _draw_dashed_polyline(
    canvas: np.ndarray,
    pts: np.ndarray,
    color: int,
    thickness: int,
    line_type: int,
    dash_len: float,
    gap_len: float,
) -> None:
    """Zeichnet eine Polylinie als Strichlinie. Die Dash-Phase läuft über
    Segmentgrenzen hinweg weiter, damit Strich- und Lückenlängen entlang
    der Kurve konstant bleiben."""
    if len(pts) < 2:
        return
    period = dash_len + gap_len
    phase = 0.0

    for i in range(len(pts) - 1):
        x1, y1 = float(pts[i, 0]), float(pts[i, 1])
        x2, y2 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        dx, dy = x2 - x1, y2 - y1
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len == 0:
            continue
        ux, uy = dx / seg_len, dy / seg_len

        traveled = 0.0
        while traveled < seg_len:
            if phase < dash_len:
                remaining_on = dash_len - phase
                step = min(remaining_on, seg_len - traveled)
                sx = x1 + ux * traveled
                sy = y1 + uy * traveled
                ex = x1 + ux * (traveled + step)
                ey = y1 + uy * (traveled + step)
                cv2.line(
                    canvas,
                    (int(round(sx)), int(round(sy))),
                    (int(round(ex)), int(round(ey))),
                    color,
                    thickness,
                    line_type,
                )
                traveled += step
                phase += step
            else:
                remaining_off = period - phase
                step = min(remaining_off, seg_len - traveled)
                traveled += step
                phase += step
                if phase >= period:
                    phase = 0.0


def make_contour_image(
    dem_path: Path,
    out_path: Path,
    n_lines: int = 20,
    offset_frac: float = 0.02,
    line_width: int = 1,
    scale: float = 1.0,
    bg: int = 255,
    fg: int = 0,
    antialias: bool = True,
    dashed: bool = False,
    major_every: int = 10,
    dash_len: float = 4.0,
    gap_len: float = 10.0,
    dash_width: float = 1.0,
) -> None:
    dem = tifffile.imread(str(dem_path))
    if dem.ndim != 2:
        raise ValueError(f"Erwarte 2D-Bild, bekommen: shape {dem.shape}")
    dem = dem.astype(np.float64)

    if scale != 1.0:
        new_shape = (
            int(round(dem.shape[0] * scale)),
            int(round(dem.shape[1] * scale)),
        )
        dem = resize(
            dem, new_shape, order=1, preserve_range=True, anti_aliasing=True
        )

    h, w = dem.shape
    vmin, vmax = float(dem.min()), float(dem.max())
    span = vmax - vmin
    if span == 0:
        raise ValueError("DEM ist flach — keine Höhenlinien möglich.")

    lo = vmin + offset_frac * span
    hi = vmax - offset_frac * span
    if n_lines < 1:
        raise ValueError("n_lines muss >= 1 sein.")
    levels = (
        np.linspace(lo, hi, n_lines) if n_lines > 1 else np.array([(lo + hi) / 2])
    )

    canvas = np.full((h, w), bg, dtype=np.uint8)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    color = int(fg)
    thickness = int(line_width)

    # Effektive Strichlinien-Breite in Pixeln
    dash_thickness_px = float(line_width) * float(dash_width)
    # Supersampling nötig, wenn die Breite < 1 oder nicht ganzzahlig ist
    need_sup = dashed and (
        dash_thickness_px < 1.0
        or abs(dash_thickness_px - round(dash_thickness_px)) > 1e-6
    )

    if need_sup:
        sup_h, sup_w = h * SUPERSAMPLE, w * SUPERSAMPLE
        sup_canvas = np.full((sup_h, sup_w), bg, dtype=np.uint8)
        sup_thickness = max(1, int(round(dash_thickness_px * SUPERSAMPLE)))
    else:
        sup_canvas = None
        sup_thickness = 0
        dash_thickness_int = max(1, int(round(dash_thickness_px))) if dashed else 0

    for idx, level in enumerate(levels):
        is_major = (idx % major_every == 0) if dashed else True
        contours = measure.find_contours(dem, level)
        for contour in contours:
            pts_float = contour[:, [1, 0]]  # (x, y) als Float
            pts_int = np.round(pts_float).astype(np.int32)
            if len(pts_int) < 2:
                continue
            if is_major:
                cv2.polylines(
                    canvas, [pts_int], isClosed=False,
                    color=color, thickness=thickness, lineType=line_type,
                )
            elif need_sup:
                sup_pts = np.round(pts_float * SUPERSAMPLE).astype(np.int32)
                _draw_dashed_polyline(
                    sup_canvas, sup_pts, color, sup_thickness, cv2.LINE_AA,
                    dash_len * SUPERSAMPLE, gap_len * SUPERSAMPLE,
                )
            else:
                _draw_dashed_polyline(
                    canvas, pts_int, color, dash_thickness_int, line_type,
                    dash_len, gap_len,
                )

    # Supersampling-Pass herunterskalieren und einblenden
    if sup_canvas is not None:
        downsampled = cv2.resize(
            sup_canvas, (w, h), interpolation=cv2.INTER_AREA
        )
        # Linien sind dunkler oder heller als der Hintergrund -- richtige
        # Kombination wählen, damit die Linien sichtbar bleiben.
        if fg < bg:
            canvas = np.minimum(canvas, downsampled)
        else:
            canvas = np.maximum(canvas, downsampled)

    tifffile.imwrite(str(out_path), canvas)


def main() -> None:
    p = argparse.ArgumentParser(description="DEM -> Höhenlinien-TIF")
    p.add_argument("input", type=Path, help="16-bit Graustufen-DEM (TIF)")
    p.add_argument("output", type=Path, help="Ziel-TIF (8-bit Graustufen)")
    p.add_argument("-n", "--lines", type=int, default=20,
                   help="Anzahl Höhenlinien (Default 20)")
    p.add_argument("-o", "--offset", type=float, default=0.02,
                   help="Offset als Anteil der Wertespanne, 0..0.5 (Default 0.02)")
    p.add_argument("-w", "--width", type=int, default=1,
                   help="Linienbreite in Pixeln (Default 1)")
    p.add_argument("-s", "--scale", type=float, default=1.0,
                   help="Skalierungsfaktor für die Ausgabeauflösung (Default 1.0)")
    p.add_argument("--bg", type=int, default=255, help="Hintergrund 0..255")
    p.add_argument("--fg", type=int, default=0, help="Linienfarbe 0..255")
    p.add_argument("--no-aa", action="store_true", help="Antialiasing aus")

    p.add_argument("--dashed", action="store_true",
                   help="Index-Linien durchgezogen, Zwischenlinien gestrichelt")
    p.add_argument("--major-every", type=int, default=10,
                   help="Jede N-te Linie ist Index-Linie (Default 10)")
    p.add_argument("--dash-len", type=float, default=4.0,
                   help="Strichlänge in Pixeln (Default 4)")
    p.add_argument("--gap-len", type=float, default=10.0,
                   help="Lücke in Pixeln (Default 10)")
    p.add_argument("--dash-width", type=float, default=1.0,
                   help="Strichlinien-Breite als Verhältnis zu --width. "
                        "z.B. 0.75 bei -w 1 ergibt 0.75 px (Default 1.0)")

    args = p.parse_args()

    make_contour_image(
        dem_path=args.input,
        out_path=args.output,
        n_lines=args.lines,
        offset_frac=args.offset,
        line_width=args.width,
        scale=args.scale,
        bg=args.bg,
        fg=args.fg,
        antialias=not args.no_aa,
        dashed=args.dashed,
        major_every=args.major_every,
        dash_len=args.dash_len,
        gap_len=args.gap_len,
        dash_width=args.dash_width,
    )


if __name__ == "__main__":
    main()

# python3 dem_contours.py ./documentation/Representative_Samples/16_bit_heightmaps/schottland_ben_nevis_aeqd_-5.0035_56.7970.tif ./documentation/Representative_Samples/16_bit_heightmaps/contours_schottland_ben_nevis.tif -n 30 -o 0 --width 2 -s 13 --dashed --major-every 5 --dash-len 70 --gap-len 30  --dash-width 0.5

