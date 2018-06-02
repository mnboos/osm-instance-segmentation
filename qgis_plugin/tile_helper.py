from __future__ import division
from builtins import str
from .log_helper import warn, debug, info
from .qgis_2to3 import *

"""
 * Top left: (lng=WORLD_BOUNDS[0], lat=WORLD_BOUNDS[3])
 * Bottom right: (lng=WORLD_BOUNDS[2], lat=WORLD_BOUNDS[1])
"""
WORLD_BOUNDS = [-180, -85.05112878, 180, 85.05112878]


def clamp(value, low=None, high=None):
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return value


def clamp_bounds(bounds_to_clamp, clamp_values):
    x_min = clamp(bounds_to_clamp["x_min"], low=clamp_values["x_min"])
    y_min = clamp(bounds_to_clamp["y_min"], low=clamp_values["y_min"])
    x_max = clamp(bounds_to_clamp["x_max"], low=x_min, high=clamp_values["x_max"])
    y_max = clamp(bounds_to_clamp["y_max"], low=y_min, high=clamp_values["y_max"])
    return create_bounds(bounds_to_clamp["zoom"], x_min, x_max, y_min, y_max, bounds_to_clamp["scheme"])


def extent_overlap_bounds(extent, bounds):
    return (bounds["x_min"] <= extent["x_min"] <= bounds["x_max"] or
            bounds["x_min"] <= extent["x_max"] <= bounds["x_max"]) and\
            (bounds["y_min"] <= extent["y_min"] <= bounds["y_max"] or
             bounds["y_min"] <= extent["y_max"] <= bounds["y_max"])


def create_bounds(zoom, x_min, x_max, y_min, y_max, scheme):
    return {
        "zoom": int(zoom),
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "width": int(x_max - x_min + 1),
        "height": int(y_max - y_min + 1),
        "scheme": scheme
    }


def convert_coordinate(source_crs, target_crs, lat, lng):
    source_crs = get_code_from_epsg(source_crs)
    target_crs = get_code_from_epsg(target_crs)

    crs_src = QgsCoordinateReferenceSystem(source_crs)
    crs_dest = QgsCoordinateReferenceSystem(target_crs)
    if QGIS3:
        xform = QgsCoordinateTransform(crs_src, crs_dest, QgsProject.instance())
    else:
        xform = QgsCoordinateTransform(crs_src, crs_dest)
    try:
        x, y = xform.transform(QgsPoint(lng, lat))
    except TypeError:
        x, y = xform.transform(lng, lat)
    return x, y


def get_code_from_epsg(epsg_string):
    code = str(epsg_string).upper()
    if code.startswith("EPSG:"):
        code = code.replace("EPSG:", "")
    return int(code)


def get_zoom_by_scale(scale):
    if scale < 0:
        return 23
    zoom = 0
    for upper_bound in sorted(_zoom_level_by_upper_scale_bound):
        if scale < upper_bound:
            zoom = _zoom_level_by_upper_scale_bound[upper_bound]
            break
    return zoom


_zoom_level_by_upper_scale_bound = {
    1000000000: 0,
    500000000: 1,
    200000000: 2,
    50000000: 3,
    25000000: 4,
    12500000: 5,
    6500000: 6,
    3000000: 7,
    1500000: 8,
    750000: 9,
    400000: 10,
    200000: 11,
    100000: 12,
    50000: 13,
    25000: 14,
    12500: 15,
    5000: 16,
    2500: 17,
    1500: 18,
    750: 19,
    500: 20,
    250: 21,
    100: 22,
    0: 23
}
