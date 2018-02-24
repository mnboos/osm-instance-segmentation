import math
import numpy as np
import os
from core.utils import MarchingSquares, georeference, get_angle, \
    parallel_or_perpendicular
from shapely import geometry
from pygeotile.tile import Tile, Point
from scipy.optimize import curve_fit



def make_fit_func(angle: float):
    def fit_line(x, m, b):
        a = angle
        return m * x + b
    return fit_line


def test_parallel():
    l1 = ((79.04974365234375, 50.8869743347168), (88.85025024414063, 38.75302505493164))
    l2 = ((7.280234336853027, 153.7339172363281), (12.71976566314697, 146.2660827636719))
    is_parallel, is_perpendicular = parallel_or_perpendicular(l1, l2)
    assert is_parallel
    assert not is_perpendicular


def test_angle_vertical_down():
    is_parallel, is_perpendicular = parallel_or_perpendicular(((0,0), (1,0)), ((2,2), (2,3)))
    assert not is_parallel
    assert is_perpendicular


def test_angle_vertical_up():
    is_parallel, is_perpendicular = parallel_or_perpendicular(((0,0), (1,0)), ((2,2), (2,1)))
    assert not is_parallel
    assert is_perpendicular


def test_angle_horizontal_right():
    is_parallel, is_perpendicular = parallel_or_perpendicular(((0,0), (1,0)), ((2,2), (3,2)))
    assert is_parallel
    assert not is_perpendicular


def test_angle_horizontal_left():
    is_parallel, is_perpendicular = parallel_or_perpendicular(((0,0), (1,0)), ((-2,-2), (-3,-2)))
    assert is_parallel
    assert not is_perpendicular


def test_get_angle_horizontal():
    assert 0 == get_angle(((1, 0), (0, 0)))


def test_get_angle_vertical():
    assert 90 == get_angle(((0, 0), (0, 1)))


def test_get_angle_diagonal():
    assert 45 == get_angle(((0, 0), (1, 1)))


def test_marchingsquares():
    p = os.path.join(os.getcwd(), "test", "data", "L.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour()
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((1 0, 2 0, 2 1, 2 2, 2 3, 3 3, 4 3, 4 4, 3 4, 2 4, 1 4, 1 3, 1 2, 1 1, 1 0))"


def test_marchingsquares_approx():
    p = os.path.join(os.getcwd(), "test", "data", "L.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour()
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((1 0, 2 0, 2 3, 4 3, 4 4, 1 4, 1 0))"


def test_marchingsquares_star():
    p = os.path.join(os.getcwd(), "test", "data", "star.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour()
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((50 13, 51 13, 52 13, 52 14, 52 15, 52 16, 52 17, 53 17, 53 18, 53 19, 53 20, 53 21, " \
                       "54 21, 54 22, 54 23, 54 24, 54 25, 55 25, 55 26, 55 27, 55 28, 56 28, 56 29, 56 30, 56 31, " \
                       "56 32, 57 32, 57 33, 57 34, 57 35, 57 36, 58 36, 58 37, 58 38, 59 38, 60 38, 61 38, 62 38, " \
                       "63 38, 64 38, 65 38, 66 38, 67 38, 68 38, 69 38, 70 38, 71 38, 72 38, 73 38, 74 38, 75 38, " \
                       "76 38, 77 38, 78 38, 79 38, 80 38, 81 38, 81 39, 81 40, 80 40, 79 40, 79 41, 78 41, 78 42, " \
                       "77 42, 77 43, 76 43, 76 44, 75 44, 75 45, 74 45, 74 46, 73 46, 72 46, 72 47, 71 47, 71 48, " \
                       "70 48, 70 49, 69 49, 69 50, 68 50, 68 51, 67 51, 67 52, 66 52, 66 53, 65 53, 64 53, 64 54, " \
                       "63 54, 63 55, 63 56, 63 57, 64 57, 64 58, 64 59, 64 60, 64 61, 65 61, 65 62, 65 63, 65 64, " \
                       "66 64, 66 65, 66 66, 66 67, 66 68, 67 68, 67 69, 67 70, 67 71, 67 72, 68 72, 68 73, 68 74, " \
                       "68 75, 69 75, 69 76, 69 77, 69 78, 69 79, 70 79, 70 80, 70 81, 69 81, 68 81, 68 80, 67 80, " \
                       "67 79, 66 79, 66 78, 65 78, 64 78, 64 77, 63 77, 63 76, 62 76, 62 75, 61 75, 61 74, 60 74, " \
                       "60 73, 59 73, 59 72, 58 72, 58 71, 57 71, 56 71, 56 70, 55 70, 55 69, 54 69, 54 68, 53 68, " \
                       "53 67, 52 67, 52 66, 51 66, 51 65, 50 65, 50 66, 49 66, 49 67, 48 67, 48 68, 47 68, 47 69, " \
                       "46 69, 46 70, 45 70, 45 71, 44 71, 44 72, 43 72, 42 72, 42 73, 41 73, 41 74, 40 74, 40 75, " \
                       "39 75, 39 76, 38 76, 38 77, 37 77, 37 78, 36 78, 35 78, 35 79, 34 79, 34 80, 33 80, 33 81, " \
                       "32 81, 32 80, 32 79, 33 79, 33 78, 33 77, 33 76, 33 75, 34 75, 34 74, 34 73, 34 72, 35 72, " \
                       "35 71, 35 70, 35 69, 35 68, 36 68, 36 67, 36 66, 36 65, 36 64, 37 64, 37 63, 37 62, 37 61, " \
                       "38 61, 38 60, 38 59, 38 58, 38 57, 39 57, 39 56, 39 55, 39 54, 38 54, 38 53, 37 53, 37 52, " \
                       "36 52, 35 52, 35 51, 34 51, 34 50, 33 50, 33 49, 32 49, 32 48, 31 48, 31 47, 30 47, 30 46, " \
                       "29 46, 28 46, 28 45, 27 45, 27 44, 26 44, 26 43, 25 43, 25 42, 24 42, 24 41, 23 41, 23 40, " \
                       "22 40, 22 39, 21 39, 21 38, 22 38, 23 38, 24 38, 25 38, 26 38, 27 38, 28 38, 29 38, 30 38, " \
                       "31 38, 32 38, 33 38, 34 38, 35 38, 36 38, 37 38, 38 38, 39 38, 40 38, 41 38, 42 38, 43 38, " \
                       "44 38, 44 37, 44 36, 44 35, 45 35, 45 34, 45 33, 45 32, 45 31, 46 31, 46 30, 46 29, 46 28, " \
                       "46 27, 47 27, 47 26, 47 25, 47 24, 48 24, 48 23, 48 22, 48 21, 48 20, 49 20, 49 19, 49 18, " \
                       "49 17, 49 16, 50 16, 50 15, 50 14, 50 13)) "


def test_roundtrip():
    zoom_level = 18
    t = Tile.from_tms(139423, 171197, zoom_level)
    minx, maxy = t.bounds[0].pixels(zoom_level)
    maxx, miny = t.bounds[1].pixels(zoom_level)
    p = Point.from_pixel(maxx, maxy, zoom=zoom_level)
    mx, my = p.meters
    t2 = Tile.for_meters(mx, my, zoom_level)
    assert t.quad_tree == t2.quad_tree


def test_georeference():
    tile = Tile.from_tms(139423, 171197, 18)
    img_path = os.path.join(os.getcwd(), "test", "data", "18_139423_171197.tif")
    m = MarchingSquares.from_file(img_path)
    points = m.find_contour()
    geo_points = georeference(points, tile)
    p = geometry.Polygon(geo_points)
    print(p.wkt)
    assert p.wkt == "POLYGON ((11.46890044212341 48.1641425687818, 11.46890580654145 48.1641425687818, 11.46890580654145 48.16404238298088, 11.46879851818085 48.16404238298088, 11.46879851818085 48.16413899072083, 11.46890044212341 48.16413899072083, 11.46890044212341 48.1641425687818))"