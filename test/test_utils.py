import numpy as np
import os
from core.utils import MarchingSquares, georeference
from shapely import geometry
from pygeotile.tile import Tile, Point


def test_marchingsquares():
    p = os.path.join(os.getcwd(), "test", "data", "L.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour(approximization_tolerance=0)
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((1 0, 2 0, 2 1, 2 2, 2 3, 3 3, 4 3, 4 4, 3 4, 2 4, 1 4, 1 3, 1 2, 1 1, 1 0))"


def test_marchingsquares_approx():
    p = os.path.join(os.getcwd(), "test", "data", "L.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour(approximization_tolerance=0.01)
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((1 0, 2 0, 2 3, 4 3, 4 4, 1 4, 1 0))"


def test_marchingsquares_star():
    p = os.path.join(os.getcwd(), "test", "data", "star.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour(2)
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((50 13, 58 38, 81 38, 63 54, 70 81, 51 65, 32 81, 39 54, 21 38, 44 38, 50 13))"


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