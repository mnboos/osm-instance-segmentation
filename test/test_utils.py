import math
import numpy as np
import os
from core.utils import MarchingSquares, georeference, SleeveFitting
from shapely import geometry
from pygeotile.tile import Tile, Point


def test_hough():
    # p = os.path.join(os.getcwd(), "test", "data", "diag.bmp")
    # p = os.path.join(os.getcwd(), "test", "data", "building.bmp")
    # p = os.path.join(os.getcwd(), "test", "data", "Untitled.bmp")
    p = os.path.join(os.getcwd(), "test", "data", "green.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour(approximization_tolerance=0.01)
    # print(list(geometry.Polygon(points).exterior.coords))
    # print(geometry.Polygon(points).simplify(3, preserve_topology=False).wkt)
    # print(geometry.Polygon(points).simplify(3, preserve_topology=True).wkt)
    # b = geometry.Polygon(points).buffer(1, join_style=3)
    # print(list(b.exterior.coords))
    # print(b.wkt)
    angle, _ = m.main_orientation(angle_in_degrees=True)
    assert 154 == angle


def test_sleeve_step_horiz():
    s = SleeveFitting(start_point=geometry.Point(0, 0), starting_angle=0)
    s.move(step_size=1)
    assert (1.0, 0.0) == s.current_position


def test_sleeve_step_vert():
    s = SleeveFitting(start_point=geometry.Point(0, 0), starting_angle=90)
    s.move(step_size=1)
    assert (0.0, 1.0) == s.current_position


def test_sleeve_step_diag():
    s = SleeveFitting(start_point=geometry.Point(0, 0), starting_angle=45)
    s.move(step_size=1)
    assert (0.71, 0.71) == s.current_position


def test_sleeve_move_diag():
    s = SleeveFitting(start_point=geometry.Point(0, 0), starting_angle=45)
    for i in range(20):
        s.move()
    s.move(angle=90, step_size=-2)
    # assert "" == s.wkt


def test_sleeve_within():
    s = SleeveFitting(start_point=(0,0), starting_angle=0)
    assert s.within_sector((0, 0))


def test_sleeve_within_vertical():
    s = SleeveFitting(start_point=(0,0), starting_angle=90)
    assert s.within_sector((0, 1))


def test_sleeve_not_within():
    s = SleeveFitting(start_point=(0,0), starting_angle=0)
    assert not s.within_sector((0, 1))


def test_sleeve_fitting():
    p = os.path.join(os.getcwd(), "test", "data", "bigL.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour(approximization_tolerance=0.1)
    hough_angle, nearest_point = m.main_orientation(angle_in_degrees=True)
    rotation_angle = hough_angle+90
    print("main orientation: ", rotation_angle)
    s = SleeveFitting(start_point=nearest_point, starting_angle=rotation_angle)
    s.fit_sleeve(points)


def test_sleeve_sector():
    s = SleeveFitting(start_point=geometry.Point(0, 0), starting_angle=0)
    assert s.within_sector((2, 0.0))
    assert not s.within_sector((2, 1))


def test_sleeve_move():
    s = SleeveFitting(start_point=geometry.Point(0, 0), starting_angle=0)
    s.move(step_size=1)
    assert (1.0, 0) == s.current_position


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