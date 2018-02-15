import math
import numpy as np
import os
from skimage.measure import approximate_polygon
from core.utils import MarchingSquares, georeference, SleeveFitting, root_mean_square_error, get_angle, parallel_or_perpendicular
from shapely import geometry
from pygeotile.tile import Tile, Point
import cv2
from PIL import Image


def test_hough():
    # p = os.path.join(os.getcwd(), "test", "data", "diag.bmp")
    # p = os.path.join(os.getcwd(), "test", "data", "building.bmp")
    # p = os.path.join(os.getcwd(), "test", "data", "Untitled.bmp")
    p = os.path.join(os.getcwd(), "test", "data", "bigL.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour(approximization_tolerance=0.01)
    main_orientation = m.main_orientation(True)
    original_points = []
    original_points.extend(points)
    im = Image.open(p).convert("L")
    img = np.asarray(im)
    all_wkts = []
    lines = []
    while points:
        seg = []
        while points and len(seg) < 3:
            seg.append(points.pop())
        thre = 2
        while True and points:
            # p_i = points.pop()
            # seg.append(points.pop())
            err = root_mean_square_error(seg[-1], points[-1])
            if err <= thre:
                seg.append(points.pop())
            else:
                break

        [vx, vy, x, y] = np.round(cv2.fitLine(points=np.asarray(seg, dtype=np.int32), distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01),2)
        if len(seg) >= 3:
            dist = geometry.Point(seg[0]).distance(geometry.Point(seg[-1]))
            x1 = float(x - dist/2 * vx)
            x2 = float(x + dist/2 * vx)
            y1 = float(y - dist/2 * vy)
            y2 = float(y + dist/2 * vy)
            wkt2 = geometry.LineString([(x1, y1), (x2, y2)]).wkt
            lines.append(((x1, y1), (x2, y2)))
            all_wkts.append(wkt2)
            # print("wkt 2:\n", wkt2)

    grouped_lines = {}
    lines = sorted(lines, key=lambda l: geometry.LineString(l).length)
    while lines:
        longest_line = lines.pop()
        main_angle = get_angle(longest_line)
        group = [longest_line]
        for l in lines.copy():
            is_parallel, is_perpendicular = parallel_or_perpendicular(longest_line, l)
            if is_parallel or is_perpendicular:
                group.append(l)
                lines.remove(l)
        grouped_lines[main_angle] = group

    for a in grouped_lines:
        print("angle: ", a)
        print(",".join(map(lambda l: geometry.LineString(l).wkt, grouped_lines[a])))

        # print("angle: ", a)

    all_wkts = map(lambda l: geometry.LineString(l).wkt, lines)
    totalwkt = ",".join(all_wkts)
    a = ""
    cv2.imwrite("lines.bmp", img)
    a = ""


    # print(list(geometry.Polygon(points).exterior.coords))
    # print(geometry.Polygon(points).simplify(3, preserve_topology=False).wkt)
    # print(geometry.Polygon(points).simplify(3, preserve_topology=True).wkt)
    # b = geometry.Polygon(points).buffer(1, join_style=3)
    # print(list(b.exterior.coords))
    # print(b.wkt)
    angle, _ = m.main_orientation(angle_in_degrees=True)
    assert 34 == angle


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
    points = m.find_contour(approximization_tolerance=10)
    # p = geometry.Polygon(points)
    # print("wkt:\n", p.wkt)
    # assert 1==2

    # hough_angle, nearest_point = m.main_orientation(angle_in_degrees=True)

    # rotation_angle = hough_angle+90
    # print("main orientation: ", rotation_angle)
    # s = SleeveFitting(start_point=nearest_point, starting_angle=rotation_angle)
    # s.fit_sleeve(points)


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