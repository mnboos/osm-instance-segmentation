import numpy as np
import os
from core.utils import MarchingSquares
from shapely import geometry


def test_marchingsquares():
    p = os.path.join(os.getcwd(), "test", "data", "L.bmp")
    m = MarchingSquares.from_file(p)
    points = m.find_contour()
    poly = geometry.Polygon([[x, y] for (x, y) in points])
    assert poly.wkt == "POLYGON ((0 1, 0 2, 1 2, 2 2, 3 2, 3 3, 3 4, 4 4, 4 3, 4 2, 4 1, 3 1, 2 1, 1 1, 0 1))"
