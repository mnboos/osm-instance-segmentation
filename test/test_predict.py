import os
from core.predict import Predictor
from shapely.geometry import Polygon
from pygeotile.tile import Tile


def test_predict():
    weights_path = os.path.join(os.getcwd(), "model", "mask_rcnn_osm_0030.h5")
    assert os.path.isfile(weights_path)

    img_path = os.path.join(os.getcwd(), "test", "data", "18_139423_171197.tiff")
    p = Predictor(weights_path)
    polygon_points = p.predict_path(img_path=img_path)
    for points in polygon_points:
        p = Polygon(points)
        print(p.wkt)
    assert 1 == 1
