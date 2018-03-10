import sys
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser, FormParser
from django.http import JsonResponse
from .serializers import InferenceRequestSerializer, InferenceRequest
# from .core.predict import Predictor
from core.predict import Predictor
import os
import base64
import numpy as np
from PIL import Image
from pygeotile.tile import Tile, Point
import io
import tempfile
from shapely import geometry
import json
import traceback

_predictor = Predictor(os.path.join(os.getcwd(), "model", "mask_rcnn_osm_0248.h5"))


"""
Request format (url: localhost:8000/inference):
{
    "bbox": {
        "lat_min": 12,
        "lat_max": 12,
        "lon_min": 12,
        "lon_max": 12
    },
    "image_data": "123"
}
"""


@api_view(['GET', 'POST'])
def request_inference(request):
    if request.method == "GET":
        return JsonResponse({'hello': 'world'})
    else:
        data = JSONParser().parse(request)
        inference_serializer = InferenceRequestSerializer(data=data)
        if not inference_serializer.is_valid():
            return JsonResponse({'errors': inference_serializer.errors})

        inference = InferenceRequest(**inference_serializer.data)
        try:
            res = _predict(inference)
            coll = "GEOMETRYCOLLECTION({})".format(", ".join(res))
            return JsonResponse({'features': res})
        except Exception as e:
            tb = ""
            if traceback:
                tb = traceback.format_exc()
            print("Server error: {}, {}", sys.exc_info(), tb)
            msg = str(e)
            return JsonResponse({'error': msg})


def _predict(request: InferenceRequest):
    b64 = base64.b64decode(request.image_data)
    barr = io.BytesIO(b64)
    img = Image.open(barr)
    img = img.convert("RGB")
    img = img.crop((0, 0, 256, 256))
    img.save(r"D:\training_images\_last_predicted\img.png")
    print("Image shape: ", img.size)
    arr = np.asarray(img)
    tile = Tile.for_point(point=Point(latitude=request.lat, longitude=request.lon), zoom=int(request.zoom_level))
    res = _predictor.predict_array(img_data=arr, tile=tile, do_rectangularization=request.rectangularize)
    polygons = [geometry.Polygon(points) for points in res]
    # return list(map(lambda p: json.dumps(geometry.mapping(p)), polygons))
    return list(map(lambda p: p.wkt, polygons))

