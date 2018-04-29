import sys
import math
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
from .serializers import InferenceRequestSerializer, InferenceRequest
from core.utils import get_contour, georeference, rectangularize
from core.predict import Predictor
import base64
import numpy as np
from PIL import Image
import io
from shapely import geometry
import traceback
from pycocotools import mask as cocomask

_predictor = Predictor(r"D:\_mapping-challenge\stage2_0.833.h5")


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
        print("Inf: ", inference)
        try:
            res = _predict(inference)
            # coll = "GEOMETRYCOLLECTION({})".format(", ".join(res))
            # with open(r"D:\training_images\_last_predicted\wkt.txt", 'w') as f:
            #     f.write(coll)
            return JsonResponse({'features': res})
        except Exception as e:
            tb = ""
            if traceback:
                tb = traceback.format_exc()
            print("Server error: {}, {}", sys.exc_info(), tb)
            msg = str(e)
            return JsonResponse({'error': msg})


def _predict(request: InferenceRequest):
    print("Decoding image")
    b64 = base64.b64decode(request.image_data)
    print("Image decoded")
    barr = io.BytesIO(b64)
    img = Image.open(barr)
    img = img.convert("RGB")
    width, height = img.size
    extent = {
        'x_min': request.x_min,
        'y_min': request.y_min,
        'x_max': request.x_max,
        'y_max': request.y_max,
        'img_width': width,
        'img_height': height
    }

    img_size = 1024

    all_polygons = []
    cols = int(math.ceil(width / float(img_size)))
    rows = int(math.ceil(height / float(img_size)))
    images_to_predict = []
    tiles_by_img_id = {}
    for col in range(0, cols):
        for row in range(0, rows):
            print("Processing tile (x={},y={})".format(col, row))
            start_width = col * img_size
            start_height = row * img_size
            img_copy = img.crop((start_width, start_height, start_width+img_size, start_height+img_size))
            arr = np.asarray(img_copy)
            img_id = "img_id_{}_{}".format(col, row)
            tiles_by_img_id[img_id] = (col, row)
            images_to_predict.append((arr, img_id))
    point_set = _predictor.predict_arrays(images=images_to_predict)
    for rle, score, img_id in point_set:
        col, row = tiles_by_img_id[img_id]
        mask = cocomask.decode(rle)
        mask = mask.reshape((img_size, img_size))
        points = get_contour(mask)
        points = list(map(lambda p: (p[0]+col*256, p[1]+row*256), points))
        if request.rectangularize:
            points = rectangularize(points)
        georeffed = georeference(points, extent)
        if georeffed:
            points = georeffed
        polygon = geometry.Polygon(points)
        all_polygons.append(polygon)

    return list(map(lambda p: p.wkt, all_polygons))

