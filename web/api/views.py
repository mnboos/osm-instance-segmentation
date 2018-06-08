import sys
import math
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from django.http import JsonResponse
from .serializers import InferenceRequestSerializer, InferenceRequest
from core.utils import georeference, rectangularize
from core.predict import Predictor
import base64
import numpy as np
from PIL import Image
import io
from shapely import geometry, wkt
import geojson
import traceback

# _predictor = Predictor(r"D:\_models\stage2_hombi_rappi_zh.h5")
_predictor = Predictor(r"D:\_models\mask_rcnn_osm_0100.h5")


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


def diff(a, b, check_intersection=True, check_containment=False, min_area=20):
    """
     * Returns a representative point for each feature from a, that has no intersecting feature in b
    :param min_area:
    :param check_containment:
    :param check_intersection:
    :param a:
    :param b:
    :return:
    """

    res = []
    for feature_a, class_name_a in a:
        if not feature_a.area >= min_area:
            continue

        hit = False
        for feature_b, class_name_b in b:
            if not feature_b.area >= min_area:
                continue

            if (check_intersection and feature_b.intersects(feature_a)) \
                    or (check_containment and feature_b.within(feature_a)):
                hit = True
                break
        if (check_intersection and not hit) or (check_containment and hit):
            res.append((feature_a, class_name_a))
    return res


def to_final_geojson(features, props, add_predicted_class_to_props=False, to_point=False):
    res = []
    if not props and add_predicted_class_to_props:
        props = {}
    for f, class_name in features:
        if add_predicted_class_to_props:
            props['class'] = class_name
        # p = f
        if not f.is_valid:
            continue
        if to_point:
            p = f.representative_point().buffer(4)
        else:
            p = f
        res.append(to_geojson(p, properties=props))
    return res


@api_view(['GET', 'POST'])
def request_inference(request):
    if request.method == "GET":
        return JsonResponse({'hello': 'world'})
    else:
        data = JSONParser().parse(request)
        inference_serializer = InferenceRequestSerializer(data=data)
        if not inference_serializer.is_valid():
            print("Errors: ", inference_serializer.errors)
            return JsonResponse({'errors': inference_serializer.errors})

        inference = InferenceRequest(**inference_serializer.data)
        try:
            res = _predict(inference)
            # coll = "GEOMETRYCOLLECTION({})".format(", ".join(res))
            # with open(r"D:\training_images\_last_predicted\wkt.txt", 'w') as f:
            #     f.write(coll)

            ref_features = list(map(lambda f: (wkt.loads(f), 'reference'), inference.reference_features))

            original = list(res)

            deleted_features = diff(ref_features, res)
            added_features = diff(res, ref_features)
            changed_features = diff(res, ref_features, check_intersection=False, check_containment=True)
            print("Deleted: ", len(deleted_features))
            print("Added: ", len(added_features))
            print("Changed: ", len(changed_features))
            print("Done")

            output = {
                'features': list(map(lambda feat: to_geojson(geom=feat[0], properties={'type': feat[1], 'area': feat[0].area}), original)),
                'deleted': to_final_geojson(deleted_features, {'type': 'deleted'}, to_point=True),
                'added': to_final_geojson(added_features, {'type': 'added'}, True),
                'changed': to_final_geojson(changed_features, {'type': 'changed'})
            }

            return JsonResponse(output)
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
    print("Received image size: ", img.size)
    extent = {
        'x_min': request.x_min,
        'y_min': request.y_min,
        'x_max': request.x_max,
        'y_max': request.y_max,
        'img_width': width,
        'img_height': height
    }

    img_size = 256

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
            # img_copy = img_copy.resize((1024, 1024), Image.ANTIALIAS)
            img_copy.save(r"C:\Users\Martin\AppData\Local\Temp\deep_osm\cropped.png")
            print("Cropped image size: ", img_copy.size)
            arr = np.asarray(img_copy)
            img_id = "img_id_{}_{}".format(col, row)
            tiles_by_img_id[img_id] = (col, row)
            images_to_predict.append((arr, img_id))
            break
        break
    point_sets = _predictor.predict_arrays(images=images_to_predict)
    # print(point_sets)

    for points, img_id, class_name in point_sets:
        col, row = tiles_by_img_id[img_id]
        points = list(map(lambda p: (p[0]+col*256, p[1]+row*256), points))
        if request.rectangularize:
            points = rectangularize(points)
        georeffed = georeference(points, extent)
        if georeffed:
            points = georeffed
        polygon = geometry.Polygon(points)
        all_polygons.append((polygon, class_name))

    return all_polygons
    # results = list(map(to_geojson, all_polygons))
    # print(results)
    # return results


def to_geojson(geom, properties=None):
    props = {}
    if properties:
        props = properties
    f = geojson.Feature(geometry=geom, properties=props)
    return f
