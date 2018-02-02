from rest_framework import serializers


class BoundingBox(object):
    def __init__(self, lat_min, lon_min, lat_max, lon_max):
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max


class InferenceRequest(object):
    def __init__(self, bbox, image_data):
        self.bbox = bbox
        self.image_data = image_data


class BoundingBoxSerializer(serializers.Serializer):
    lat_min = serializers.FloatField(min_value=-85.05112878, max_value=85.05112878)
    lat_max = serializers.FloatField(min_value=-85.05112878, max_value=85.05112878)
    lon_min = serializers.FloatField(min_value=-180, max_value=180)
    lon_max = serializers.FloatField(min_value=-180, max_value=180)


class InferenceRequestSerializer(serializers.Serializer):
    bbox = BoundingBoxSerializer(required=True)
    image_data = serializers.CharField(required=True, allow_blank=False, allow_null=False)