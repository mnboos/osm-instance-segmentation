from rest_framework import serializers
import base64


class InferenceRequest(object):
    def __init__(self, lat: float, lon: float, zoom_level: int, image_data: str, approximiation_tolerance: float = None):
        self.lat = lat
        self.lon = lon
        self.zoom_level = zoom_level
        self.image_data = image_data
        self.approximiation_tolerance = approximiation_tolerance


def validate_base64(s: str) -> None:
    try:
        base64.standard_b64decode(s)
    except Exception:
        msg = "Ensure this value is a base64 encoded."
        raise serializers.ValidationError(msg)


class InferenceRequestSerializer(serializers.Serializer):
    approximiation_tolerance = serializers.FloatField(required=False,
                                                      min_value=0)
    lat = serializers.FloatField(required=True,
                                 min_value=-85.05112878,
                                 max_value=85.05112878,
                                 help_text="Latitude of the top left corner regarding the image to be tested")
    lon = serializers.FloatField(required=True,
                                 min_value=-180,
                                 max_value=180,
                                 help_text="Longitude of the top left corner regarding the image")
    zoom_level = serializers.FloatField(required=True,
                                        min_value=17,
                                        max_value=19)
    image_data = serializers.CharField(required=True,
                                       allow_blank=False,
                                       allow_null=False,
                                       help_text="Image data as base64 encoded string",
                                       validators=[validate_base64])
