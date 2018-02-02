from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser, FormParser
from django.http import JsonResponse
from web.api.serializers import InferenceRequestSerializer, InferenceRequest


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
            return JsonResponse({'error': inference_serializer.errors})

        inference = InferenceRequest(**inference_serializer.data)
        print("Inf: ", inference)
        return JsonResponse({'postresponse': 'hello world'})
