from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse


@api_view(['GET'])
def api_root(request):
    print("req: ", request)
    return JsonResponse({'hello': 'world'})
