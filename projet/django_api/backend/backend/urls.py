from django.urls import include, path
from rest_framework import routers
from api import views


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('aifromfile', views.getFromFile),
    path('aifromurl', views.getFromUrl),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
