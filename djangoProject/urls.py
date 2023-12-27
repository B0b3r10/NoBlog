from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include("temperature.urls")),
    path('', include('myblog.urls'))

]
