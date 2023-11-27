from django.urls import path

from .views import BlogListView, AboutPageView, InputPageView

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('imput/', InputPageView.as_view(), name='imput')
]
