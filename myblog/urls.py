from django.urls import path
from .views import *

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('imput/', InputPageView.as_view(), name='imput'),
    path('search/', SearchView.search_view, name='search'),
    path('search_post/', SearchViewPost.as_view(), name='search_post'),
    path('<slug:slug>/', PostDetail.as_view(), name='post_detail'),
    ]
