from django.urls import path
from .views import *

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about/', AboutPageView.as_view(), name='about'),
    path('imput/', InputPageView.as_view(), name='imput'),
    path('login/', LoginUser.as_view(), name='login'),
    path('logout/', logout_user, name='logout'),
    path('register/', RegisterUser.as_view(), name='register'),
    path('profile/', profile, name='profile'),
    path('search/', SearchView.search_view, name='search'),
    path('advanced_search/', AdvancedSearchView.advanced_search_view, name='advanced_search'),
    path('<slug:slug>/', PostDetail.as_view(), name='post_detail')
    ]
