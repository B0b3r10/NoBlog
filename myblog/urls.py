from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about', AboutPageView.as_view(), name='about'),
    path('imput/', InputPageView.as_view(), name='imput'),
    path('search/', SearchView.search_view, name='search'),
    path('advanced_search/', AdvancedSearchView.advanced_search_view, name='advanced_search'),
    path('login/', LoginUser.as_view(), name='login'),
    path('logout/', logout_user, name='logout'),
    path('register/', RegisterUser.as_view(), name='register'),
    path('profile/', profile, name='profile'),
    path('<slug:slug>/', PostDetail.as_view(), name='post_detail')
    ]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)