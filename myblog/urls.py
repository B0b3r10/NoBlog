from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static
from TemperatureModel import views as Tviews
from TemperatureModel import views


urlpatterns = [
    path('', BlogListView.as_view(), name='home'),
    path('about', AboutPageView.as_view(), name='about'),
    path('imput', InputPageView.as_view(), name='imput'),
    path('login', LoginUser.as_view(), name='login'),
    path('logout', logout_user, name='logout'),
    path('register', RegisterUser.as_view(), name='register'),
    path('profile', profile, name='profile'),
    path('search/', SearchView.search_view, name='search'),
    path('advanced_search/', AdvancedSearchView.advanced_search_view, name='advanced_search'),
    path('search', SearchView.search_view, name='search'),
    path('Temperature_learning', Tviews.Temperature_learning,name='Temperature_learning'),
    path('Temperature_predict', Tviews.Temperature_predict,name='Temperature_predict'),
    path('Toxic_learning', views.Toxic_learning, name='Toxic_learning'),
    path('Toxic_predict', views.Toxic_predict, name='Toxic_predict'),
    # path('search_post', SearchViewPost.as_view(), name='search_post'),
    path('<slug:slug>/', PostDetail.as_view(), name='post_detail')
    ]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
