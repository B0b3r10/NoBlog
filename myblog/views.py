from django.views.generic import ListView, TemplateView, DetailView
from .models import Post
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
from django.views import generic


class BlogListView(ListView):
    paginate_by = 4
    model = Post
    template_name = 'home.html'


class AboutPageView(TemplateView):
    template_name = 'about.html'


class InputPageView(TemplateView):
    template_name = 'imput.html'


class PostDetail(generic.DetailView):
    model = Post
    template_name = 'post_detail.html'

