from django.views.generic import ListView, TemplateView, DetailView
from .models import Post
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
from django.views import generic
from django.shortcuts import redirect, render, get_object_or_404
from django.views import View
from .forms import SearchForm

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

class SearchView(TemplateView):
    def search_view(request):
        query = request.GET.get('q', '')

        if not query:
            return redirect('home')

        results = Post.objects.filter(title__icontains=query) | Post.objects.filter(body__icontains=query)

        context = {'results': results, 'query': query, 'item': "post"}
        return render(request, 'search_results.html', context)

from .forms import SearchForm

class SearchViewPost(View):
    def get(self, request):
        search_form = SearchForm(request.GET)
        if search_form.is_valid():
            search_word = search_form.cleaned_data['search_word']
            posts = Post.objects.filter(body__icontains=search_word) | Post.objects.filter(title__icontains=search_word)
            return render(request, 'search_results_post.html', {'posts': posts})
        else:
            return render(request, 'search_results_post.html', {'posts': []})