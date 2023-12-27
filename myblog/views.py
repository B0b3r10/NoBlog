from django.contrib.auth import authenticate, login, logout
from django.views.generic import ListView, TemplateView, CreateView, DetailView
from django.views.generic.edit import FormView
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from .forms import *
from .models import *

class BlogListView(ListView):
    model = Post
    template_name = 'home.html'
    paginate_by = 4


class AboutPageView(TemplateView):
    template_name = 'about.html'


class InputPageView(TemplateView):
    template_name = 'imput.html'


class PostDetail(DetailView):
    model = Post
    template_name = 'post_detail.html'

class RegisterUser(CreateView):
    form_class = RegisterUserForm
    template_name = 'register.html'
    success_url = reverse_lazy('login')

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Регистрация"
        return context

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('home')


class LoginUser(LoginView):
    form_class = LoginUserForm
    template_name = 'login.html'

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = "Авторизация"
        return context

    def get_success_url(self):
        return reverse_lazy('home')


def logout_user(request):
    logout(request)
    return redirect('home')

@login_required
def profile(request):
    user_profiles = UserProfile.objects.filter(user=request.user)

    if user_profiles.exists():
        user_profile = user_profiles.first()
    else:
        user_profile = UserProfile(user=request.user)
        user_profile.save()

    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=user_profile)
        if form.is_valid():
            form.save()
    else:
        form = UserProfileForm(instance=user_profile)

    return render(request, 'profile.html', {'form': form, 'user_profile': user_profile})


def edit_profile(request):
    user_profile = request.user.userprofile

    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=user_profile)
        if form.is_valid():
            form.save()
            return redirect('profile')  # Перенаправьте пользователя на страницу профиля после сохранения
    else:
        form = UserProfileForm(instance=user_profile)

    return render(request, 'profile.html', {'form': form, 'user_profile': user_profile})


class SearchView(TemplateView):
    def search_view(request):
        query = request.GET.get('q')

        if not query:
            return redirect('home')

        results = Post.objects.filter(title__icontains=query) | Post.objects.filter(body__icontains=query)

        context = {'results': results, 'query': query, 'item': "post"}
        return render(request, 'search_results.html', context)


class AdvancedSearchView(TemplateView):
    def advanced_search_view(request):
        form = AdvancedSearchForm(request.GET)
        results = []

        if form.is_valid():
            title_query = form.cleaned_data.get('title', '')
            body_query = form.cleaned_data.get('body', '')

            results = Post.objects.filter(title__icontains=title_query, body__icontains=body_query)

        context = {'form': form, 'results': results}
        return render(request, 'advanced_search.html', context)