from django.contrib.auth.views import LoginView
from django.views.generic import ListView, TemplateView, CreateView
from .models import *
from django.urls import reverse_lazy
from django.contrib.auth import logout, login
from django.shortcuts import redirect
from .forms import *


class BlogListView(ListView):
    model = Post
    template_name = 'home.html'


class AboutPageView(TemplateView):
    template_name = 'about.html'


class InputPageView(TemplateView):
    template_name = 'imput.html'

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
    return redirect('login')

# Registration Window
# from django.shortcuts import render
# from .forms import CustomUserCreationForm, CustomAuthenticationForm
#
# def login_view(request):
#     if request.method == 'POST':
#         form = CustomAuthenticationForm(request, request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             login(request, user)
#             return redirect('home')
#     else:
#         form = CustomAuthenticationForm()
#     return render(request, 'login2.html', {'form': form})
#
# def register_view(request):
#     if request.method == 'POST':
#         form = CustomUserCreationForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user)
#             return redirect('home')
#     else:
#         form = CustomUserCreationForm()
#     return render(request, 'register2.html', {'form': form})