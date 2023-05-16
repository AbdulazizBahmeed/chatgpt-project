from django.urls import path

from . import views

urlpatterns = [
    path("", views.HomePageView.as_view(), name="homePage"),
    path("analyze", views.analyze, name="homePage"),
]