from django.urls import path
from . import views

urlpatterns = [
    path('',views.main,name='home'),
    path('handle-query/', views.handle_query, name='handle_query')
]