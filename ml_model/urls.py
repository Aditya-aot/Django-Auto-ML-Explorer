from django.urls import path
from . import views


urlpatterns = [
    path('', views.home , name='home') , 
    path('functions/', views.functions , name='functions') , 

    path('dataset_input/', views.data_in , name='data_in') , 
    path('analysis_input/', views.data_analysis_in , name='data_analysis_in') ,

    path('analysis_dashboard/', views.analysis_dashboard , name='analysis_dashboard') ,
    path('upload-csv/', views.upload_csv, name='upload_csv'),
    path('train-model/', views.train_model, name='train_model'),

    path('dashboard/', views.dashboard, name='dashboard'),

]