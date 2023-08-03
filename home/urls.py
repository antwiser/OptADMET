# -*- coding: utf-8 -*-
"""
# @Time    : 2021/6/11 下午3:45
# @Author  : Jiacai Yi
# @FileName: urls.py
# @E-mail  ：1076365758@qq.com
"""
from django.urls import path, include
from . import views

app_name = 'home'
urlpatterns = [
    path('', views.index, name="index"),
    path('tutorial/', views.tutorials, name="tutorials"),
    path('contact/', views.contact, name="contact"),
    # path('download/', views.download, name="download"),
    path('pub/', views.publication, name="pub"),
    path('terms/', views.term, name="term"),
    path('checker/', views.checker, name="checker"),
    path('checkerCal/', views.checkercal, name='checkercal'),
    path('result-datasource/', views.result_datasource, name='result_datasource'),
    path('result/<str:filename>', views.result_file, name='result'),
    path('result/<str:filename>/<str:index>/<str:data_idx>', views.result_mol, name="result_mol"),
    path('result/<str:filename>/<str:index>/<str:data_idx>/<str:dire_idx>', views.result_mol_direct, name="result_mol"),
    path('screening/', views.final_screening, name='final_screening'),
    path('final_result/<str:filename>/<str:properties>', views.final_result_file, name='final_result'),
    path('final-result-datasource/', views.final_result_datasource, name='final_result_datasource'),
    path('s_download/', views.s_download, name="s_download"),
    path('d_download/', views.d_download, name="d_download"),
    path('d_pdf/<str:filename>/<str:index>', views.d_pdf, name="d_pdf"),
]
