# -*- coding: utf-8 -*-
"""
# @Time    : 2021/6/11 下午10:02
# @Author  : Jiacai Yi
# @FileName: urls.py
# @E-mail  ：1076365758@qq.com
"""
from django.urls import path, include
from . import views

app_name = 'database'
urlpatterns = [
    path('property/', views.property, name="property"),
    path('property-data-source/', views.peroperty_datasource, name='peroperty_data_source'),
    path('gdetail/<idx>/', views.tranformation_detail, name='transformation_detail'),
    path('pgdetail/<idx>/', views.pre_tranformation_detail, name='pre_transformation_detail'),
    path('local-data-source/<idx>/', views.local_datasource, name='local_data_source'),
    path('plocal-data-source/<idx>/', views.pre_local_datasource, name='pre_local_data_source'),
    path('ldetail/<idx>/', views.local_detail, name='local_detail'),
    path('pldetail/<idx>/', views.pre_local_detail, name='pre_local_detail'),
    path('mmp-data-source/<idx>/', views.mmp_datasource, name='mmp_data_source'),
    path('pmmp-data-source/<idx>/', views.pre_mmp_datasource, name='pre_mmp_data_source'),
    path('mmp-ldata-source/<idx>/', views.mmp_ldatasource, name='mmp_ldata_source'),
    path('pmmp-ldata-source/<idx>/', views.pre_mmp_ldatasource, name='pre_mmp_ldata_source'),
    path('structure/', views.structure, name="structure"),
    path('structure-data-source/', views.structure_datasource, name='structure_data_source'),
    path('sdetail/<idx>/', views.structure_detail, name='structure_detail'),
    path('psdetail/<idx>/', views.pre_structure_detail, name='pre_structure_detail'),
    path('global-data-source/<idx>/', views.global_datasource, name='global_data_source'),
    path('pglobal-data-source/<idx>/', views.pre_global_datasource, name='pre_global_data_source'),
    path('ldata-source/<idx>/', views.ldatasource, name='ldatasource'),
    path('pldata-source/<idx>/', views.pre_ldatasource, name='pre_ldatasource'),
]
