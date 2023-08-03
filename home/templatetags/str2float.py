# -*- coding: utf-8 -*-
"""
# @Time    : 2021/1/30 上午10:36
# @Author  : Jiacai Yi
# @FileName: str2float.py
# @E-mail  ：1076365758@qq.com
"""
from django.template import Library

# 将注册类实例化为register对象
register = Library()


@register.filter(name='str2float')
def str2float(string):
    return float(string)
