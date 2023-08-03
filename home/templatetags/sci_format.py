# -*- coding: utf-8 -*-
"""
# @Time    : 2021/1/30 上午11:27
# @Author  : Jiacai Yi
# @FileName: sci_format.py
# @E-mail  ：1076365758@qq.com
"""

from django.template import Library

# 将注册类实例化为register对象
register = Library()


@register.filter(name='sci_format')
def sci_format(value):
    return "%.2g" % value
