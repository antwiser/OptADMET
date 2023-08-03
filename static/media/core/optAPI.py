# -*- coding: utf-8 -*-
"""
ADMET was developed by Jiacai Yi et al. of CBDD GROUP of CSU China.
This project is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
Based on a work at http://home.scbdd.com.
Permissions beyond the scope of this license may be available at http://home.scbdd.com/. If you have any questions, please feel free to contact us.

# @Time    : 2020/10/2 上午10:04
# @Author  : Jiacai Yi
# @FileName: optAPI.py
# @E-mail  ：1076365758@qq.com
"""
import sys
import os
from django.conf import settings
import numpy as np
import pandas as pd

sys.path.append(os.path.join(settings.SITE_ROOT, 'static') + '/media/core')
# os.chdir(os.path.join(settings.SITE_ROOT, 'static') + '/media/core')

from .predictor import pred_admet


# 输入单个SMILES字符串
def singleSMILES(smiles, cache_bin_path, cache_csv_path, result_path):
    """.
    Parameters
    ----------
    smiles : string
    """
    smiles_list = [smiles]
    success_cnt, atom, medatom = pred_admet(smiles_list=smiles_list, cache_bin_path=cache_bin_path,
                                            cache_csv_path=cache_csv_path, result_path=result_path, gnn_name='rgcn')
    return success_cnt, atom, medatom


# 输入SMILES文件
def multiSMILES(data, cache_bin_path, cache_csv_path, result_path):
    success_cnt, result_df = pred_admet(smiles_list=data, cache_bin_path=cache_bin_path,
                             cache_csv_path=cache_csv_path, result_path=result_path, gnn_name='rgcn')
    return True, success_cnt, result_df
