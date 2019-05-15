# -*- coding:utf-8 -*-
# created_date: 2019-02-19
# author: buxy
"""
数据预处理工具，包含一些对象或者是一些函数
需要能够在 sklearn 中的 pipeline 内使用
"""

from .min_max_norm import MinMaxNorm

__all__ = [
    'MinMaxNorm'
]
