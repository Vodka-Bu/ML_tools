# -*- coding:utf-8 -*-
# created_date: 2019-01-29
# author: buxy
"""
根据 sklearn 框架 创建基分类器，并引入一些自己的功能
直接创建基分类器必须要有的函数: fit， predict，用来处理一些通用的数据处理功能
在下层中主要用来 _fit 和 _predict 来编写函数，不需要考虑数据结构和类型不对的问题
"""

from .knn import KNN

__all__ = [
    'KNN'
]
