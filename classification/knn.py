# -*- coding:utf-8 -*-
#created_date: 2019-02-18
#author: buxy
'''
knn 学习器


'''

import logging
import numpy as np
from . import ClassifierEstimator

logger = logging.getLogger(__name__)

class KNN(ClassifierEstimator):
    def __init__(self, n, dist_func = 'euclid'):

        self.n = n
        if dist_func == 'euclid':
            self.dist_func = self.euclid_dist
        else:
            assert hasattr(dist_func, '__call__')
            self.dist_func = dist_func
        pass


    @staticmethod
    def euclid_dist(x1, x2):
        '''
        欧几里得距离
        :param x1: 一维向量, 1 * n 的格式
        :param x2: 一维向量, 1 * n 的格式，内部进行转置
        :return: float dist 距离
        '''
        x1_mat = np.mat(x1)
        x2_mat = np.mat(x2)
        delta = x1_mat - x2_mat
        dist = np.sqrt(np.dot(delta, delta.T))
        return dist[0,0]