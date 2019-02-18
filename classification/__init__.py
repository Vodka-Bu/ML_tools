# -*- coding:utf-8 -*-
#created_date: 2019-01-29
#author: buxy
'''
根据 sklearn 框架 创建基分类器，并引入一些自己的功能
直接创建基分类器必须要有的函数: fit， predict，同时做一些通用的数据处理方面的操作
'''

import numpy as np
import logging
from sklearn.base import BaseEstimator,ClassifierMixin

logger = logging.getLogger(__name__)


from .knn import KNN





__all__ = []


class ClassifierEstimator(BaseEstimator,ClassifierMixin):
    '''
    基分类器，同时会有一些新的功能
    '''

    @staticmethod
    def to_array(data):
        '''
        将输入数据转换成 np.array
        如果转换不成功则直接报错
        :param data: 数据，期望是 array-like, 但有可能不是
        '''
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise ValueError('期望数据是 np.ndarray 格式')
        return data



    def fit(self, X, y):
        '''
        所有基分类器的通用环节(主要用于输入数据格式检查)
        曾考虑要不要用装饰器的方法实现，感觉应该没有这种方式方便
        :param X: array-like， n_samples * n_features, 如果是单变量，期望是 n * 1 的结构
        :param y: array-like， n_samples

        注意这里不 return self，return 的动作在下一步做
        '''
        self.X = self.to_array(X)
        self.y = self.to_array(y)
        try:
            self.n_samples, self.n_features = self.X.shape
        except ValueError:
            ## 说明x是单变量
            self.n_samples = len(self.X)
            self.n_features = 1
            self.X = self.X.reshape(self.n_samples, self.n_features)
            logger.warning('X 可能是单变量，已经自动转置')



    def predict(self, test_data = None):
        '''
        当 test_data 是 None 的时候，将 X 赋予 test_data
        :param test_data: array-like, 会检查一下格式是否准确
        '''
        if test_data is None:
            self.test_data = self.X
        else:
            self.test_data = self.to_array(test_data)



