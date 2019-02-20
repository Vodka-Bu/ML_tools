# -*- coding:utf-8 -*-
#created_date: 2019-01-29
#author: buxy
'''
根据 sklearn 框架 创建基分类器，并引入一些自己的功能
直接创建基分类器必须要有的函数: fit， predict，用来处理一些通用的数据处理功能
在下层中主要用来 _fit 和 _predict 来编写函数，不需要考虑数据结构和类型不对的问题
'''

import numpy as np
import logging
from sklearn.base import BaseEstimator,ClassifierMixin

logger = logging.getLogger(__name__)






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
        :param X: array-like， n_samples * n_features， 如果是单变量，期望是 n * 1 的结构
        :param y: array-like， n_samples，这里的y暂时限定只能是符合模型要求的数字，不可以是label
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
        self._fit()
        return self


    def predict(self, X = None):
        '''
        当 test_data 是 None 的时候，将 X 赋予 test_data
        :param test_data: array-like, 会检查一下格式是否准确
        '''
        if X is None:
            self.test_data = self.X
        else:
            # 需要判断一下数据格式是否准确
            self.test_data = self.to_array(X)
            try:
                self.n_test_samples, self.n_test_features = self.test_data.shape
            except ValueError:
                ## 说明 test_data 是单变量
                self.n_test_samples = len(self.test_data)
                self.n_test_features = 1
                self.test_data = self.test_data.reshape(self.n_test_samples, self.n_test_features)
                logger.warning('X 可能是单变量，已经自动转置')
                if self.n_features != self.n_test_features:
                    raise ValueError('测试数据特征数量与训练数据集不一致')
        self.test_result = self._predict()
        return self.test_result


    def _fit(self):
        logger.info('该对象还未配置_fit方法')


    def _predict(self):
        logger.info('该对象还未配置_predict方法')
        self.test_result = None



from .knn import KNN


__all__ = [
    'ClassifierEstimator',
    'KNN'
]