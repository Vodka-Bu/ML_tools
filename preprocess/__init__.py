# -*- coding:utf-8 -*-
# created_date: 2019-02-19
# author: buxy
"""
数据预处理工具，包含一些对象或者是一些函数
需要能够在 sklearn 中的 pipeline 内使用
"""

import numpy as np
import logging
from sklearn.base import BaseEstimator,TransformerMixin




class PreprocessTransformer(BaseEstimator,TransformerMixin):
    '''标准数据预处理用Transformer'''
    logger = logging.getLogger(__name__)

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

    def fit(self, X, y = None):
        '''
        所有transformer的通用环节(主要用于输入数据格式检查)
        :param X: array-like， n_samples * n_features， 如果是单变量，期望是 n * 1 的结构
        :param y: array-like， n_samples，可选 ,这里的y暂时限定只能是符合模型要求的数字，不可以是label
        '''
        self.X = self.to_array(X)
        if y is not None:
            self.y = self.to_array(y)
        else:
            self.y = None

        try:
            self.n_samples, self.n_features = self.X.shape
        except ValueError:
            ## 说明x是单变量
            self.n_samples = len(self.X)
            self.n_features = 1
            self.X = self.X.reshape(self.n_samples, self.n_features)
            self.logger.warning('X 可能是单变量，已经自动转置')
        self._fit()
        return self


    def transform(self, X = None, y = None):
        '''
        当 test_data 是 None 的时候，将 X 赋予 test_data
        :param test_data: array-like, 会检查一下格式是否准确
        '''
        if X is None:
            self.transform_data = self.X
        else:
            # 需要判断一下数据格式是否准确
            self.transform_data = self.to_array(X)
            try:
                self.n_transform_samples, self.n_transform_features = self.transform_data.shape
            except ValueError:
                ## 说明 test_data 是单变量
                self.n_transform_samples = len(self.transform_data)
                self.n_transform_features = 1
                self.transform_data = self.transform_data.reshape(self.n_transform_samples, self.n_transform_features)
                self.logger.warning('X 可能是单变量，已经自动转置')
                if self.n_features != self.n_transform_features:
                    raise ValueError('数据特征数量与原数据集不一致')

        if y is not None:
            self.transform_y = self.to_array(y)
        else:
            self.transform_y = None

        self.transformed_data = self._transform()
        return self.transformed_data


    def _fit(self):
        self.logger.info('该对象还未配置_fit方法')


    def _transform(self):
        self.logger.info('该对象还未配置_transform方法')
        self.test_result = None


from .min_max_norm import MinMaxNorm

__all__ = [
    'PreprocessTransformer',
    'MinMaxNorm'
]