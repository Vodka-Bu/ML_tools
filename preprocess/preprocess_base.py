# -*- coding:utf-8 -*-
# created_date: 2019-05-15
# author: buxy
"""
数据预处理基类
"""


import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """标准数据预处理用Transformer"""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.transform_data = None
        self.transform_y = None
        self.transformed_data = None
        self.n_transform_samples = None
        self.n_transform_features = None

    @staticmethod
    def to_array(data):
        """
        将输入数据转换成 np.array
        如果转换不成功则直接报错
        :param data: 数据，期望是 array-like, 但有可能不是
        """
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError('期望数据是 np.ndarray 格式: {0}'.format(e.__str__()))
        return data

    def format_data(self, data):
        """返回数据的n_sample，n_feature，并且当是一维向量时，进行转置"""
        shape = data.shape
        if len(shape) > 1:
            n_samples, n_features = shape
        else:
            n_samples = len(data)
            n_features = 1
            data = data.reshape(n_samples, n_features)
            self.logger.warning('X 可能是单变量，已经自动转置')
        return data, n_samples, n_features

    def fit(self, X, y=None):
        """
        所有transformer的通用环节(主要用于输入数据格式检查)
        :param X: array-like， n_samples * n_features， 如果是单变量，期望是 n * 1 的结构
        :param y: array-like， n_samples，可选 ,这里的y暂时限定只能是符合模型要求的数字，不可以是label
        """
        data = self.to_array(X)
        self.X, self.n_samples, self.n_features = self.format_data(data)
        if y is not None:
            self.y = self.to_array(y)
        else:
            self.y = None
        self._fit()
        return self

    def transform(self, X=None, y=None):
        """
        当 test_data 是 None 的时候，将 X 赋予 test_data
        """
        if X is None:
            self.transform_data = self.X
            self.n_transform_samples = self.n_samples
            self.n_transform_features = self.n_features
            self.transform_y = self.y
        else:
            # 需要判断一下数据格式是否准确
            transform_data = self.to_array(X)
            self.transform_data, self.n_transform_samples, self.n_transform_features = self.format_data(transform_data)
            assert self.n_features == self.n_transform_features, '测试数据与训练数据变量长度不一致'
        if y is not None:
            self.transform_y = self.to_array(y)
        self.transformed_data = self._transform()
        return self.transformed_data

    def _fit(self):
        self.logger.info('该对象还未配置_fit方法')

    def _transform(self):
        self.logger.info('该对象还未配置_transform方法')
        transformed_data = None
        return transformed_data
