# -*- coding:utf-8 -*-
# created_date: 2019-05-15
# author: buxy
"""
根据 sklearn 框架 创建基分类器，并引入一些自己的功能
直接创建基分类器必须要有的函数: fit， predict，用来处理一些通用的数据处理功能
在子类中主要用来 _fit 和 _predict 来编写函数，不需要考虑数据结构和类型不对的问题
"""

import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassifierEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self):
        """加入一个日志功能"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.test_data = None
        self.n_test_samples = None
        self.n_test_features = None
        self.test_result = None

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

    def fit(self, X, y):
        """
        所有基分类器的通用环节(主要用于输入数据格式检查)
        曾考虑要不要用装饰器的方法实现，感觉应该没有这种方式方便
        :param X: array-like， n_samples * n_features， 如果是单变量，期望是 n * 1 的结构
        :param y: array-like， n_samples，这里的y暂时限定只能是符合模型要求的数字，不可以是label
        """
        X = self.to_array(X)
        self.y = self.to_array(y)
        self.X, self.n_samples, self.n_features = self.format_data(X)
        self._fit()
        return self

    def predict(self, X=None):
        """
        当 X 是 None 的时候，将 训练数据 X 赋予 test_data
        :param X: array-like, 会检查一下格式是否准确
        """
        if X is None:
            self.test_data = self.X
            self.n_test_samples = self.n_samples
            self.n_test_features = self.n_features
        else:
            test_data = self.to_array(X)
            self.test_data, self.n_test_samples, self.n_test_features = self.format_data(test_data)
            assert self.n_features == self.n_test_features, '测试数据与训练数据变量长度不一致'
        self.test_result = self._predict()
        return self.test_result

    def _fit(self):
        self.logger.info('该对象还未配置_fit方法')

    def _predict(self):
        self.logger.info('该对象还未配置_predict方法')
        self.test_result = None
        return None