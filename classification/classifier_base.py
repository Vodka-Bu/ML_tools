# -*- coding:utf-8 -*-
# created_date: 2019-05-15
# author: buxy
"""
根据 sklearn 框架 创建基分类器
直接创建基分类器必须要有的: fit、predict
在子类中主要编写_fit,_predict, 有关X y的各种基本性质初始化已经做了，不需要再关心了
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassifierEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.n_samples = None
        self.n_features = None
        self.feature_names = None
        self.test_data = None
        self.n_test_samples = None
        self.n_test_features = None

    @staticmethod
    def to_array(data):
        """
        将输入数据转换成 np.array
        如果转换不成功则报错
        :param data: 数据，期望是 array-like, 但有可能不是
        """
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
            except Exception as e:
                raise ValueError('期望数据是 np.ndarray 格式: {0}'.format(e.__str__()))
        return data

    @staticmethod
    def format_data(data):
        """
        格式化数据，转换成n_samples * n_features 的格式，如果不是则强行转
        :param data: 特征
        :return: 特征, n_samples, n_features
        """
        shape = data.shape
        if len(shape) > 1:
            n_samples, n_features = shape
        else:
            n_samples = len(data)
            n_features = 1
            data = data.reshape(n_samples, n_features)
        return data, n_samples, n_features

    def fit(self, X, y, feature_names=None):
        """
        :param X: array-like, n_samples * n_features， 如果是单变量，期望是 n * 1 的结构
        :param y: array-like, n_samples，这里的y暂时限定只能是符合模型要求的数字，不可以是label
        :param feature_names: list n_features 可以为None
        """
        X = self.to_array(X)
        self.y = self.to_array(y)
        self.X, self.n_samples, self.n_features = self.format_data(X)
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = list(range(self.n_features))
        else:
            assert len(feature_names) == self.n_features, 'feature_names和n_features长度不一致'
            self.feature_names = feature_names
        self._fit()
        return self

    def predict(self, X):
        """
        :param X: array-like, 会检查一下格式是否准确
        """
        test_data = self.to_array(X)
        self.test_data, self.n_test_samples, self.n_test_features = self.format_data(test_data)
        assert self.n_features == self.n_test_features, '测试数据与训练数据变量长度不一致'
        return self._predict()

    def _fit(self):
        raise NotImplementedError('还未完成_fit')

    def _predict(self):
        raise NotImplementedError('还未完成_predict')
