# -*- coding:utf-8 -*-
# created_date: 2019-02-19
# author: buxy
"""
最大最小值变换
"""

import numpy as np
import logging
from .preprocess_base import PreprocessTransformer


class MinMaxNorm(PreprocessTransformer):

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _fit(self):
        """
        最大最小值标准化
        通过数据每个特征的最大值，最小值，通过转换将所有数据标准化
        没有用到矩阵的运算，就不需要转换成matrix

        注意: 当出现唯一值变量时，minmax 会报错
        :return: None
        """
        self.min_values = self.X.min(0)
        self.max_values = self.X.max(0)
        self.ranges = self.max_values - self.min_values
        if (self.ranges == 0).any():
            self.logger.warning('特征中出现唯一值，相应特征值的range被调整成一个极小数')
            self.ranges[self.ranges == 0] = 1e-10

    def _transform(self):
        transformed_data = self.transform_data - np.tile(self.min_values, (self.n_transform_samples, 1))
        transformed_data = transformed_data/np.tile(self.ranges, (self.n_transform_samples, 1))
        return transformed_data
