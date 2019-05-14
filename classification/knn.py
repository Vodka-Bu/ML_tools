# -*- coding:utf-8 -*-
#created_date: 2019-02-18
#author: buxy
'''
knn 学习器
'''

import logging
import numpy as np
from . import ClassifierEstimator



class KNN(ClassifierEstimator):
    '''
    自制KNN
    可以自定义距离函数，距离函数的传入参数是 x，X，输出是对应所有点的距离
    在考虑要不要加上根据y各类占比，选择最终结果的方案（当各类别非常不均衡时，在一个区域小概率出现的概率很可能小于 50%），但这个就不是KNN的定义了。
    关于排序： KNN 可以不需要全排序就停止
    '''

    def __init__(self, k, dist_func = None):
        self.k = k
        if dist_func is None:
            self.dist_func = self.euclid_dist
        else:
            assert hasattr(dist_func, '__call__')
            self.dist_func = dist_func



    @staticmethod
    def euclid_dist(x, X):
        '''
        欧几里得距离
        :param x: 一维向量, 1 * n 的格式
        :param X: 二维矩阵, m * n 的格式
        :return: 一维向量, array_like, 长度为m，也就是 x 到 m 个点的距离
        '''
        m = X.shape[0]
        diff_mat = np.tile(x, (m, 1)) - X
        # 对上式结果平方
        sq_diff_mat = np.square(diff_mat)
        # 对上式结果按照列加总（横向加总）
        sq_distances = sq_diff_mat.sum(axis=1)
        # 开根号后得到距离所有点的距离
        distances = np.sqrt(sq_distances)
        return np.array(distances).reshape(m)


    @staticmethod
    def argsort_part(distances, k):
        '''
        部分排序，并返回相应index
        :param distances: 距离
        :param k: 停止轮数
        :return: 长度为k的一个list
        '''
        distinces_copy = distances.copy()
        max_dist = max(distinces_copy)

        arg_list = []
        for i in range(k):
            temp_min_value = max_dist
            temp_min_index = -1
            for index, value in enumerate(distinces_copy):
                if value <= temp_min_value:
                    temp_min_value = value
                    temp_min_index = index
            arg_list.append(temp_min_index)
            distinces_copy[temp_min_index] = max_dist + 1
        return arg_list



    def _fit(self):
        '''除了数据格式化以外，knn 不存在 _fit 步骤'''
        self.X_mat = np.mat(self.X)



    def _predict(self):
        '''
        knn 预测函数
        因为knn 的预测是单个样本独立预测，无法通过矩阵的形式统一处理，因此采用单循环的方式
        :return: 最终结果
        '''

        self.test_result_list = []
        for i in range(self.n_test_samples):
            self.test_result_list.append(self.__predict(self.test_data[i, :]))
        return self.to_array(self.test_result_list)



    def __predict(self, x):
        '''
        knn 预测函数
        不需要全排序
        :return 预测结果，也就是对应的最多的 y
        '''
        x_mat = np.mat(x)
        distances = self.dist_func(x_mat, self.X_mat)
        self.distances = distances
        sorted_k_indicies = self.argsort_part(distances,self.k)
        class_count = {}
        for index in sorted_k_indicies:
            label = self.y[index]
            class_count[label] = class_count.get(label,0) + 1
        temp_value = 0
        temp_class = None
        for key, value in class_count.items():
            if value > temp_value:
                temp_class = key
                temp_value = value
        return temp_class