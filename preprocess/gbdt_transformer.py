# -*- coding: utf-8 -*-
# create_date: 2019-06-29
# author: buxy
"""
利用gbdt训练数据, 返回apply(X)
可以观察每棵树所有叶子节点的路径
一般会和 onehot 一起使用，后接各种模型
仍然具备直接 predict 的功能
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import TransformerMixin


class GBClassifier(GradientBoostingClassifier, TransformerMixin):

    @staticmethod
    def describe_leave_path(tree, feature_names=None):
        """
        :param tree: 树 sklearn.tree._tree.Tree
        :param feature_names: 真实的变量名
        :return: list，每个叶子节点的路径
        """
        # 确定树的各个属性
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature.copy()
        if feature_names is not None:
            feature = [feature_names[i] for i in feature]
        else:
            feature = ['X[:, {0}]'.format(i) for i in feature]
        threshold = tree.threshold.copy()
        threshold = np.round(threshold, 4)
        node_stack = [(0, 'root')]
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        nodes_desc = [str(i) for i in range(n_nodes)]
        while len(node_stack) > 0:
            node_id, node_desc = node_stack.pop()
            nodes_desc[node_id] = node_desc
            if children_left[node_id] == children_right[node_id]:
                # 当两个相等，说明都是-1，该节点为叶子节点
                is_leaves[node_id] = True
                nodes_desc[node_id] = nodes_desc[node_id] + ' -> leave_node:{0}'.format(node_id)
                continue
            if children_left[node_id] != -1:
                left_node_desc_part = '{0} <= {1}'.format(feature[node_id], threshold[node_id])
                node_stack.append(
                    (children_left[node_id], '{0} -> {1}'.format(node_desc, left_node_desc_part)))
            if children_right[node_id] != -1:
                right_node_desc_part = '{0} > {1}'.format(feature[node_id], threshold[node_id])
                node_stack.append(
                    (children_right[node_id], '{0} -> {1}'.format(node_desc, right_node_desc_part)))
        return [nodes_desc[i] for i in range(n_nodes) if is_leaves[i]]

    def describe_trees(self, feature_names=None):
        all_leaves_path = []
        for estimator in self.estimators_.reshape(self.n_estimators_):
            tree_ = estimator.tree_
            all_leaves_path = all_leaves_path + self.describe_leave_path(tree_, feature_names)
        return all_leaves_path

    def transform(self, X):
        n_samples = len(X)
        return self.apply(X).reshape(n_samples, self.n_estimators_)
