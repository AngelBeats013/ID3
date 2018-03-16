import random
from math import log2

from tree import TreeNode


def train(data):
    '''
    Train a decision tree using id3 algorithm
    :param data: list of DataPoint for training. Assume all DataPoint has same features and binary values for feature
                 and class
    :return: root node of decision tree
    '''
    if len(data) == 0:
        return None
    feature_list = list(data[0].feature_map.keys())

    root = TreeNode('')
    for index in range(len(data)):
        root.data_indexes.append(index)
    parent_entropy = calc_entropy(root, data)

    iter_level_nodes = [] # nodes on current iteration depth level
    iter_level_nodes.append(root)

    for _ in range(len(feature_list)):
        if len(iter_level_nodes) == 0: # No node for calculation. All leaf nodes are pure
            break
        max_info_gain = 0.0
        best_feature = ''
        # for each feature, calculate info_gain
        for feature in feature_list:
            child_entropy = 0.0
            for node in iter_level_nodes:
                construct_child_node(node, feature, data)
                child_entropy += calc_child_entropy(node, data)
            if parent_entropy - child_entropy > max_info_gain:
                max_info_gain = parent_entropy - child_entropy
                best_feature = feature
        next_level_nodes = []
        for node in iter_level_nodes:
            construct_child_node(node, best_feature, data)
            if node.left:
                next_level_nodes.append(node.left)
            if node.right:
                next_level_nodes.append(node.right)
        iter_level_nodes = next_level_nodes
        feature_list.remove(best_feature)
        parent_entropy -= max_info_gain
    node_num, leaf_num = label_tree(root)
    return root, node_num, leaf_num


def test(root, data, stop_index=set()):
    '''
    Test with given data and return accuracy
    :param root: root of decision tree
    :param data: list of DataPoint for testing
    :param stop_index: set of node indexes that algorithm stop going deep if reach any. Default is empty (no stopping)
    :return: test accuracy on data set
    '''
    correct = 0
    current_node = root
    for dp in data:
        while current_node.left or current_node.right:
            if current_node.node_index in stop_index:
                break
            if dp.feature_map[current_node.feature_name] == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
        if current_node.pure:
            # If leaf node is pure, just use class of first data
            result = data[current_node.data_indexes[0]].class_name
        else:
            # If not, get majority of class name as result
            num_class_0 = 0
            for index in current_node.data_indexes:
                num_class_0 += 1 if data[index].class_name == '0' else 0
            result = '0' if num_class_0 > len(current_node.data_indexes)/2 else '1'
        if result == dp.class_name:
            correct += 1
        current_node = root
    return float(correct) / float(len(data))


def construct_child_node(node, feature, data):
    '''
    Construct child nodes for a node. If this node is pure, nothing will change
    :param node: node to construct children
    :param feature: feature used to sort data for children
    :param data: data set
    '''
    if node.pure:
        return
    node.feature_name = feature
    left_node = TreeNode('')
    right_node = TreeNode('')
    for index in node.data_indexes:
        if data[index].feature_map[feature] == '0':
            left_node.data_indexes.append(index)
        else:
            right_node.data_indexes.append(index)
    if len(left_node.data_indexes) == 0 or len(right_node.data_indexes) == 0:
        # If no data in left or right node, then this node is pure, just return
        node.pure = True
        return
    left_node.pure = True
    right_node.pure = True
    left_class_name = data[left_node.data_indexes[0]].class_name
    right_class_name = data[right_node.data_indexes[0]].class_name
    for index in left_node.data_indexes:
        if data[index].class_name != left_class_name:
            left_node.pure = False
            break
    for index in right_node.data_indexes:
        if data[index].class_name != right_class_name:
            right_node.pure = False
            break
    node.left = left_node
    node.right = right_node


def calc_child_entropy(node, data):
    '''
    Calculate entropy of children of a node
    :param node: node to calculate children entropy
    :param data: data set
    :return: entropy of children nodes
    '''
    child_entropy = 0.0
    if node.left:
        child_entropy += float(len(node.left.data_indexes)) / float(len(data)) * calc_entropy(node.left, data)
    if node.right:
        child_entropy += float(len(node.right.data_indexes)) / float(len(data)) * calc_entropy(node.right, data)
    return child_entropy


def calc_entropy(node, data):
    '''
    Calculate entropy of the node's data
    :param node: node to calculate entropy
    :param data: data set
    :return: entropy of the node's data
    '''
    num_class_0 = 0
    for index in node.data_indexes:
        num_class_0 += 1 if data[index].class_name == '0' else 0
    p_class_0 = float(num_class_0) / float(len(node.data_indexes))
    p_class_1 = 1.0 - p_class_0
    if p_class_0 == 0 or p_class_1 == 0:
        node.pure = True
        return 0.0
    return 0 - p_class_0 * log2(p_class_0) - p_class_1 * log2(p_class_1)


def label_tree(root):
    '''
    Label each node with index
    :param root: root of the tree
    :return: node count and leaf count
    '''
    index = 0
    node = root
    nodes = [node]
    leaf = 0
    while len(nodes) > 0:
        n = nodes.pop(0)
        n.node_index = index
        index += 1
        if n.left or n.right:
            nodes.append(n.left)
            nodes.append(n.right)
        else:
            leaf += 1
    return index, leaf


def prune_and_test(root, node_num, prune_num, base_accuracy, data, max_iter=50):
    '''
    Randomly choose prune_num nodes to prune and test for accuracy. If cannot improve accuracy after max_iter iterations,
    function will return original accuracy and empty pruning nodes set
    :param root: root of decision tree
    :param node_num: total number of nodes
    :param prune_num: number of nodes to prune
    :param base_accuracy: base accuracy to compare with
    :param data: data set
    :param max_iter: maximum iterations
    :return: accuracy after pruning and set of pruning node indexes
    '''
    stop_indexes= set()
    while max_iter > 0:
        stop_indexes.clear()
        while len(stop_indexes) < prune_num :
            prune_index = random.randint(0, node_num)
            if prune_index not in stop_indexes:
                stop_indexes.add(prune_index)
        prune_accuracy = test(root, data, stop_indexes)
        if prune_accuracy - base_accuracy > 0.01:
            return prune_accuracy, stop_indexes
        max_iter -= 1
    return base_accuracy, set()


def count_nodes_after_prune(root, stop_indexes):
    '''
    Count nodes and leaves after pruning
    :param root: root of tree
    :param stop_indexes: set of pruned nodes
    :return: node count and leaf count
    '''
    index = 0
    node = root
    nodes = [node]
    leaf = 0
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n.node_index in stop_indexes:
            index += 1
            leaf += 1
            continue
        index += 1
        if n.left or n.right:
            nodes.append(n.left)
            nodes.append(n.right)
        else:
            leaf += 1
    return index, leaf
