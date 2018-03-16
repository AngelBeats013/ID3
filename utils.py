from data_format import DataPoint


def read_data(file):
    '''
    Read data points from csv file, return a list of DataPoint
    :return: list of DataPoint
    '''
    from csv import DictReader

    data = []
    with open(file) as f:
        reader = DictReader(f)
        for row in reader:
            class_name = row['Class']
            del row['Class']
            dp = DataPoint(dict(row), class_name)
            data.append(dp)
    return data

def print_tree_helper(node, level, data):
    if node is None: return
    if node.pure:
        print(data[node.data_indexes[0]].class_name)
        return
    print('')
    print('| ' * level + node.feature_name + ' = 0 : ', end='')
    print_tree_helper(node.left, level+1, data)
    print('| ' * level + node.feature_name + ' = 1 : ', end='')
    print_tree_helper(node.right, level+1, data)


def print_tree(root, data):
    '''
    Print decision tree according to format defined in Assignment 2.pdf
    :param root: root node of decision tree
    '''
    print('Decision tree:', end='')
    print_tree_helper(root, 0, data)