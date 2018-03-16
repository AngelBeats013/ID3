class TreeNode:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.node_index = 0
        self.data_indexes = list()
        self.left = None
        self.right = None
        self.pure = False
