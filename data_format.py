class DataPoint:
    feature_map = dict()
    class_name = '0'

    def __init__(self, feature_map, class_name):
        self.feature_map = feature_map
        self.class_name = class_name