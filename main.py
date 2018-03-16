import sys

import id3
import utils

# Read args from cmd line
if len(sys.argv) != 5:
    print('Four arguments needed! Found: %s' % (len(sys.argv)-1))
    exit(1)
training_data_file = sys.argv[1]
validation_data_file = sys.argv[2]
test_data_file = sys.argv[3]
prune_factor = float(sys.argv[4])

print('Use training data from %s' % training_data_file)
print('Use validation data from %s' % validation_data_file)
print('Use training data from %s' % test_data_file)
print('Use prune factor: %s' % prune_factor)

print('')
training_data = utils.read_data(training_data_file)
validation_data = utils.read_data(validation_data_file)
test_data = utils.read_data(test_data_file)

root, node_num, leaf_num = id3.train(training_data)
utils.print_tree(root, training_data)

print('')
print('Pre-Pruned Accuracy')
print('- - - - - - - - - - - - -')
train_accuracy = id3.test(root, training_data) * 100
print('Number of training instances = %s' % len(training_data))
print('Number of training attributes = %s' % len(training_data[0].feature_map))
print('Total number of nodes in the tree = %s' % node_num)
print('Number of leaf nodes in the tree = %s' % leaf_num)
print('Accuracy of the model on the training dataset = %.1f%%' % train_accuracy)

validation_accuracy = id3.test(root, validation_data) * 100
print('')
print('Number of validation instances = %s' % len(validation_data))
print('Number of validation attributes = %s' % len(validation_data[0].feature_map))
print('Accuracy of the model on the validation dataset before pruning = %.1f%%' % validation_accuracy)

test_accuracy = id3.test(root, test_data) * 100
print('')
print('Number of validation instances = %s' % len(test_data))
print('Number of validation attributes = %s' % len(test_data[0].feature_map))
print('Accuracy of the model on the validation dataset before pruning = %.1f%%' % test_accuracy)

print('')
print('Post-Pruned Accuracy')
print('- - - - - - - - - - - - -')
prune_num = int(float(node_num) * prune_factor)

pruned_validation_accuracy, stop_indexes = id3.prune_and_test(root, node_num, prune_num, validation_accuracy / 100.0, validation_data, validation_accuracy)
node_num, leaf_num = id3.count_nodes_after_prune(root, stop_indexes)
pruned_training_accuracy  = id3.test(root, training_data, stop_indexes) * 100.0
print('Number of training instances = %s' % len(training_data))
print('Number of training attributes = %s' % len(training_data[0].feature_map))
print('Total number of nodes in the tree = %s' % node_num)
print('Number of leaf nodes in the tree = %s' % leaf_num)
print('Accuracy of the model on the training dataset = %.1f%%' % pruned_training_accuracy)

pruned_validation_accuracy *= 100.0
print('')
print('Number of validation instances = %s' % len(validation_data))
print('Number of validation attributes = %s' % len(validation_data[0].feature_map))
print('Accuracy of the model on the validation dataset after pruning = %.1f%%' % pruned_validation_accuracy)
if len(stop_indexes) == 0:
    print('Pruning did not improve validation accuracy!')

pruned_test_accuracy = id3.test(root, test_data, stop_indexes) * 100.0
print('')
print('Number of validation instances = %s' % len(test_data))
print('Number of validation attributes = %s' % len(test_data[0].feature_map))
print('Accuracy of the model on the test dataset after pruning = %.1f%%' % pruned_test_accuracy)