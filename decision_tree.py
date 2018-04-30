import numpy as np
import math

def uncertainty(S):
	numPos, numNeg = 0, 0
	total = np.size(S, 0)
	for row in S:
		if row[0] == 1:
			numPos += 1
		elif row[0] == -1:
			numNeg += 1
	
	#print("numneg" + str(numNeg))
	#print("numpos" + str(numPos))
	#print("total" + str(total))
	#print("neg/total" + str(float(numNeg)/total))
	#print("pos/total" + str(float(numPos)/total))
	

	if float(numPos)/total == 0:
		p = -float(numNeg/total)*math.log(float(numNeg)/total, 2)
	elif float(numNeg)/total == 0:
		p = -(float(numPos/total))*math.log(float(numPos)/total, 2)
	else:
		p = -(float(numPos)/total)*math.log(float(numPos)/total, 2) - float(numNeg)/total*math.log(float(numNeg)/total, 2)
	#print("p is: " + str(p))
	return p


def info_gain(S, S1, S2):
	numInS1 = np.size(S1, 0)
	numInS2 = np.size(S2, 0)
	total = numInS1 + numInS2
	#print("in infogain")
	#print(numInS1)
	#print(numInS2)
	#print(total)

	if numInS1 == 0:
		return uncertainty(S) - (float(numInS2)/total)*uncertainty(S2)
	elif numInS2 == 0:
		return uncertainty(S) - (float(numInS1)/total)*uncertainty(S1)
	else:
		return uncertainty(S) - (float(numInS1)/total)*uncertainty(S1)- (float(numInS2)/total)*uncertainty(S2)


#index is an attribute, value is the value on which to split. Used in find_best_split
def test_split(index, value, data):
	left, right = list(), list()
	for row in data:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

#tests splitting on each attribute for each of its values. uses info_gain to find max info gain and returns
#best split.
def find_best_split(data):
	class_values = list(set(row[-1] for row in data))
	b_index, b_value, b_score, b_groups = 999, 999, -1, None
	for index in range( 1, len(data[0])):
		for row in data:
			groups = test_split(index, row[index], data)
			left, right = test_split(index, row[index], data)
			infoGain = info_gain(data, left, right)
			print('X%d < %.3f InfoGain=%.3f\n' % ((index), row[index], infoGain))
			if infoGain > b_score:
				b_index, b_value, b_score, b_groups = index, row[index], infoGain, groups
	print("index: " + str(b_index) + "value: " + str(b_value))
	return {'index': b_index, 'value': b_value, 'groups': b_groups}

def normalize(X):
    for i in range(np.size(X, 1)):
        X[:, i] = X[:, i] / np.matrix.max(np.matrix(X[:, i]))

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[0] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = find_best_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = find_best_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = find_best_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

data = np.genfromtxt('knn_train.csv', delimiter=',')
#normalize(data[:, 1:np.size(data, 1)])
#data now properly defined
'''
data = [[-1,2.771244718,1.784783929],
	[-1,1.728571309,1.169761413],
	[-1,3.678319846,2.81281357],
	[-1,3.961043357,2.61995032],
	[-1,2.999208922,2.209014212],
	[1,7.497545867,3.162953546],
	[1,9.00220326,3.339047188],
	[1,7.444542326,0.476683375],
	[1,10.12493903,3.234550982],
	[1,6.642287351,3.319983761]]
'''
#split = find_best_split(data)
#print('Split: [X%d < %.3f]' % ((split['index']), split['value']))
tree = build_tree(data, 1, 1)
print_tree(tree)

numErrors = 0
for row in data:
	prediction = predict(tree, row)
	if row[0] * prediction < 0:
		numErrors+=1
	print('Expected=%d, Got=%d' % (row[0], prediction))

print("Number of errors: " + str(numErrors))
