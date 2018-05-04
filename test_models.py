from tec.ic.ia.p1.models.Decision_Tree import DecisionTree, DTree

'''
[TREE] An object of the DTree class is created and its attributes are checked.
Parameters: None
Requirements: None
Outputs: Initialized attributes according to the type.
'''
def test_new_tree():
    tree = DTree()
    assert tree.leaf_nodes == []
    assert tree.nodes_conditions == []
    assert tree.attribute == None

'''
[TREE] Check the output of the function that calculates the gain to an attribute according to a set of data.
Parameters: None
Requirements: Declare a data set (samples_train) and enter the attribute (0) that will be evaluated.
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: result gain float number.
'''
def test_gain():
	samples_train = [[["some"],["full" ],["some" ],["full" ],["full" ],["some" ],
			         ["none" ],["some" ],["full" ],["full" ],["none" ],["full" ]],
			         ["yes", "no" , "yes", "yes", "no" , "yes", "no" , "yes", "no" , "no" , "no" , "yes"]]

	decisionTree = DecisionTree(samples_train, [], "", 0)
	gain = decisionTree.gain(0,[0,1,2,3,4,5,6,7,8,9,10,11])
	print(gain)
	assert gain == 0.5408520829727552

'''
[TREE] Check that the function take_out_information according to a set of given data 
and a specific attribute, takes the values of the children of the attribute, 
the quantity for each one and the outputs.
Parameters: None
Requirements: Declare a data set (samples_train) and enter the attribute (0) that will be evaluated.
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: - attributes_list: list with string values. Contains the attribute values.
         - attributes_values: list with int values. Contains the amount of attributes values.
         - outputs_list: list with string values. Contains the outputs values.
         - outputs_values:  list with int values. Contains the amount of outputs values.
         - total: list with sublist int values. Contains the amount of outputs values per attribute value.
'''
def test_take_out_information():
	samples_train = [[["some"],["full" ],["some" ],["full" ],["full" ],["some" ],
			         ["none" ],["some" ],["full" ],["full" ],["none" ],["full" ]],
			         ["yes", "no" , "yes", "yes", "no" , "yes", "no" , "yes", "no" , "no" , "no" , "yes"]]

	decisionTree = DecisionTree(samples_train, [], "", 0)
	decisionTree.take_out_information(0,[0,1,2,3,4,5,6,7,8,9,10,11])
	assert decisionTree.attributes_list == ['some', 'full', 'none']
	assert decisionTree.attributes_values == [4, 6, 2]
	assert decisionTree.total == [[4, 0], [2, 4], [0, 2]]
	assert decisionTree.outputs_list == ['yes', 'no']
	assert decisionTree.outputs_values == [6, 6]

'''
[TREE] Check the output of the general_entropy function which returns the entropy of the parent attribute.
Parameters: None
Requirements: Declare the list that contains the amount of output values
(outputs_values) and the total amount of outputs (N).
If the outputs are ["PAC", "RN","BLANCOS","NULOS"] the list will be [5,4,7,8] depending on the data,
and N will be 5+4+7+8.
Outputs: numeric value of the entropy of the attribute.
'''
def test_general_entropy():
	decisionTree = DecisionTree([], [], "", 0)
	decisionTree.outputs_values = [6, 6]
	decisionTree.N = 12
	decisionTree.general_entropy()
	assert decisionTree.attribute_entropy == 1

'''
[TREE] Check the output of the total_gain function which returns the gain of the parent attribute.
Parameters: None
Requirements: 
	- attributes_values: list with int values. Contains the amount of attributes values.
	- outputs_values:  list with int values. Contains the amount of outputs values.
	- total: list with sublist int values. Contains the amount of outputs values per attribute value.
	- N: len(examples).
	- attribute_entropy: entropy of the parent attribute.
Outputs: numeric value of the gain of the attribute.
'''
def test_total_gain():
	decisionTree = DecisionTree([], [], "", 0)
	decisionTree.attributes_values = [4, 6, 2]
	decisionTree.outputs_values = [6, 6]
	decisionTree.total = [[4, 0], [2, 4], [0, 2]]
	decisionTree.N = 12
	decisionTree.attribute_entropy = 1
	assert decisionTree.total_gain() == 0.5408520829727552

'''
[TREE] Check that the function generate_ranges converts the numerical values of a list 
to a String of the range where the number is found.
Parameters: None
Requirements: Declare a data set (samples_train).
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: List without continuous values.
'''
def test_generate_ranges():
	samples_train = [[[0.2145],[0.4987],[0.6445],[0.858]],[]]
	decisionTree = DecisionTree(samples_train, [], "", 0)
	decisionTree.generate_ranges()
	assert decisionTree.samples_train == [[["[0 , 0.25["],["[0.25 , 0.50["],["[0.50 , 0.75["],["[0.75 , 1]"]],[]]

'''
[TREE] Check that the decision_tree_learning function returns a tree according to a set of data.
Parameters: None
Requirements: Declare a data set (samples_train).
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: 
	- tree.nodes_conditions: list with String values. Contains the attributes values.
	- tree.leaf_nodes:  list with String values. Contains the outputs values.
	- tree.attribute: int value. The attribute that evaluates. 
'''
def test_decision_tree_learning():
	samples_train = [[["some"],["full" ],["some" ],["full" ],["full" ],["some" ],
			         ["none" ],["some" ],["full" ],["full" ],["none" ],["full" ]],
			         ["yes", "no" , "yes", "yes", "no" , "yes", "no" , "yes", "no" , "no" , "no" , "yes"]]

	decisionTree = DecisionTree(samples_train, [], "", 0)
	tree = decisionTree.decision_tree_learning([0,1,2,3,4,5,6,7,8,9,10,11],[0],[0,1,2,3,4,5,6,7,8,9,10,11])
	assert  tree.leaf_nodes == ['yes', 'no', 'no']
	assert  tree.nodes_conditions == ['some', 'full', 'none']
	assert  tree.attribute == 0

'''
[TREE] Check that the classification function returns False when the outputs of the samples_train are not the same.
Parameters: None
Requirements: Declare a data set (samples_train).
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: Boolean value.
'''
def test_false_classification():
	samples_train = [[["some"],["full" ],["some" ]],
			         ["yes", "no" , "yes"]]
	decisionTree = DecisionTree(samples_train, [], "", 0)
	assert decisionTree.classification([0,1,2]) == False

'''
[TREE] Check that the classification function returns True when the outputs of the samples_train are the same.
Parameters: None
Requirements: Declare a data set (samples_train).
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: Boolean value.
'''
def test_true_classification():
	samples_train = [[["some"],["full" ],["some" ]],
			         ["yes", "yes" , "yes"]]
	decisionTree = DecisionTree(samples_train, [], "", 0)
	assert decisionTree.classification([0,1,2]) == True

'''
[TREE] Check that the plurality value function returns the input value with the highest number of occurrences in the list.
Parameters: None
Requirements: Declare a data set (samples_train).
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: String value.
'''
def test_plurality_value():
	samples_train = [[["some"],["full" ],["some"]],
			         ["yes", "no" , "yes"]]
	decisionTree = DecisionTree(samples_train, [], "", 0)
	assert decisionTree.plurality_value([0,1,2]) == "yes"

'''
[TREE] Check that the function total_deviation returns the deviation of an attribute 
(node of the tree), evaluating in the table of chi squared.
Parameters: None
Requirements: Declare a data set (samples_train).
It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
Outputs: float value between 0-1.
'''
def test_total_deviation():
	samples_train = [[["some"],["full" ],["some" ],["full" ],["full" ],["some" ],
			         ["none" ],["some" ],["full" ],["full" ],["none" ],["full" ]],
			         ["yes", "no" , "yes", "yes", "no" , "yes", "no" , "yes", "no" , "no" , "no" , "yes"]]

	decisionTree = DecisionTree(samples_train, [], "", 0)
	result = decisionTree.total_deviation(['full', 'none', 'some'],[[1,3,4,8,9,11],[6,10],[0,2,5,7]])
	assert result == 0.03567399334725241

'''
[TREE] Verify that the threshold_pruning_tree function does not prune the tree if the 
threshold is higher than the entropy of the nodes.
Parameters: None
Requirements: 
	- Declare a data set (samples_train) and initialize it.
	  It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
	- Declare a DTree and initialize it. 
	- Parameter examples: list of int values that corresponds to the index of the values of samples_train[0].
	- Threshold: in this case is 0.25.
Outputs: DTree value intact.
'''
def test_high_threshold_pruning_tree():
	samples_train = [[["some"],["full" ],["some" ],["full" ],["full" ],["some" ],
			         ["none" ],["some" ],["full" ],["full" ],["none" ],["full" ]],
			         ["yes", "no" , "yes", "yes", "no" , "yes", "no" , "yes", "no" , "no" , "no" , "yes"]]
	decisionTree = DecisionTree(samples_train, [], "", 0.25)
	tree = DTree()
	tree.leaf_nodes = ['yes', 'no', 'no']
	tree.nodes_conditions = ['some', 'full', 'none']
	tree.attribute = 0
	result = decisionTree.pruning_tree([0,1,2,3,4,5,6,7,8,9,10,11],tree)
	assert result.attribute == tree.attribute
	assert result.leaf_nodes == tree.leaf_nodes
	assert result.nodes_conditions == tree.nodes_conditions

'''
[TREE] Verify that the threshold_pruning_tree function does prune the tree if the 
threshold is lower than the entropy of the nodes.
Parameters: None
Requirements: 
	- Declare a data set (samples_train) and initialize it.
	  It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
	- Declare a DTree and initialize it. 
	- Parameter examples: list of int values that corresponds to the index of the values of samples_train[0].
	- Threshold: in this case is 0.02.
Outputs: Prune DTree. In this case is a String value.
'''
def test_low_threshold_pruning_tree():
	samples_train = [[["some"],["full" ],["some" ],["full" ],["full" ],["some" ],
			         ["none" ],["some" ],["full" ],["full" ],["none" ],["full" ]],
			         ["yes", "yes" , "yes", "yes", "no" , "yes", "no" , "yes", "no" , "no" , "no" , "yes"]]
	decisionTree = DecisionTree(samples_train, [], "", 0.02)
	tree = DTree()
	tree.leaf_nodes = ['yes', 'no', 'no']
	tree.nodes_conditions = ['some', 'full', 'none']
	tree.attribute = 0
	result = decisionTree.pruning_tree([0,1,2,3,4,5,6,7,8,9,10,11],tree)
	assert result == "yes"

'''
[TREE] Verify that the validate_data function use the created tree and returns 
the classification of the data.
Parameters: None
Requirements: 
	- Declare a data set (samples_train) and initialize it.
	  It must have the form [ [[attributes values], ... ,[attributes values]] , [outputs]]
	- Declare a DTree and initialize it. 
	- votes: List with sublists of options that will make the conversion of String outputs to int outputs.
	- prediction: index of the "votes" sublist that will be used.
Outputs: List with int values. Classification of the data list.
'''
def test_validate_data():
	samples_train = [[["some"],["full" ],["none" ],["full" ]],
			         ["yes", "no" , "no", "no"]]
	decisionTree = DecisionTree([], [], "", 0)
	tree = DTree()
	tree.leaf_nodes = ['yes', 'no', 'no']
	tree.nodes_conditions = ['some', 'full', 'none']
	tree.attribute = 0
	decisionTree.main_tree = tree
	decisionTree.votes = [["yes","no"]]
	decisionTree.prediction = 0
	result = decisionTree.validate_data(samples_train)
	assert result == [0,1,1,1]