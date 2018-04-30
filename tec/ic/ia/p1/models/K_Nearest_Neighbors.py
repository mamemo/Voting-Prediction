from tec.ic.ia.p1.models.Model import Model
import numpy as np
from collections import Counter


class KNearestNeighbors(Model):
	def __init__(self, samples_train, samples_test, prefix, k):
		super().__init__(samples_train, samples_test, prefix)
		self.k = k


	def create_kdtree(self, samples, dimension=0):
		samples_data = samples[0]
		samples_vote = samples[1]
		n = len(samples_data)
		if n <= 0:
			return None

		breaking_point = dimension % len(samples_data[0])
		ordered_samples, ordered_votes= (list(t) for t in zip(*sorted(zip(samples_data, samples_vote), key=lambda pair: pair[0][breaking_point])))
		half = round(n/2)
		return {
			'sample': ordered_samples[half],
			'vote': ordered_votes[half],
			'left_son': self.create_kdtree([ordered_samples[:half],ordered_votes[:half]], dimension + 1),
			'right_son': self.create_kdtree([ordered_samples[half + 1:],ordered_votes[half + 1:]], dimension + 1)
		}

	def calculate_manhattan_distance(self, p1, p2):
		return sum(abs(x - y) for x,y in zip(p1, p2))


	def kdtree_closest_point(self, root, point, depth=0):
		knn_best = []
		knn_distances = []
		if root is None:
			return None

		axis = depth % len(point)

		next_branch = None
		opposite_branch = None

		if point[axis] < root['sample'][axis]:
			next_branch = root['left_son']
			opposite_branch = root['right_son']
		else:
			next_branch = root['right_son']
			opposite_branch = root['left_son']

		root_distance = self.calculate_manhattan_distance(point,root['sample'])
		best_left = self.kdtree_closest_point(next_branch,point,depth + 1)
		if best_left == None:
			best_left = [[],[]]
		knn_best = best_left[0] + [root['vote']]
		knn_distances = best_left[1] + [root_distance]
		knn_distances,knn_best= (list(t) for t in zip(*sorted(zip(knn_distances,knn_best), key=lambda pair: pair[0])))
		best_right = []
		if abs(point[axis] - root['sample'][axis]) < knn_distances[:self.k][-1]:
			best_right = self.kdtree_closest_point(opposite_branch,point,depth + 1)
			if best_right == None:
				best_right = [[],[]]

			knn_best += best_right[0] 
			knn_distances += best_right[1] 
			knn_distances,knn_best= (list(t) for t in zip(*sorted(zip(knn_distances,knn_best), key=lambda pair: pair[0])))
		return [knn_best[:self.k], knn_distances[:self.k]]
 
	def execute(self):
		print("\n\n----------KNN kdtree-----------\n\n")

		print("Cantidad ejemplos training: ", len(self.samples_train[0]))
		print("Cantidad ejemplos testing: ", len(self.samples_test[0]),"\n\n") 

		tree = self.create_kdtree(self.samples_train)  
		posit = 0
		neg = 0
		for i in range(0,len(self.samples_test[0])):
			event = self.kdtree_closest_point(tree,self.samples_test[0][i])
			events = event[0]
			data = Counter(events)
			muestra = self.samples_test[1][i] 
			result = max(events, key=data.get)
			if muestra == result:
				posit+=1
			else:
				neg+=1
		print("positivo%: ",(posit*100)/len(self.samples_test[0]))
		print(posit)
		print(neg)
                
                

