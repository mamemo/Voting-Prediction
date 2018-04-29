from tec.ic.ia.p1.models.Model import Model
import numpy as np

class KNearestNeighbors(Model):
    def __init__(self, samples_train, samples_test, prefix, k):
        super().__init__(samples_train, samples_test, prefix)
        self.k = k

    def execute(self):
        pass

    def create_kdtree(points, depth=0):
        n = len(points)

        if n <= 0:
            return None

        axis = depth % k

        sorted_points = sorted(points, key=lambda point: point[axis])

        return {
            'point': sorted_points[n / 2],
            'left': create_kdtree(sorted_points[:n / 2], depth + 1),
            'right': create_kdtree(sorted_points[n/2 + 1:], depth + 1)
        }

    def calculate_manhattan_distance(p1, p2):
        return sum(abs(x - y) for x,y in zip(p1, p2))

    def find_closest_samples(samples, new_sample, k):
        knn_points = []
        best_distances = []

        for sample in samples:
            sample_dist = distance(new_sample, sample)
             if len(knn_points) < 5:
                 knn_points.append(sample)
                 best_distances.append(sample_dist)             
             elif sample_dist < max(best_distances):
                 worst_point_index = best_distances.index(max(best_distances))
                 knn_points[worst_point_index] = sample
                 best_distances[worst_point_index] = sample_dist
        return knn_points


    k = 2



    kdtree = create_kdtree(samples)


    def closer_distance(pivot, p1, p2):
        if p1 is None:
            return p2

        if p2 is None:
            return p1

        d1 = calculate_manhattan_distance(pivot, p1)
        d2 = calculate_manhattan_distance(pivot, p2)

        if d1 < d2:
            return [p1,d1]
        else:
            return [p2,d2]


    def kdtree_closest_point(root, point, depth=0):
        knn_best = []
        knn_distances = []
        if root is None:
            return None

        axis = depth % k

        next_branch = None
        opposite_branch = None

        if point[axis] < root['point'][axis]:
            next_branch = root['left']
            opposite_branch = root['right']
        else:
            next_branch = root['right']
            opposite_branch = root['left']

        best = closer_distance(point,
                               kdtree_closest_point(next_branch,
                                                    point,
                                                    depth + 1),
                               root['point'])
            

        

        if calculate_manhattan_distance(point, best) > abs(point[axis] - root['point'][axis]):
            best = closer_distance(point,
                                   kdtree_closest_point(opposite_branch,
                                                        point,
                                                        depth + 1),
                                   best)
        if len(knn_best) < self.k:
            knn_best.append(best[0])
            knn_distances.append(best[1])             
        elif best[1] < max(knn_distances):
            worst_point_index = best_distances.index(max(knn_distances))
            knn_best[worst_point_index] = best[0]
            knn_distances[worst_point_index] = best[1]

        return knn_best
            
                
                

