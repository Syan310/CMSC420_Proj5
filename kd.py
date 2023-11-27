from __future__ import annotations
import json
import math
from queue import PriorityQueue
from typing import List

# Datum class.
# DO NOT MODIFY.
class Datum():
    def __init__(self,
                 coords : tuple[int],
                 code   : str):
        self.coords = coords
        self.code   = code
    def to_json(self) -> str:
        dict_repr = {'code':self.code,'coords':self.coords}
        return(dict_repr)

# Internal node class.
# DO NOT MODIFY.
class NodeInternal():
    def  __init__(self,
                  splitindex : int,
                  splitvalue : float,
                  leftchild,
                  rightchild):
        self.splitindex = splitindex
        self.splitvalue = splitvalue
        self.leftchild  = leftchild
        self.rightchild = rightchild

# Leaf node class.
# DO NOT MODIFY.
class NodeLeaf():
    def  __init__(self,
                  data : List[Datum]):
        self.data = data

# KD tree class.
class KDtree():
    def  __init__(self,
                  k    : int,
                  m    : int,
                  root = None):
        self.k    = k
        self.m    = m
        self.root = root

    # For the tree rooted at root, dump the tree to stringified JSON object and return.
    # DO NOT MODIFY.
    def dump(self) -> str:
        def _to_dict(node) -> dict:
            if isinstance(node,NodeLeaf):
                return {
                    "p": str([{'coords': datum.coords,'code': datum.code} for datum in node.data])
                }
            else:
                return {
                    "splitindex": node.splitindex,
                    "splitvalue": node.splitvalue,
                    "l": (_to_dict(node.leftchild)  if node.leftchild  is not None else None),
                    "r": (_to_dict(node.rightchild) if node.rightchild is not None else None)
                }
        if self.root is None:
            dict_repr = {}
        else:
            dict_repr = _to_dict(self.root)
        return json.dumps(dict_repr,indent=2)

    # Insert the Datum with the given code and coords into the tree.
    # The Datum with the given coords is guaranteed to not be in the tree.
    
    def insert(self, point: tuple[int], code: str):
        if self.root is None:
            self.root = NodeLeaf([Datum(point, code)])
        else:
            self.root = self._insert(self.root, Datum(point, code), 0)

    def _insert(self, node, datum, depth):
        if isinstance(node, NodeLeaf):
            node.data.append(datum)
            if len(node.data) > self.m:
                return self._split_leaf(node, depth)
            else:
                return node
        else:
            if datum.coords[node.splitindex] < node.splitvalue:
                node.leftchild = self._insert(node.leftchild, datum, depth + 1)
            else:
                node.rightchild = self._insert(node.rightchild, datum, depth + 1)
            return node

    def _split_leaf(self, node, depth):
        # Choosing the split dimension and value
        dimension, split_value = self._choose_split(node)
        left_data, right_data = [], []

        for datum in node.data:
            if datum.coords[dimension] < split_value:
                left_data.append(datum)
            else:
                right_data.append(datum)

        # Creating new nodes
        left_child = NodeLeaf(left_data) if left_data else None
        right_child = NodeLeaf(right_data) if right_data else None

        return NodeInternal(dimension, split_value, left_child, right_child)

    def _choose_split(self, node):
        max_spread = -1
        split_dimension = -1
        for dim in range(self.k):
            values = [datum.coords[dim] for datum in node.data]
            min_val, max_val = min(values), max(values)
            spread = max_val - min_val
            if spread > max_spread:
                max_spread = spread
                split_dimension = dim

        # Sort the data points based on the chosen dimension
        node.data.sort(key=lambda datum: datum.coords[split_dimension])

        # Find the median value and explicitly cast it to float
        median_index = len(node.data) // 2
        if len(node.data) % 2 == 0:
            split_value = float((node.data[median_index - 1].coords[split_dimension] + node.data[median_index].coords[split_dimension]) / 2)
        else:
            split_value = float(node.data[median_index].coords[split_dimension])

        return (split_dimension, split_value)

        
    def delete(self, point: tuple[int]):
        if self.root is not None:
            self.root, _ = self._delete(self.root, point, 0)

    def _delete(self, node, point, depth):
        
        if isinstance(node, NodeLeaf):
            node.data = [datum for datum in node.data if datum.coords != point]
            if len(node.data) == 0:
                return None, True  # Node is empty, should be removed
            return node, False  # Node is not empty, no need to recalculate split

        if isinstance(node, NodeInternal):
            if point[node.splitindex] < node.splitvalue:
                node.leftchild, child_altered = self._delete(node.leftchild, point, depth + 1)
            else:
                node.rightchild, child_altered = self._delete(node.rightchild, point, depth + 1)

            if child_altered:
                # Check if either child is None
                if node.leftchild is None or node.rightchild is None:
                    # Replace with the non-empty child if it's a leaf
                    return (node.leftchild or node.rightchild), True

                # Merge children if they are both underpopulated leaf nodes
                if self._is_underpopulated(node.leftchild) and self._is_underpopulated(node.rightchild):
                    return self._merge_children(node.leftchild, node.rightchild), True

            return node, False

        return node, False
    
    
    def _handle_underpopulated(self, node):
        if isinstance(node, NodeInternal):
            # Check if either child is None (empty)
            if node.leftchild is None or node.rightchild is None:
                surviving_child = node.leftchild if node.leftchild is not None else node.rightchild
                # Replace internal node with its surviving child if it's a leaf
                if isinstance(surviving_child, NodeLeaf):
                    return surviving_child
                # If surviving child is internal, it's already handled
            else:
                # If both children are present and underpopulated, merge them
                if self._is_underpopulated(node.leftchild) and self._is_underpopulated(node.rightchild):
                    return self._merge_children(node.leftchild, node.rightchild)
        
        return node

    def _is_underpopulated(self, node):
        # Check if a node is underpopulated, defined here as having fewer points than half of max capacity
        return isinstance(node, NodeLeaf) and len(node.data) < self.m / 2

    def _merge_children(self, left_child, right_child):
        # Merge the data of two leaf nodes into a new leaf node
        merged_data = left_child.data + right_child.data
        return NodeLeaf(merged_data)
            
        
    def knn(self, k: int, point: tuple[int]) -> str:
        self.leaves_checked = 0
        self.knn_list = PriorityQueue()
        self._knn_recursive(self.root, point, k, 0)
        knn_list = []

        while not self.knn_list.empty():
            knn_list.append(self.knn_list.get()[1].to_json())

        # Update the key to "leaveschecked" to match the expected format
        return json.dumps({"leaveschecked": self.leaves_checked, "points": knn_list}, indent=2)

    def _knn_recursive(self, node, point, k, depth):
        if node is None:
            return

        if isinstance(node, NodeLeaf):
            # Check each point in the leaf node
            for datum in node.data:
                dist = self._distance(datum.coords, point)
                if len(self.knn_list.queue) < k or dist < self.knn_list.queue[0][0]:
                    if len(self.knn_list.queue) == k:
                        self.knn_list.get()  # Remove the farthest point if list is full
                    self.knn_list.put((dist, datum))  # Add this point
            self.leaves_checked += 1
        else:
            # Determine which subtree is nearer based on the split value
            dim = depth % self.k  # Split dimension
            if point[dim] < node.splitvalue:
                nearer_node, farther_node = node.leftchild, node.rightchild
            else:
                nearer_node, farther_node = node.rightchild, node.leftchild

            # Search the nearer subtree first
            self._knn_recursive(nearer_node, point, k, depth + 1)

            # Check if we need to search the farther subtree
            if self._should_search_farther_subtree(point, node, k, dim):
                self._knn_recursive(farther_node, point, k, depth + 1)
                

    def _should_search_farther_subtree(self, point, node, k, dim):
        # Calculate squared distance to the split plane
        dist_to_split_squared = (point[dim] - node.splitvalue) ** 2

        # If k neighbors have not been found yet, always search the farther subtree
        if len(self.knn_list.queue) < k:
            return True

        # Compare with the squared distance of the farthest point in the knn list
        farthest_dist_squared = self.knn_list.queue[0][0] ** 2
        return dist_to_split_squared < farthest_dist_squared
                

                
    def _distance(self, coords1, coords2):
        return sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)) ** 0.5
    
    def _closer_to_bounding_box(self, point, node, dim):
        # The split value at the current dimension forms one 'side' of the bounding box
        split = node.splitvalue

        # Calculate the distance from the point to this 'side' of the bounding box
        # If the point's coordinate in the split dimension is less than the split,
        # the closest point on the bounding box is at the split value.
        # Otherwise, it's on the other side of the split.
        point_coord = point[dim]
        if point_coord < split:
            closest_point_coord = split
        else:
            closest_point_coord = split

        # Calculate the squared distance for this dimension
        # We can compare squared distances to avoid taking square roots, which is more efficient
        dist_squared = (point_coord - closest_point_coord) ** 2

        # Get the squared distance of the farthest point in the current k-nearest neighbors
        # Since the queue is a max heap based on distance, the farthest point is at the root
        farthest_dist_squared = self.knn_list.queue[0][0] ** 2

        # If the distance to the bounding box is less than the distance to the farthest neighbor,
        # then we need to check this subtree
        return dist_squared < farthest_dist_squared
