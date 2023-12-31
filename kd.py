from __future__ import annotations
import heapq
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
        leaves_checked = 0
        nearest_neighbors = []  # Min-heap based on negative distances

        def search(node):
            nonlocal leaves_checked
            if isinstance(node, NodeLeaf):
                # Check if this leaf should be processed
                current_furthest_distance = -nearest_neighbors[0][0] if nearest_neighbors else float('inf')
                if len(nearest_neighbors) < k or self._bounding_box_distance(node, point) <= current_furthest_distance:
                    leaves_checked += 1
                    for datum in node.data:
                        distance_squared = self._euclidean_distance_squared(datum.coords, point)
                        if len(nearest_neighbors) < k:
                            heapq.heappush(nearest_neighbors, (-distance_squared, datum))
                        elif distance_squared < current_furthest_distance:
                            heapq.heappushpop(nearest_neighbors, (-distance_squared, datum))
                return

            if isinstance(node, NodeInternal):
                axis = node.splitindex
                dist_to_split = max(0, (point[axis] - node.splitvalue) ** 2)
                primary, secondary = (node.leftchild, node.rightchild) if point[axis] < node.splitvalue else (node.rightchild, node.leftchild)

                # Visit the primary subtree
                search(primary)

                # Re-evaluate the condition for visiting the secondary subtree
                furthest_neighbor_distance = -nearest_neighbors[0][0] if nearest_neighbors else float('inf')
                if len(nearest_neighbors) < k or dist_to_split <= furthest_neighbor_distance:
                    search(secondary)

        search(self.root)

        # Sorting the neighbors based on actual distance (not squared)
        sorted_neighbors = sorted([datum for _, datum in nearest_neighbors], key=lambda x: self._euclidean_distance_squared(x.coords, point))
        return json.dumps({
            "leaveschecked": leaves_checked,
            "points": [datum.to_json() for datum in sorted_neighbors]
        }, indent=2)

    def _euclidean_distance_squared(self, point1, point2):
        # Helper method to calculate squared Euclidean distance between two points
        return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))

    def _bounding_box_distance(self, node, target):
    # Assuming node.data contains the points in the leaf node
    # and each point is a tuple (x, y, ...) representing coordinates.

        if not node.data:
            return float('inf')  # No data in the node, return infinite distance

        # Initialize min and max coordinates for each dimension
        min_coords = [float('inf')] * len(target)
        max_coords = [-float('inf')] * len(target)

        # Find the bounding box of the node
        for datum in node.data:
            for i, coord in enumerate(datum.coords):
                min_coords[i] = min(min_coords[i], coord)
                max_coords[i] = max(max_coords[i], coord)

        # Calculate the squared distance from the target to the bounding box
        distance_squared = 0
        for i, coord in enumerate(target):
            if coord < min_coords[i]:  # Target is left of the bounding box
                distance_squared += (min_coords[i] - coord) ** 2
            elif coord > max_coords[i]:  # Target is right of the bounding box
                distance_squared += (coord - max_coords[i]) ** 2

        return distance_squared