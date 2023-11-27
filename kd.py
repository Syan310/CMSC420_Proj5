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
        # Create a Datum object from the point and code
        datum = Datum(point, code)

        # If the tree is empty, create a new leaf node as the root
        if self.root is None:
            self.root = NodeLeaf([datum])
        else:
            # Start the insertion process
            self._insert_recursive(self.root, None, datum, 0)

    def _insert_recursive(self, node, parent, datum, depth):
        if isinstance(node, NodeLeaf):
            # Add datum to the leaf node
            node.data.append(datum)

            # Check if the leaf node needs to be split
            if len(node.data) > self.m:
                self._split_leaf(node, parent, depth)
        else:
            # Determine the dimension to compare based on depth
            dim = depth % self.k

            # Decide whether to go left or right
            if datum.coords[dim] < node.splitvalue:
                self._insert_recursive(node.leftchild, node, datum, depth + 1)
            else:
                self._insert_recursive(node.rightchild, node, datum, depth + 1)

    def _find_split_dimension(self, leaf):
        max_spread = -float('inf')
        split_dim = 0

        for dim in range(self.k):
            coords = [datum.coords[dim] for datum in leaf.data]
            min_val, max_val = min(coords), max(coords)
            spread = max_val - min_val
            if spread > max_spread:
                max_spread = spread
                split_dim = dim

        # Ensure we are sorting and calculating the median for the correct dimension
        sorted_coords = sorted([datum.coords[split_dim] for datum in leaf.data])
        median_index = len(sorted_coords) // 2
        if len(sorted_coords) % 2 == 0:
            split_value = (sorted_coords[median_index - 1] + sorted_coords[median_index]) / 2
        else:
            split_value = sorted_coords[median_index]

        return split_dim, split_value


    def _split_leaf(self, leaf, parent, depth):
        dim, split_value = self._find_split_dimension(leaf)

        # Assign points to the left child if they are less than or equal to the split value
        left_data = [d for d in leaf.data if d.coords[dim] <= split_value]
        right_data = [d for d in leaf.data if d.coords[dim] > split_value]

        left_child = NodeLeaf(left_data)
        right_child = NodeLeaf(right_data)

        new_internal_node = NodeInternal(dim, split_value, left_child, right_child)
        if parent is None:
            self.root = new_internal_node
        else:
            if parent.leftchild == leaf:
                parent.leftchild = new_internal_node
            else:
                parent.rightchild = new_internal_node

        # Check if further splitting is needed for each child
        if len(left_child.data) > self.m:
            self._split_leaf(left_child, new_internal_node, depth + 1)
        if len(right_child.data) > self.m:
            self._split_leaf(right_child, new_internal_node, depth + 1)




    
    def delete(self, point: tuple[int]):
        if self.root is not None:
            self.root = self._delete_recursive(self.root, point, 0)

    def _delete_recursive(self, node, point, depth):
        if node is None:
            return None

        if isinstance(node, NodeLeaf):
            # Remove the point from the leaf node
            node.data = [d for d in node.data if d.coords != point]
            # If the leaf is empty, return None to remove the leaf
            return None if not node.data else node
        else:
            # Determine the dimension to compare based on depth
            dim = depth % self.k

            # Check which subtree the point belongs to
            if point[dim] < node.splitvalue:
                node.leftchild = self._delete_recursive(node.leftchild, point, depth + 1)
            else:
                node.rightchild = self._delete_recursive(node.rightchild, point, depth + 1)

            # If one of the children is now None, replace node with the other child
            if node.leftchild is None:
                return node.rightchild
            elif node.rightchild is None:
                return node.leftchild

            return node
        
        
    def knn(self, k: int, point: tuple[int]) -> str:
        self.leaves_checked = 0
        self.knn_list = PriorityQueue()
        self._knn_recursive(self.root, point, k, 0)
        knn_list = []

        while not self.knn_list.empty():
            knn_list.append(self.knn_list.get()[1].to_json())

        return json.dumps({"leaves_checked": self.leaves_checked, "points": knn_list}, indent=2)

    def _knn_recursive(self, node, point, k, depth):
        if node is None:
            return

        if isinstance(node, NodeLeaf):
            self.leaves_checked += 1
            for datum in node.data:
                dist = self._distance(datum.coords, point)
                if self.knn_list.qsize() < k or dist < self.knn_list.queue[0][0]:
                    if self.knn_list.qsize() == k:
                        self.knn_list.get()
                    self.knn_list.put((dist, datum))
        else:
            # Determine which subtree is closer
            dim = depth % len(point)
            if point[dim] < node.splitvalue:
                nearer_node, farther_node = node.leftchild, node.rightchild
            else:
                nearer_node, farther_node = node.rightchild, node.leftchild

            # Search the nearer subtree first
            self._knn_recursive(nearer_node, point, k, depth + 1)

            # Check if we need to search the farther subtree
            if self.knn_list.qsize() < k or self._closer_to_bounding_box(point, node, dim):
                self._knn_recursive(farther_node, point, k, depth + 1)

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
    