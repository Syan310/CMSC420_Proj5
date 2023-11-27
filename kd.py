from __future__ import annotations
import json
import math
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
    
    
    def insert(self, point: tuple[int], code: str):
        def _insert_recursive(node, depth):
            if node is None:
                return NodeLeaf([Datum(point, code)])

            idx = depth % self.k
            if point[idx] < (node.splitvalue if node.splitvalue is not None else float('inf')):
                node.leftchild = _insert_recursive(node.leftchild, depth + 1)
            else:
                node.rightchild = _insert_recursive(node.rightchild, depth + 1)
            return node

        self.root = _insert_recursive(self.root, 0)


    def delete(self, point: tuple[int]):
        def _delete_recursive(node, depth):
            if node is None:
                return None

            idx = depth % self.k
            if node.data and any(datum.coords == point for datum in node.data):
                # If it's a leaf node, remove the datum
                node.data = [datum for datum in node.data if datum.coords != point]
                return node if node.data else None
            elif point[idx] < node.splitvalue:
                node.leftchild = _delete_recursive(node.leftchild, depth + 1)
            else:
                node.rightchild = _delete_recursive(node.rightchild, depth + 1)
            return node

        self.root = _delete_recursive(self.root, 0)


    def knn(self, k: int, point: tuple[int]) -> str:
        def _knn_recursive(node, depth):
            if node is None:
                return []
            
            idx = depth % self.k
            next_branch = node.leftchild if point[idx] < node.splitvalue else node.rightchild
            opposite_branch = node.rightchild if next_branch is node.leftchild else node.leftchild

            best = _knn_recursive(next_branch, depth + 1)
            if len(best) < k or abs(point[idx] - node.splitvalue) < self._distance(point, best[-1].coords):
                best.extend(_knn_recursive(opposite_branch, depth + 1))
            
            return sorted(best, key=lambda datum: self._distance(point, datum.coords))[:k]

        return json.dumps({"leaveschecked": 0, "points": [datum.to_json() for datum in _knn_recursive(self.root, 0)]}, indent=2)