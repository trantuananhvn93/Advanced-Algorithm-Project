import numpy as np
import datetime as d
import math
import queue
import time
from random import randint
import matplotlib.pyplot as plt


class Node:
    def __init__(self, A, B, cost):
        self.A = A
        self.B = B
        self.cost = cost
        self.h = heuristic(self.A, self.B)
        self.f = self.cost + self.h
        self.path = ''


def heuristic(A, B):
    return (abs(len(A) - len(B)))


def Branch_and_Bound(A, B):
    Q = queue.LifoQueue()
    node0 = Node(A, B, 0)
    bound = math.inf
    Q.put(node0)
    path = ''

    while (Q.empty() == False):
        node = Q.get()

        # if the node can be expanded:
        if (len(node.A) != 0 and len(node.B) != 0):
            # expand to three new nodes
            diff = 0 if node.A[-1] == node.B[-1] else 1
            nextNodes = []
            nextNodes.append(Node(node.A[:-1], node.B, node.cost + 1))
            nextNodes.append(Node(node.A, node.B[:-1], node.cost + 1))
            nextNodes.append(Node(node.A[:-1], node.B[:-1], node.cost + diff))
            # update the path of nodes
            nextNodes[0].path = node.path + 'U'
            nextNodes[1].path = node.path + 'L'
            nextNodes[2].path = node.path + 'D'

            # if the value of node higher than the bound, so cutoff, otherwise put it into the queue to expande next time
            for _node in nextNodes:
                if _node.f <= bound:
                    Q.put(_node)

        # if solution is found, the node cannot be expanded:
        elif (len(node.A) == 0 or len(node.B) == 0):
            # update the bound if higher than the value of the final node (solution),
            # and also update the path leading the new solution
            if bound >= node.f:
                path = node.path + len(node.B) * 'L' + len(node.A) * 'U'
                bound = node.f

    _A, _B = alignment(A, B, None, path)
    return bound, _A, _B


def alignment(A, B, ptr, L=None):
    # create a trackback line
    if L is None:
        L = []
        row = ptr.shape[0] - 1
        col = ptr.shape[1] - 1
        while (row != 0 or col != 0):
            if (ptr[row, col] == 1) or (ptr[row, col] == 7):
                L.append('D')
                row = row - 1
                col = col - 1
            elif (ptr[row, col] == 4) or (ptr[row, col] == 5) or (ptr[row, col] == 6):
                L.append('U')
                row = row - 1
            elif (ptr[row, col] == 2) or (ptr[row, col] == 3):
                L.append('L')
                col = col - 1
            else:
                print("------ERROR----")
                print("ptr:\n", ptr, "\nwhere ptr=", ptr[row, col], "row=", row, "col=", col)
                return [], []


                # Update A and B with alignment
    A2 = []
    B2 = []
    k = 0
    h = 0
    for i in range(len(L), 0, -1):
        if L[i - 1] == 'D':
            A2.append(A[k])
            B2.append(B[h])
            k += 1
            h += 1
        elif L[i - 1] == 'U':
            A2.append(A[k])
            B2.append('-')
            k += 1
        elif L[i - 1] == 'L':
            A2.append('-')
            B2.append(B[h])
            h += 1

    return A2, B2