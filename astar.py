import numpy as np


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def dist_func(position1: tuple, position2: tuple) -> float:
    """Calculate Euclidean distance"""
    return np.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2)


def search(maze: np.array, start: tuple, end: tuple) -> list:
    """
    AStar path search
    :param maze: two-dimensional array, element value 0 is walkable and 1 is terrain
    :param start: start position eg: (0, 0)
    :param end: end position eg: (5, 5)
    :return: a list of tuples as a path from the given start to the given end in the given maze
    """
    start_node = Node(None, tuple(start))
    end_node = Node(None, tuple(end))
    open_list = [start_node]  # in this list we will put all node that are yet_to_visit for exploration
    close_list = []  # in this list we will put all node those already explored so that we don't explore it again

    n_row, n_col = np.shape(maze)
    # (8 movements) from every position
    moves = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (1, 1), (-1, 1)]
    while len(open_list) > 0:
        # get the current node which has minimal f value
        current_node = open_list[0]
        current_index = 0
        for i, node in enumerate(open_list):
            if node.f < current_node.f:
                current_node = node
                current_index = i
        open_list.pop(current_index)
        close_list.append(current_node)

        # backtracking parent to get the path
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        for move in moves:
            new_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])
            # make sure new_position is in maze boundary
            if not (0 <= new_position[0] < n_row and 0 <= new_position[1] < n_col):
                continue
            # make sure new_position is walkable
            if maze[new_position[0]][new_position[1]] != 0:
                continue
            new_node = Node(current_node, new_position)
            if new_node in close_list:
                continue
            new_node.g = current_node.g + dist_func(current_node.position, new_node.position)
            # using distance to end_node as heuristic cost
            new_node.h = dist_func(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h
            if new_node in open_list:
                if open_list[open_list.index(new_node)].g < new_node.g:
                    continue
            open_list.append(new_node)
