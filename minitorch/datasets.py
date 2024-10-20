import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate a list of N random 2D points.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A list of N tuples, where each tuple contains two random float values between 0 and 1 to represent a 2Dpoint.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A class representing a graph with points and their classifications.

    Attributes
    ----------
        N : The number of points in the graph.
        X : A list of 2D points.
        y : A list of classifications for each point (0 or 1).

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a simple N-point dataset with a vertical decision boundary at x=0.5.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a N-point dataset with a diagonal decision boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a N-point dataset with two vertical decision boundaries at x=0.2 and x=0.8.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = (
            1 if x_1 < 0.2 or x_1 > 0.8 else 0
        )  # X within the boundaries: 0, X outside the boundaries: 1
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a N-point dataset with an XOR-like decision boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a N-point dataset with a circular decision boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their classifications.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a dataset with a spiral-like decision boundary.

    Args:
    ----
        N: The number of points to generate.

    Returns:
    -------
        A Graph object containing the generated points and their classifications.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


# Dictionary mapping dataset names to their corresponding generator functions
datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
