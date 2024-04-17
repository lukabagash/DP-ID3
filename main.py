from math import sqrt
import numpy as np
from pathlib import Path
import csv


class Node:
    def __init__(self, key):
        self.children = []
        self.edges = []
        self.key = key

    def __repr__(self):
        return f"Node(key={self.key}, children={self.children}, edges={self.edges})"

    def add_child(self, obj, edge_key):
        self.children.append(obj)
        self.edges.append(edge_key)


def DP_ID3(data: dict, attributes, epsilon, max_depth, label):
    n = len(data) + np.random.laplace(1 / epsilon)
    f = max([len(set(data[a])) for a in attributes]) if len(attributes) > 0 else 1

    if (
        (len(attributes) == 0)
        or (max_depth == 0)
        or (n / (f * len(set(data[label]))) < (sqrt(2) / epsilon))
    ):
        label_count = dict()
        for i in set(data[label]):
            label_count[i] = data[label].count(i) + np.random.laplace(1 / epsilon)

        mode = [k for k, v in label_count.items() if v == max(label_count.values())][0]
        return Node(mode)

    G = dict()
    for a in attributes:
        G[a] = find_entropy_split(data, a, n, epsilon, label)

    print(G)

    a_hat = [k for k, v in G.items() if v == min(G.values())][0]

    root = Node(a_hat)

    for j in set(data[a_hat]):
        data_j = {
            k: [v for ind, v in enumerate(data[k]) if data[a_hat][ind] == j]
            for k in data.keys()
        }

        attributes_j = attributes[:]
        attributes_j.remove(a_hat)

        node_j = DP_ID3(data_j, attributes_j, epsilon, max_depth - 1, label)
        root.add_child(node_j, j)

    return root


def find_entropy_split(data, attribute, n, epsilon, label):
    total = 0
    for j in set(data[attribute]):
        subtree_count = data[attribute].count(j) + np.random.laplace(1 / epsilon)
        subtree_entropy = 0

        for i in set(data[label]):
            count = len(
                [
                    ind
                    for ind, v in enumerate(list(data.values())[0])
                    if (data[attribute][ind] == j) and (data[label][ind] == i)
                ]
            ) + np.random.laplace(1 / epsilon)

            p_i = abs(count / subtree_count)
            subtree_entropy -= p_i * np.log(p_i)
    total += (subtree_count * subtree_entropy) / n
    return total


def predict(tree, data, i):
    if len(tree.children) == 0:
        return tree.key
    else:
        for ind, child in enumerate(tree.children):
            if data[tree.key][i] == tree.edges[ind]:
                return predict(child, data, i)


data_file = input("Enter the path of the data file: ")
with open(data_file, "r") as f:
    lines = f.readlines()
    data = dict(
        [
            (k.strip("\n"), [v.strip("\n").split(",")[ind] for v in lines[1:]])
            for ind, k in enumerate(lines[0].split(","))
        ]
    )
    label = list(data.keys())[-1]
    tree = DP_ID3(data, list(data.keys()), 10, 10, label)
    print(tree)
    print("Tree generated")
    non_label_data = {k: v for k, v in data.items() if k != label}

    with open("output.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            list(data.keys())[:-1] + [f"{list(data.keys())[-1]} (Predicted)"]
        )
        for i in range(len(data[list(data.keys())[0]])):
            row = []
            for k, v in non_label_data.items():
                row.append(v[i])
            row.append(predict(tree, data, i))
            writer.writerow(row)

    print("Predictions written to output.csv")
