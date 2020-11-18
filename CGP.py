# -*- coding: utf-8 -*-
# Author: YeXiaona
# Date  : 2020-11-11
# CGP类文件

import numpy as np

class CartesionMap:

    def __init__(self, n_row, n_col, n_input):
        self.n_row = n_row
        self.n_col = n_col
        self.n_input = n_input

    def generateIndividual(self):
        size = self.n_row * self.n_col
        available_number = self.n_input
        individual = []
        for col in range(self.n_col):
            for row in range(self.n_row):
                node = CartesionNode(first_number=np.random.randint(0, available_number),
                                     second_number=np.random.randint(0, available_number),
                                     function=np.random.randint(0, 4),
                                     number=available_number + row)
                while self.isValid(individual, node) == False:
                    node = CartesionNode(first_number=np.random.randint(0,available_number),
                                         second_number=np.random.randint(0,available_number),
                                         function=np.random.randint(0,4),
                                         number=available_number+row)
                individual.append(node)
            available_number += self.n_row
        self.individual = individual
        return individual

    def isValid(self, exist_nodes, check_node):
        for node in exist_nodes:
            if node.isSame(check_node):
                return False
        return True

    def get_mapped_results(self, inputs):
        self.inputs = inputs
        mapped_results = []
        for i in range(self.n_row):
            mapped_results.append(self.get_node_result(self.individual[-1-i]))
        return mapped_results

    def get_node_result(self, node):
        if node.first_number >= self.n_input:
            first_input = self.get_node_result(self.individual[node.first_number-self.n_input])
        else:
            first_input = self.inputs[node.first_number]
        if node.second_number >= self.n_input:
            second_input = self.get_node_result(self.individual[node.second_number-self.n_input])
        else:
            second_input = self.inputs[node.second_number]
        if node.function == 0:
            return first_input + second_input
        elif node.function == 1:
            return  first_input - second_input
        elif node.function == 2:
            return  first_input * second_input
        else:
            if second_input == 0:
                return 0
            else:
                return first_input / second_input


class CartesionNode:

    def __init__(self, first_number, second_number, function, number):
        self.first_number = first_number
        self.second_number = second_number
        self.function = function
        self.number = number

    def isSame(self, node):
        if self.function == node.function:
            if self.function == 0 | self.function == 2:
                # 与计算顺序无关
                if self.first_number == node.first_number & self.second_number == node.second_number:
                    return True
                elif self.first_number == node.second_number & self.second_number == node.first_number:
                    return True
                else:
                    return False
            else:
                # 与计算顺序有关
                if self.first_number == node.first_number & self.second_number == node.second_number:
                    return True
                else:
                    return False
        else:
            return False