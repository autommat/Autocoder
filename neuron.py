import numpy as np
import random

class Neuron:
    def __init__(self, size, lr):
        # self.weights = np.array([random.uniform(-0.5,0.5) for _ in range(size)])
        self.weights = np.ones(size)
        self.lr = lr

    def calculate(self, input):
        if input.size != self.weights.size:
            raise ValueError
        return np.dot(self.weights, input)

    def adjust_weights(self, error: float, input):
        if input.size != self.weights.size:
            raise ValueError
        # print(input[0])
        delta = self.lr * error * input
        # print(error)
        # print(delta[0])
        # print("****")
        self.weights = self.weights + delta

    def get_nth_weight(self, n):
        return self.weights[n]
