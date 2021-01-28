from neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, neurons_num, neur_size, lr):
        self.neurons = [Neuron(neur_size, lr) for _ in range(neurons_num)]
        self.neur_size = neur_size

    def get_output(self, input):
        return np.array([neu.calculate(input) for neu in self.neurons])

    def get_error_hidden_layer(self, errors):
        if len(self.neurons) != len(errors):
            raise ValueError
        errors_out = [0]*self.neur_size
        for n in range(self.neur_size):
            sum =0
            for i in range(len(self.neurons)):
                sum += (errors[i] * self.neurons[i].get_nth_weight(n))
            errors_out[n] = sum
        # for i, neu in enumerate(self.neurons):
        #     for n in range(self.neur_size):
        #         errors_out[n] += neu.get_nth_weight(n)*errors[i]
        return np.array(errors_out)

    def adjust_weights(self, errors, input):
        if len(input) != self.neur_size or len(errors)!=len(self.neurons):
            raise ValueError
        for i, neu in enumerate(self.neurons):
            neu.adjust_weights(errors[i], input)

