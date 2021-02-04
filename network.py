from layer import Layer
import statistics

class Network:
    def __init__(self, in_out_neurs, hid_neurs, learning_rate):
        self.hidden_layer = Layer(hid_neurs, in_out_neurs, learning_rate)
        self.final_layer = Layer(in_out_neurs, hid_neurs, learning_rate)

    def train(self, input):
        hid_out = self.hidden_layer.get_output(input)
        # print(hid_out[0])
        fin_out = self.final_layer.get_output(hid_out)
        fin_err = input - fin_out
        hid_err = self.final_layer.get_error_hidden_layer(fin_err)
        # print(statistics.mean(abs(fin_err))) # debug
        self.final_layer.adjust_weights(fin_err, hid_out)
        self.hidden_layer.adjust_weights(hid_err, input)

    def test(self, input):
        return self.final_layer.get_output(self.hidden_layer.get_output(input))