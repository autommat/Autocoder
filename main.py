import math
import random
import statistics

import numpy as np
import cv2
from layer import Layer
from neuron import Neuron
import numpy as np

CR=2
EPOCHS = 5_000
FRAG_SIZE = 4
LEARNING_RATE = 5*10**-7
NUM_PATTERNS = 50

def PSNR(original, reproduced):
    sum_of_squared_diffs = 0
    for x in range(original.shape[0]):
        for y in range(original.shape[1]):
            sum_of_squared_diffs += int((original[x,y]-reproduced[x,y])**2)
        print(sum_of_squared_diffs)
    value = 255**2 * 512**2 / sum_of_squared_diffs
    return 10*math.log10(value)

def test_network():
    pass

if __name__ == '__main__':
    in_image_rgb = cv2.imread('img/01.bmp')
    rgb_shape = in_image_rgb.shape
    in_image = in_image_rgb[:, :, 0]#.flatten()
    # print("test PSNR:", PSNR(in_image,in_image))
    shape = in_image.shape
    print(shape)

    fragments = []
    for x in range(0, shape[0], FRAG_SIZE):
        for y in range(0, shape[1], FRAG_SIZE):
            fragments.append(in_image[x:x + FRAG_SIZE, y:y + FRAG_SIZE])

    print(len(fragments))
    chosen_fragments = random.sample(fragments, NUM_PATTERNS)
    flat_chosen_fragments = [chosen_fragment.flatten() for chosen_fragment in chosen_fragments]

    # v for test images, do testowania sieci, a nie do treningu
    fragment_shape = fragments[0].shape
    flat_fragments = [fragment.flatten() for fragment in fragments]
    # ^ for test images

    N = len(flat_fragments[0])
    M = int(N/CR)
    hidden_layer = Layer(M,N,LEARNING_RATE)
    final_layer = Layer(N,M,LEARNING_RATE)
    # hid_neurs = [Neuron(4,0.01,np.array([0.1,-0.2,0.3,-0.4])), Neuron(4,0.01,np.array([0.5,-0.6,0.7,-0.8]))]
    # fin_neurs = [Neuron(2,0.01,np.array([0.8,-0.7])), Neuron(2,0.01,np.array([0.6,-0.5])), Neuron(2,0.01,np.array([0.4,-0.3])), Neuron(2,0.01,np.array([0.2,-0.1]))]
    # hidden_layer=Layer(2,4,0.01, neurons=hid_neurs)
    # final_layer = Layer(4, 2, 0.01, neurons=fin_neurs)


    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            print(f"====EPOCH {epoch}====")
        for pattern in flat_chosen_fragments:
            # pattern=np.array([1,2,3,4])
            hid_out = hidden_layer.get_output(pattern)
            # print(hid_out[0])
            fin_out = final_layer.get_output(hid_out)
            fin_err = pattern-fin_out
            hid_err = final_layer.get_error_hidden_layer(fin_err)
            # print(hid_err)
            if(epoch%10==0):
                # print(statistics.mean(abs(fin_err))) # relevant
                pass
            final_layer.adjust_weights(fin_err,hid_out)
            hidden_layer.adjust_weights(hid_err,pattern)
            pass

    for i, flat_fragment in enumerate(flat_fragments):
        network_output = final_layer.get_output(hidden_layer.get_output(flat_fragment))
        # network_output = flat_fragment
        for j, pix in enumerate(network_output):
            if pix > 255:
                network_output[j] = 255
            elif pix < 0:
                network_output[j] = 0
            else:
                network_output[j]=int(pix)
        flat_fragments[i] = network_output

    reshaped_fragments = [flat_f.reshape(fragment_shape) for flat_f in flat_fragments]

    i=0
    out_image=np.empty_like(in_image)
    for x in range(0, shape[0], FRAG_SIZE):
        for y in range(0, shape[1], FRAG_SIZE):
            out_image[x:x + FRAG_SIZE, y:y + FRAG_SIZE] = reshaped_fragments[i]
            i+=1
    print(out_image.shape)
    # frag = img1[:32,:32]
    # frag = frag *2
    # img1[:32,:32] = frag

    print("PSNR value:", PSNR(in_image,out_image))
    # print(len(img1))+

    out_image_rgb = cv2.merge([out_image]*3)
    print(out_image_rgb.shape)
    # reshaped = out_image.reshape(rgb_shape)
    cv2.imshow("test", out_image_rgb)
    cv2.waitKey(0)
