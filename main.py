import random

import numpy as np
import cv2
from layer import Layer

CR=2
EPOCHS = 500
FRAG_SIZE = 2
LEARNING_RATE = 0.001

if __name__ == '__main__':
    in_image_rgb = cv2.imread('res/01.bmp')
    rgb_shape = in_image_rgb.shape
    in_image = in_image_rgb[:, :, 0]#.flatten()
    shape = in_image.shape
    print(shape)

    fragments = []
    for x in range(0, shape[0], FRAG_SIZE):
        for y in range(0, shape[1], FRAG_SIZE):
            fragments.append(in_image[x:x + FRAG_SIZE, y:y + FRAG_SIZE])

    print(len(fragments))
    chosen_fragments = random.sample(fragments, 100)
    flat_chosen_fragments = [chosen_fragment.flatten() for chosen_fragment in chosen_fragments]

    # v for test images, do testowania sieci, a nie do treningu
    fragment_shape = fragments[0].shape
    flat_fragments = [fragment.flatten() for fragment in fragments]
    # ^ for test images

    N = len(flat_fragments[0])
    M = int(N/CR)
    hidden_layer = Layer(M,N,LEARNING_RATE)
    final_layer = Layer(N,M,LEARNING_RATE)

    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            print(f"epoch{epoch}")
        for pattern in flat_chosen_fragments:
            # pattern=np.array([1,2,3,4])
            hid_out = hidden_layer.get_output(pattern)
            # print(hid_out[0])
            fin_out = final_layer.get_output(hid_out)
            fin_err = pattern-fin_out
            hid_err = final_layer.get_error_hidden_layer(fin_err)
            # print(hid_err)
            # print(fin_err)
            final_layer.adjust_weights(fin_err,hid_out)
            hidden_layer.adjust_weights(hid_err,pattern)

    for i, flat_fragment in enumerate(flat_fragments):
        network_output = final_layer.get_output(hidden_layer.get_output(flat_fragment))
        for j, pix in enumerate(network_output):
            if pix > 255:
                network_output[j] = 255
            elif pix < 0:
                network_output[j] = 0
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


    # print(len(img1))

    out_image_rgb = cv2.merge([out_image]*3)
    print(out_image_rgb.shape)
    # reshaped = out_image.reshape(rgb_shape)
    cv2.imshow("test", out_image_rgb)
    cv2.waitKey(0)



class Network:
    pass


