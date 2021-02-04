import math
import random
import numpy as np
import cv2
from network import Network

CR=6
EPOCHS = 5_000
FRAG_SIZE = 8
LEARNING_RATE = 5*10**-7
NUM_PATTERNS = 50

def calculate_PSNR(original, reproduced):
    sum_of_squared_diffs = 0
    for x in range(original.shape[0]):
        for y in range(original.shape[1]):
            sum_of_squared_diffs += (int(original[x,y,0])-int(reproduced[x,y,0]))**2
    value = 255**2 * 512**2 / sum_of_squared_diffs
    return 10*math.log10(value)

def opencv_image_to_net_input(image):
    image = image[:,:,0]
    shape = image.shape
    fragments=[]
    for x in range(0, shape[0], FRAG_SIZE):
        for y in range(0, shape[1], FRAG_SIZE):
            flat = image[x:x + FRAG_SIZE, y:y + FRAG_SIZE].flatten()
            fragments.append(flat)
            # print(flat)
    return fragments

def net_output_to_opencv_image(net_out):
    size = FRAG_SIZE * int((len(net_out)) ** 0.5)
    out_img = np.empty((size,size), dtype=np.uint8)
    # out_img = np.empty_like(in_image_rgb[:,:,0])
    i = 0
    for x in range(0, size, FRAG_SIZE):
        for y in range(0, size, FRAG_SIZE):
            out_img[x:x + FRAG_SIZE, y:y + FRAG_SIZE] = net_out[i].reshape((FRAG_SIZE, FRAG_SIZE))
            i += 1
    # out_img = normalize_image(out_img)
    return cv2.merge([out_img] * 3)

def normalize_val(val):
    if val>255:
        return 255
    elif val<0:
        return 0
    else:
        return int(val)

def normalize_np_array(array):
    for i in range(len(array)):
        array[i] = normalize_val(array[i])
    return array

def calculate_compr_degree(size, num_hid_neu, num_fin_neu):
    nom = size*size * 8
    denom = num_fin_neu * num_hid_neu * 8 + num_hid_neu * 12 * (size/FRAG_SIZE)**2
    return nom/denom

if __name__ == '__main__':
    train_filenames = ['img/01.bmp', "img/05.bmp"]
    train_patterns = []
    size = None
    for filename in train_filenames:
        in_image_rgb = cv2.imread(filename)
        if size is None:
            size = in_image_rgb.shape[0]
        fragments = opencv_image_to_net_input(in_image_rgb)
        train_patterns += random.sample(fragments, NUM_PATTERNS)

    in_out_neurs_number = len(train_patterns[0])
    hid_neurs_number = int(in_out_neurs_number/CR)
    net = Network(in_out_neurs_number, hid_neurs_number, LEARNING_RATE)

    psnr_file = open(f"out/psnr_CR{CR}_{EPOCHS}ep.txt", 'w')
    comp_deg = calculate_compr_degree(size,hid_neurs_number,in_out_neurs_number)
    psnr_file.write(f"cr: {CR} hid neurs: {hid_neurs_number}\n")
    psnr_file.write(f"compression degree: {comp_deg}\n")

    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            print(f"====EPOCH {epoch}====")
        for pattern in train_patterns:
            net.train(pattern)


    for x in range(1,9):
        img_in = cv2.imread(f"img/0{x}.bmp")
        fragments = opencv_image_to_net_input(img_in)
        output = [normalize_np_array(net.test(x)) for x in fragments]
        out_image= net_output_to_opencv_image(output)
        cv2.imwrite(f"out/img{x}_CR{CR}_{EPOCHS}ep.bmp", out_image)
        psnr = calculate_PSNR(img_in, out_image)
        print(psnr)
        psnr_file.write(f"{x} psnr {psnr}\n")
        with open(f"out/img{x}.csv", "a") as ratio:
            ratio.write(f"{comp_deg}, {psnr}\n")
    psnr_file.close()
    # cv2.imshow("function", out_image)
    # cv2.waitKey(0)

