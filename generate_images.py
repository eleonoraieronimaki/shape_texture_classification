import diplib as dip
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
from utilities import *


def main():
    # parent_dir = "transformed/"
    # orig_images, dip_names = load_dip_images('data/')

    # # create original rescaled images
    # embr_norm = normalize_resize(orig_images)
    # #save_images(embr_norm, parent_dir+"orig_resize", "orig_resize")

    # # generate embryos images
    # embryos =  [dip.Kuwahara(x, 30, 10) for x in orig_images ]
    # embryos = normalize_resize(embryos)
    # # choose relevant channel
    # embr_gray = [ dip.Image(np.array(x)[:,:,2]) for x in embryos]
    # # threshold
    # embr_thresh = threshold_images(embr_gray)
    # #save_images(embr_thresh, parent_dir+"embr_thresh", "thresh")
    # # transform
    # embr_transf = apply_transformations(embr_thresh)
    # #save_images(embr_transf, parent_dir+"embr_transf", "transf")
    # # Cropping images
    # features = ['Minimum','Maximum', 'Size']
    # # diplib measurements
    # labeled_images, measurements = measure_elements(embr_transf, embr_gray, features=features)
    # # parse into numpy
    # np_mes = parse_features(measurements)
    # # define min max values to crop images
    # max_idxes = [np.argmax(x[:,4]) for x in np_mes]
    # maxs = [[np_mes[i][max_idxes[i]][2], 
    #         np_mes[i][max_idxes[i]][3]] 
    #             for i in range(len(np_mes))]
    # mins = [[np_mes[i][max_idxes[i]][0], 
    #         np_mes[i][max_idxes[i]][1]] 
    #             for i in range(len(np_mes))]
    # embr_norm = crop_images(embr_norm, minimum=mins, maximum=maxs)
    # save_images(embr_norm, parent_dir+"crop_orig_resize", "orig")
    # # cropped transformed images
    # embr_transf = crop_images(embr_transf, minimum=mins, maximum=maxs)
    # save_images(embr_transf, parent_dir+"crop_embr_transf", "transf")
    # # save grayscale cropped images for measurements
    # embr_gray = [ dip.Image(np.array(x)[:,:,2]) for x in embr_norm]
    # save_images(embr_gray, parent_dir+"crop_embr_gray", "gray")


    # # generate blues
    # orig_uint = [dip.Convert(x, "UINT8") for x in orig_images]
    # uint_rescaled = normalize_resize(orig_uint, normalize=False)
    # # apply gauss
    # blue_gauss = [dip.Gauss(x,sigmas=[3,3]) for x in uint_rescaled]
    # # choose relevant channel
    # blue_gray = [ dip.Image(np.array(x)[:,:,0]) for x in blue_gauss]
    # # threshold
    # blue_inverted = threshold_images(blue_gray)
    # blue_thresh = invert_colors(blue_inverted)
    # # crop relevant area based on embryos boundaries
    # blue_thresh = crop_images(blue_thresh, minimum=mins, maximum=maxs)
    # save_images(blue_thresh, parent_dir+"blue_thresh_crop", "thresh")
    # # save blue area gray images
    # uint_rescaled = crop_images(uint_rescaled, minimum=mins, maximum=maxs)
    # blue_gauss = [dip.Gauss(x,sigmas=[3,3]) for x in uint_rescaled]
    # blue_gray = [ dip.Image(np.array(x)[:,:,0]) for x in blue_gauss]
    # save_images(blue_gray, parent_dir+"blue_gray_crop", "gray")

    
    # No normalization
    parent_dir = "no_norm/"
    orig_images, dip_names = load_dip_images('data/')

    # create original rescaled images
    #embr_norm = normalize_resize(orig_images, normalize=False)
    #save_images(embr_norm, parent_dir+"orig_resize", "orig_resize")

    # generate embryos images
    embryos =  [dip.Kuwahara(x, 30, 10) for x in orig_images ]
    #embryos = normalize_resize(embryos, normalize=False)
    # choose relevant channel
    embr_gray = [ dip.Image(np.array(x)[:,:,2]) for x in embryos]
    # threshold
    embr_thresh = threshold_images(embr_gray)
    save_images(embr_thresh, parent_dir+"embr_thresh", "thresh")
    # transform
    embr_transf = apply_transformations(embr_thresh)
    save_images(embr_transf, parent_dir+"embr_transf", "transf")
    # Cropping images
    features = ['Minimum','Maximum', 'Size']
    # diplib measurements
    labeled_images, measurements = measure_elements(embr_transf, 
                                                    embr_gray, 
                                                    features=features)
    # parse into numpy
    np_mes = parse_features(measurements)
    # define min max values to crop images
    max_idxes = [np.argmax(x[:,4]) for x in np_mes]
    maxs = [[np_mes[i][max_idxes[i]][2], 
            np_mes[i][max_idxes[i]][3]] 
                for i in range(len(np_mes))]
    mins = [[np_mes[i][max_idxes[i]][0], 
            np_mes[i][max_idxes[i]][1]] 
                for i in range(len(np_mes))]
    embr_norm = crop_images(orig_images, minimum=mins, maximum=maxs)
    save_images(embr_norm, parent_dir+"crop_orig_resize", "orig")
    # cropped transformed images
    embr_transf = crop_images(embr_transf, minimum=mins, maximum=maxs)
    save_images(embr_transf, parent_dir+"crop_embr_transf", "transf")
    # save grayscale cropped images for measurements
    embr_gray = [ dip.Image(np.array(x)[:,:,2]) for x in embr_norm]
    save_images(embr_gray, parent_dir+"crop_embr_gray", "gray")

    # generate blues
    #orig_uint = [dip.Convert(x, "UINT8") for x in orig_images]
    #uint_rescaled = normalize_resize(orig_uint, normalize=False)
    # apply gauss
    blue_gauss = [dip.Gauss(x,sigmas=[3,3]) for x in orig_images]
    # choose relevant channel
    blue_gray = [ dip.Image(np.array(x)[:,:,0]) for x in blue_gauss]
    # threshold
    blue_inverted = threshold_images(blue_gray)
    blue_thresh = invert_colors(blue_inverted)
    # crop relevant area based on embryos boundaries
    blue_thresh = crop_images(blue_thresh, minimum=mins, maximum=maxs)
    save_images(blue_thresh, parent_dir+"blue_thresh_crop", "thresh")
    # save blue area gray images
    uint_rescaled = crop_images(orig_images, minimum=mins, maximum=maxs)
    blue_gauss = [dip.Gauss(x,sigmas=[3,3]) for x in uint_rescaled]
    blue_gray = [ dip.Image(np.array(x)[:,:,0]) for x in blue_gauss]
    save_images(blue_gray, parent_dir+"blue_gray_crop", "gray")


if __name__ == "__main__":
    main()