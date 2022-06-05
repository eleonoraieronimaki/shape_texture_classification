import diplib as dip
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
from utilities import *
from sklearn.cluster import KMeans


def main():
    features = ['Perimeter', 'Size',
                'Circularity', 'Roundness',
                'StandardDeviation', 'Minimum','Maximum']


    dip_images, dip_names = load_dip_images('data/')
    dip_images = normalize(dip_images)
    save_images(dip_images, 'normalized', "data", file_type=["tif"])
    dip_blue = make_grayscale(dip_images)
    save_images(dip_blue, 'blues', "blue", file_type=["tif"])
    print("Saved dip_blue")
    dip_thresh = threshold_images(dip_blue)
    save_images(dip_thresh, 'thresh', "thresh", file_type=["tif"])
    print("Saved thresh")
    dip_transf = apply_transformations(dip_thresh)
    save_images(dip_transf, 'transf', "transf", file_type=["tif"])
    print("Saved transf")
    label_images, measurements = measure_elements(dip_transf,dip_blue, features)
    areas, perimeters, circularity, roundness, stand_dev, minimum, maximum = measurements_array(measurements, features)
    dip_transf = crop_images(dip_transf,minimum,maximum)
    save_images(dip_transf, 'transf', "transf", file_type=["tif"])
    print("Saved cropped transf")
    dip_blue = crop_images(dip_blue,minimum,maximum)
    save_images(dip_blue, 'blues', "blue", file_type=["tif"])
    crop_original = crop_images(dip_images,minimum,maximum)
    save_images(crop_original, 'crop_original', "crop_original", file_type=["tif"])
    blue_areas, m_blues, blue_grays = blue_area(crop_original,features)
    save_images(blue_areas, 'blue_areas', "blue_area", file_type=["tif"])
    print("Saved blue_areas")
    save_images(blue_grays, 'blue_grays', "blue_gray", file_type=["tif"])
    label_images, measurements = measure_elements(dip_transf, dip_blue, features)
    areas, perimeters, circularity, roundness, stand_dev, minimum, maximum = measurements_array(measurements, features)
    areas_blue, perimeters_blue, circularity_blue, roundness_blue, std_blue, minimum_blue, maximum_blue = measurements_array(m_blues, features)
    hog_embryo = hog_img(dip_transf, orientation = 8, pixels_per_cell=(16,64))
    hog_blue = hog_img(blue_areas, orientation = 8, pixels_per_cell=(16,64))
    print("Creating hog_blue")
    save_images(hog_embryo, 'hog_embryos', "hog_embryos", file_type=["tif"])
    save_images(hog_blue, 'hog_blues', "hog_blue", file_type=["tif"])


if __name__ == "__main__":
    main()