import os
import numpy as np
import diplib as dip
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, distance
from torch import threshold


def load_dip_images(images_dir):
    """ Will only load the tif images of directory.
        Input:
            - images_dir: string containing the path
                of the directory of the image series
        Returns:
            - dip_images: list of loaded dip images
            - img_names: list of filenames
    """
    # check that directory is defined correctly
    if images_dir[-1] != "/":
        images_dir += "/"
    # load all .tif images
    dip_images = []
    img_names = []
    for filename in sorted(os.listdir(images_dir)):
        if ".tif" in filename:
            dip_images.append(dip.ImageReadTIFF(images_dir+filename))
            img_names.append(filename)
    return dip_images, img_names

def save_images(dip_images, images_dir, name_temp="", file_type="tif"):
    """ Saves the imput images
        Input:
            - dip_images: python list with the images to save
            - images_dir: string with the directory to save the images
            - name_temp: string containing the name structure 
                        to give to the series of images
            - file_type: can be string or list of file types to save image
    """
    # check that path is correctly defined
    if images_dir[-1] != "/":
        images_dir += "/"
    # create directories if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    # check name
    if name_temp != "" and name_temp[-1] != "-":
        name_temp += "-"
    # save images loop
    counter = 0
    for img in dip_images:
        if  "tif" in file_type:
            dip.ImageWriteTIFF(img,images_dir+name_temp+str(counter))
        if "jpeg" in file_type:
            # if the image is binary we need to multiply by 255
            if img.DataType() == "BIN":
                img = img * 255
            # save jpg image
            dip.ImageWriteJPEG(img,images_dir+name_temp+str(counter), 100)
            exit()
        counter += 1

def make_grayscale(dip_images: list):
    """ Converts the diplip images to grayscale
    """
    blues = []
    grays = []
    for img in dip_images:
        
        array= np.array(img)
        new_img = dip.ColorSpaceManager.Convert(img, "gray")
        blue_img = dip.Image(array[:,:,2])
        blues.append(blue_img)
        grays.append(new_img)
    return grays, blues

def threshold_images(dip_images, use_otsu=False):
    """ Triangle threshold all the dip_images in the list.
        The images will be converted to grayscale if 
        not already grayscale.

        Input:
            dip_images: list of grayscale diplib images to threshold
            save: string of dir to save output images.
                Images only saved if dir is specified.
        Return:
            list of the thresholded images
    """
    images_thresh = []
    for img in dip_images:
        # curr_img = dip.ContrastStretch(img)
        kuwahara =  dip.Kuwahara(img,kernel=30,threshold=10)
        curr_img = dip.TriangleThreshold(kuwahara)
        images_thresh.append(curr_img)
    return images_thresh

def apply_transformations(dip_images: list):
    """ Applies the defined transform to the dip_images
        Input:
            - dip_images: python list of diplib images
        Return:
            - out_images: python list with the transformed diplib images 
    """
    out_images = dip_images.copy()
    for i in range(len(out_images)):
        # opening to remove white pixel noise
        out_images[i] = dip.Opening(out_images[i])
        # closing to fill dark holes:
        out_images[i] = dip.Closing(out_images[i])
        # erosion to remove boundary pixels
        out_images[i] = dip.Erosion(out_images[i])
        out_images[i] = dip.Erosion(out_images[i])
        # dilation to extend object boundary to background
        # out_images[i] = dip.Dilation(out_images[i])
        # close any hole in the image
        # opening to remove white pixel noise
        # out_images[i] = dip.Opening(out_images[i])
        # out_images[i] = dip.FillHoles(out_images[i])
    return out_images

def measure_elements(dip_to_measure: list, dip_grayscale: list, 
                    features, min_size=None, max_size=None):
    """ Input: 
            - dip_to_measure: python list of the thresholded images to measure
            - dip_grayscale: python list of the grayscale versions of the image
            - features: python list with the features to be mesured
            - min_size: float representing the minimum area size to be considered as an object
            - max_size: float representing the maximum area size to be considered as an object
        Return: 
            -out_images: labeled images used fore the measurements
            -measurements: diplib measurements object with all the measurements
    
    """
    out_images = dip_to_measure.copy()
    measurements = []
    for i in range(len(out_images)):
        curr_img = out_images[i]
        curr_img = dip.EdgeObjectsRemove(curr_img)
        if min_size != None and max_size != None:
            curr_img = dip.Label(curr_img, connectivity=1,
                                    minSize=min_size, 
                                    maxSize=max_size,
                                    boundaryCondition=["remove"])
        elif min_size != None:
            curr_img = dip.Label(curr_img, connectivity=1,
                                minSize=min_size,
                                boundaryCondition=["remove"])
        elif max_size != None:
            curr_img = dip.Label(curr_img, connectivity=1,
                                maxSize=max_size,
                                boundaryCondition=["remove"])
        else:
            curr_img = dip.Label(curr_img, connectivity=1,
                                boundaryCondition=["remove"])
        out_images[i] = curr_img

        measurements.append(dip.MeasurementTool.Measure(curr_img, 
                                dip_grayscale[i], 
                                features))

    return out_images, measurements

def parse_features(measurements):
    """ Transforms the measurements object in numpy arrays
    """
    features = []
    for mes in measurements:
        curr_features = []
        for i in range(mes.NumberOfFeatures()):
            curr_feature = mes.Features()[i].name
            curr_feats = []

            for j in range(1, mes.NumberOfObjects()+1):
                curr_elem = mes[curr_feature][j]
                curr_feats.append(np.vstack(curr_elem))
            curr_feats = np.hstack(curr_feats)    
            curr_features.append(np.vstack(curr_feats))
        curr_features = np.vstack(curr_features).T
        features.append(curr_features)
    return features

