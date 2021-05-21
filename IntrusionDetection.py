"""
IntrusionDetection.py

This program builds a background model using MOG2
and extracts the intruder location and
segments it using the watershed algorithm

Give path to directory with images as input after running the program

@author: Anushree Das (ad1707)
"""
from os.path import isfile, join
import cv2 as cv
from os import listdir
from os import path
import numpy as np
from common import *

def mog(pathIn):
    """
    Build background model using MOG and extract moving foreground object at different positions
    and create foreground mosaic
    :param filename: path to video file
    :return: None
    """
    # create MOG2 BackgroundSubtractor object
    mog = cv.createBackgroundSubtractorMOG2(
        history=20, varThreshold=16, detectShadows=True)

    # get files from the directory
    files = [f for f in listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])

    # used to compare brightness of 2 images
    src = cv.imread(join(pathIn, files[0]))
    src_mean = np.mean(src)

    n = len(files)
    frame_array = []
    colorimgs = []
    img_list = []

    keep_processing = True
    counter = -10
    filecounter = 0
    # Read all frames and build background model
    while (keep_processing):
        counter += 1
        # build background model
        if filecounter < n:
            print('Building background model..file:',files[filecounter])
            filename = join(pathIn, files[filecounter])
            # reading each files
            img = cv.imread(filename)
            colorimgs.append(img)
            # convert to grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # compare brightness of 2 images and make it equal for the new image
            img_mean = np.mean(img)
            if abs(img_mean - src_mean) > 0.5:
                img = img + ((img_mean - src_mean) / (img.shape[0]) + img.shape[1])

            # inserting the frames into an image array
            frame_array.append(img)
            mog.apply(img)
            filecounter += 1

        if counter>=0 and counter<n:
            frame = frame_array[counter]
            # apply MOG2 background subtraction on the frame
            fgmask = mog.apply(frame)

            # kernel for erosion
            erosionkernel = np.ones((8,8), np.uint8)
            # apply erosion to remove small changes that happen in the background
            erosion = cv.erode(fgmask, erosionkernel, iterations=7)
            # preprocessing
            erosion[:300,:] = 0
            erosion[-300:,:] = 0
            erosion[:,:400] = 0
            erosion[:,-100:] = 0

            # remove too large or too small masks
            nb_components, output, stats, centroids = cv.connectedComponentsWithStats(erosion, connectivity=8)
            # connectedComponentswithStats yields every seperated component with information on each of them, such as size
            # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
            sizes = stats[1:, -1];
            nb_components = nb_components - 1

            # minimum size and maximumm of particles we want to keep (number of pixels)
            # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
            min_size = 100
            max_size = 50000
            # result image after removing
            img2 = erosion.copy()
            # for every component in the image, you remove it only if it's not in range
            for i in range(0, nb_components):
                if not sizes[i] >= min_size or not sizes[i] < max_size:
                    img2[output == i + 1] = 0

            # if there is still a foreground object after preprocessing
            if cv.countNonZero(img2) > 0:
                # print(sizes)
                print('Creating foreground mosaic..file:',files[counter])

                src = colorimgs[counter]

                # kernel for dilation
                dilationkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 150))
                # apply dilation to make the region big enough to include the intruder
                dilation = cv.dilate(img2, dilationkernel, iterations=2)
                # get intruder mask
                intrudermask = cv.bitwise_and(fgmask, fgmask, mask=dilation)

                # process foreground to remove noise
                erosion = cv.erode(intrudermask, np.ones((2, 2)),iterations=2)
                foreground = cv.morphologyEx(erosion, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
                foreground = cv.dilate(foreground, np.ones((2, 2), np.uint8), anchor=(-1, -1), iterations=5)
                foreground = cv.morphologyEx(foreground, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
                foreground = np.uint8(foreground)

                # get foreground markers
                _,markers = cv.connectedComponents(foreground)
                markers = markers + 1
                # apply watershed
                markers = cv.watershed(src, markers)

                # create segmented image
                result = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
                for i in range(markers.shape[0]):
                    for j in range(markers.shape[1]):
                        index = markers[i, j]
                        # if marker of current point is equal to selected random point(50,50) in background
                        # make it green
                        if index == markers[50,50]:
                            result[i, j, :] = (0,255,0)
                        else:
                            # else make it red
                            result[i, j, :] = (0, 0, 255)
                result = cv.addWeighted(src, 0.5, result, 0.5, 0.0)
                img_list.append(result)

            # create mosaic
            if len(img_list)>=10:
                # call function from the common.py file
                temp = mosaic(5, img_list)
                cv.imshow('mosiac', temp)
                cv.imwrite('mosiac'+str(counter)+'.jpg',temp)
                cv.waitKey(0)
                cv.destroyAllWindows()
                img_list = []

        if counter >= n:
            keep_processing = False

    temp = mosaic(5, img_list)
    cv.imshow('mosiac', temp)
    cv.imwrite('mosiac' + str(counter) + '.jpg', temp)
    cv.waitKey(0)
    cv.destroyAllWindows()



def main():
    # src = 'SEQ_201_STILL_FRAMES__for_INTRUDER_IN_BACK_YARD'

    # take filename or directory name from user
    src = input('Enter filename or directory name:')

    if path.exists(src):
        # if input is directory
        if path.isdir(src):
            mog(src)
    else:
        print("File doesn't exist")


if __name__ == "__main__":
    main()