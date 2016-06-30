__author__ = 'Asad Raheem, Muhammad Ahsen Khawaja, Ruhaib ul Hassan'

import cv2
import sys
from copy import deepcopy


def preprocess(source_image, min_threshold=50, max_threshold=80):
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)  # convert image into gray scale
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)  # apply gaussian blur
    canny = cv2.Canny(gaussian, min_threshold, max_threshold)
    return canny, gaussian, gray


def find_ellipses(source_image, canny_image, min_points=5, \
                  axes_ratio=1.5, minor_axes_ratio=25, major_axes_ratio=15):
    # declaring variables
    i = 0
    height, width, channels = source_image.shape
    ellipse_list = []

    # find all the contours
    contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    number_of_contours = len(contours)

    # finding and filtering ellipses
    while i < number_of_contours:
        if len(contours[i]) >= min_points:
            ellipse = cv2.fitEllipse(contours[i])
            (x, y), (minor_axis, major_axis), angle = ellipse
            if minor_axis != 0 and major_axis != 0 and major_axis / minor_axis <= axes_ratio:
                ellipse_min_ratio = width / minor_axis
                ellipse_maj_ratio = height / major_axis

                if minor_axes_ratio >= ellipse_min_ratio >= 1.5 and major_axes_ratio >= ellipse_maj_ratio >= 1.5:
                    ellipse_list.append(ellipse)

        i += 1

    return ellipse_list


def main():
    source_image = cv2.imread(sys.argv[1], 1)
    output_image = deepcopy(source_image)
    canny, gaussian, gray = preprocess(source_image)
    ellipse_list = find_ellipses(source_image, canny)

    for ellipse in ellipse_list:
        cv2.ellipse(output_image, ellipse, (255, 0, 0), 6)

    output_file = "output_"+sys.argv[1]
    cv2.imwrite(output_file, output_image)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()

    else:
        print ("Usage: python tsd.py <JPG File Name>")
