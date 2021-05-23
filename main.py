from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops

import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_largest_contour(contours):
    largest_surface = 0
    best_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > largest_surface:
            largest_surface = w * h
            best_contour = contour
    return best_contour

# smoothens the contour lines
# we do not use this as it is unnecessary for our use case
def smoothen(contour, span=3):
    new_contour = []
    # mean filter over the positions on the line
    for idx in range(span - 1, len(contour)):
        points = contour[idx - span + 1: idx + 1]
        new_contour.append((sum(points) / span).astype(int))
    return np.array(new_contour)


# returns the extreme points in the contour
def get_extreme_points(contour):
    fourPoints = np.zeros((4, 2), dtype=int)
    contour = contour.reshape(len(contour), 2)

    # assumption: point with the max and min x + y value will be the lower right and upper left corner
    s = contour.sum(axis=1)
    fourPoints[0] = contour[np.argmin(s)]
    fourPoints[2] = contour[np.argmax(s)]

    # assumption: point with the max and min x - y value will be the lower left and upper right corner
    d = np.diff(contour, axis=1)
    fourPoints[1] = contour[np.argmin(d)]
    fourPoints[3] = contour[np.argmax(d)]
    return fourPoints


if __name__ == "__main__":
    image_path = "./images/1_input.JPG"

    original = cv2.imread(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # blurring image
    blurred = cv2.GaussianBlur(gray, (11, 11), 5, 5)
    # thresholding gradient magnitude with canny
    cannied = cv2.Canny(blurred, 5, 20)
    # dilation operation to fill holes in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(cannied, kernel, iterations=4)

    # find contours
    # img, contours, _ = cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = get_largest_contour(contours)

    if not len(largest_contour):
        raise Exception("no contour found")

    extreme_points = (get_extreme_points(largest_contour))
    h, w, c = original.shape
    # point correspondences for homography in case of horizontal or vertical photo
    if h > w:
        dst = np.float32([[0, 0], [800, 0], [800, 1200], [0, 1200]])
    else:
        dst = np.float32([[800, 0], [800, 1200], [0, 1200], [0, 0]])

    cv2_M = cv2.getPerspectiveTransform(extreme_points.astype(np.float32), dst)
    cv2_undistorted = cv2.warpPerspective(original.copy(), cv2_M, (800, 1200))

    cv2.drawContours(original, [largest_contour], -1, (255, 0, 0), 3)
    for point in extreme_points:
        cv2.circle(original, tuple(point), 10, (0, 0, 255), 10)

    # cv2.namedWindow('Four Points', cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Scanned", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Four Points', 500, 600)
    # cv2.resizeWindow('Scanned', 800, 1200)

    # cv2.imshow("Four Points", original)
    # cv2.imshow("Scanned", cv2_undistorted)

    cv2.imwrite("./images/2_fourPoints.png", original)
    cv2.imwrite("./images/3_scanned.png", cv2_undistorted)
    print("Perspective Transformation completed!")
    print("")


    grayImage = cv2.cvtColor(cv2_undistorted, cv2.COLOR_BGR2GRAY)
    _, blackAndWhiteImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./images/4_scanned.png", blackAndWhiteImage)
    print("Changed to Black and White!")
    print("")

    print("Step 1/2 completed!")
    print("")



    ############################### HANDWRITING DETECTION AND REMOVAL ###############################

    # the parameters are used to remove small size connected pixels outliar
    constant_parameter_1 = 84
    constant_parameter_2 = 250
    constant_parameter_3 = 100

    # the parameter is used to remove big size connected pixels outliar
    constant_parameter_4 = 18

    # read the input image
    input_image = blackAndWhiteImage
    img = cv2.threshold(blackAndWhiteImage, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area / counter)

    a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3

    a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)

    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > (a4_big_size_outliar_constant)
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0

    plt.imsave('pre_version.png', pre_version)

    img = cv2.imread('pre_version.png', 0)
    cv2.imwrite("./images/5_handwritingExtractions.png", img)
    print("Handwriting identified!")
    print("")

    pre_version = img
    # ensure binary
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    inter_image = img

    for row in range(input_image.shape[0] - 1):
        for col in range(input_image.shape[1] - 1):
            if pre_version[row][col] != 30:
                input_image[row][col] = 255

                input_image[row + 1][col] = 255
                input_image[row - 1][col] = 255
                input_image[row][col + 1] = 255
                input_image[row][col - 1] = 255

                input_image[row + 1][col - 1] = 255
                input_image[row - 1][col + 1] = 255
                input_image[row + 1][col + 1] = 255
                input_image[row - 1][col - 1] = 255

    cv2.imwrite('./images/6_final.png', input_image)
    print("Handwriting extracted!")
    print("")
    print("Step 2/2 completed!")

