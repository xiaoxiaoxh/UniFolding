import cv2
import numpy as np


def segment(color_image, return_contour=False, remove_bottom=True, thresh_value=90):
    if len(color_image.shape) == 3:
        img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = color_image

    _, img_thresh = cv2.threshold(img_gray, thresh_value, 255, 0)

    img_thresh = cv2.erode(img_thresh, np.ones((8, 8), np.uint8))
    img_thresh = cv2.dilate(img_thresh, np.ones((16, 16), np.uint8))
    img_thresh = cv2.erode(img_thresh, np.ones((8, 8), np.uint8))

    # Remove robot at the bottom of image
    if remove_bottom:
        img_thresh[-100:, :] = 0

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_index = np.argmax([cv2.contourArea(c) for c in contours])
    all_contour = contours[contour_index]

    result = np.zeros_like(img_thresh)
    cv2.fillPoly(result, [all_contour], 255)

    moments = cv2.moments(all_contour)
    area = moments['m00'] / (color_image.shape[0] * color_image.shape[1])
    x = (moments['m10'] / moments['m00']) / color_image.shape[1]
    y = (moments['m01'] / moments['m00']) / color_image.shape[0]
    bounding_rect = cv2.boundingRect(all_contour)

    info = {'area': area, 'x': x, 'y': y, 'rect': bounding_rect}

    if return_contour:
        return result, info, all_contour
    return result, info