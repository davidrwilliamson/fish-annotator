import os
import pickle
import numpy as np
import cv2 as cv

from skimage import morphology as sk_morph
from skimage import measure as sk_measure
from skimage import transform as sk_transform

"""
TODO:
    Each body should contain at least one and no more than two eyes
    Each body should have at most one yolk sac
        Each yolk sac should have exactly one part
    Some kind of body length:width ration?
    Label fish with corresponding body parts
"""

output_folder = '/home/dave/Desktop/bernard_test/'
image_folder = '/media/dave/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-ctrl/'


def load_image(f: str) -> np.ndarray:
    im = np.load(os.path.join(image_folder, f + '.silc')).astype(np.uint8).squeeze()
    im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)

    return im


def draw_rois(im: np.ndarray, rois: np.ndarray, body_idx: np.ndarray, eye_idx: np.ndarray, yolk_idx: np.ndarray)\
        -> np.ndarray:
    for roi in rois[body_idx]:
        # NN outputs roi in form T, L, B, R
        # Remember that we're in BGR!
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (0, 0, 255), 2)
    for roi in rois[eye_idx]:
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (255, 0, 0), 2)
    for roi in rois[yolk_idx]:
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (0, 127, 0), 2)

    return im


def draw_masks(im: np.ndarray, masks: np.ndarray, body_idx: np.ndarray, eye_idx: np.ndarray, yolk_idx: np.ndarray)\
        -> np.ndarray:
    body_masks = masks[:, :, body_idx].astype(np.uint8)
    n = body_masks.shape[2]
    for i in range(n):
        mask = body_masks[:, :, i].squeeze()
        body_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[body_outline != 0] = (0, 0, 255)

    return im


def check_rois(class_ids: np.ndarray, rois: np.ndarray, im_width: int) -> [np.ndarray, np.ndarray, np.ndarray]:
    body_idx, eye_idx, yolk_idx = get_idx_by_class(class_ids)

    # We check for eyes/yolks inside body roi FIRST, because even if we discard a body for crossing the image
    # boundaries the inner components might still be fine
    eye_idx = get_valid_inner_rois(rois, body_idx, eye_idx, im_width)
    yolk_idx = get_valid_inner_rois(rois, body_idx, yolk_idx, im_width)

    body_idx = get_valid_body_rois(rois, body_idx, im_width)

    return body_idx, eye_idx, yolk_idx


def get_idx_by_class(class_ids: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    body_idx = eye_idx = yolk_idx = np.array([False] * class_ids.size)
    if 2 in class_ids:
        body_idx = (class_ids == 2)
    if 3 in class_ids:
        eye_idx = (class_ids == 3)
    if 1 in class_ids:
        yolk_idx = (class_ids == 1)

    return body_idx, eye_idx, yolk_idx


def get_valid_body_rois(rois: np.ndarray, body_idx: np.ndarray, im_width: int, margin: int = 0) -> np.ndarray:
    """ Return body rois that aren't touching or close to (margin > 0) the left and right image edges."""
    good_roi_idx = body_idx

    # Check that rois aren't at edge of screen
    for i in range(body_idx.size):
        if body_idx[i]:
            if (rois[i][1] <= 0 + margin) or (rois[i][3] >= im_width - margin):
                good_roi_idx[i] = False

    return good_roi_idx


def get_valid_inner_rois(rois: np.ndarray, body_idx: np.ndarray, inner_idx: np.ndarray,
                         im_width: int, margin: int = 0, tolerance: int = 10) -> np.ndarray:
    """ Check that inner ROIs are inside the body rois, within some tolerance.
        Check also that they are not crossing the image boundary. """
    good_roi_idx = inner_idx

    for i in range(inner_idx.size):
        if inner_idx[i]:
            # Inside body check
            in_at_least_one_body = False
            for j in range(body_idx.size):
                if body_idx[j]:
                    if np.any([rois[i][0] >= rois[j][0] - tolerance, rois[i][1] >= rois[j][1] - tolerance,
                               rois[i][2] <= rois[j][2] + tolerance, rois[i][3] <= rois[j][3] + tolerance]):
                        in_at_least_one_body = True
                        break  # Don't keep checking once we've found the ROI is inside a body
            good_roi_idx[i] = in_at_least_one_body

            # Boundary check
            if (rois[i][1] <= 0 + margin) or (rois[i][3] >= im_width - margin):
                good_roi_idx[i] = False

    return good_roi_idx


def measure_eyes(masks: np.ndarray, eye_idx: np.ndarray):
    eye_masks = masks[:, :, eye_idx].astype(np.uint8)
    n = eye_masks.shape[2]
    for i in range(n):
        mask = eye_masks[:, :, i].squeeze()
        lbl = sk_morph.label(mask > 0)
        region_props = sk_measure.regionprops(lbl, cache=False)

        for region in region_props:
            binary_image = region.image.astype(np.uint8)
            min_eye_diameter_1 = region.minor_axis_length
            convex_hull = region.convex_image.astype(np.uint8)
            hull_props = sk_measure.regionprops(convex_hull)
            min_eye_diameter_2 = hull_props[0].minor_axis_length

            edges = cv.dilate(binary_image, np.ones([3, 3])) - binary_image
        foo = -1


def correct_body_masks(masks: np.ndarray, body_idx: np.ndarray) -> np.ndarray:
    corrected_masks = masks.copy()
    body_masks = masks[:, :, body_idx]
    n = body_masks.shape[2]

    for i in range(n):
        mask = body_masks[:, :, i].squeeze()
        lbl = sk_morph.label(mask)
        region_props = sk_measure.regionprops(lbl, cache=False)

        # Find the largest region and take that as the body mask - assume other regions are artefacts
        areas = np.array([region.area for region in region_props])
        largest_region_idx = areas.argmax()
        body = region_props[largest_region_idx]

        new_mask = np.zeros((850, 2448), dtype=bool)
        new_mask[body.bbox[0]:body.bbox[2], body.bbox[1]:body.bbox[3]] = body.filled_image

        body_masks[:, :, i] = new_mask

    corrected_masks[:, :, body_idx] = body_masks

    return corrected_masks


def remove_overlapping_body_masks(masks: np.ndarray, body_idx: np.ndarray) -> np.ndarray:
    new_body_idx = body_idx.copy()
    if np.sum(body_idx) <= 1:
        return body_idx

    body_idx_n = [j for j in range(body_idx.size) if body_idx[j]]
    body_masks = [masks[:, :, i] for i in body_idx_n]

    for i in range(len(body_masks)):
        for j in range(len(body_masks)):
            if i != j:
                intersection = body_masks[i] * body_masks[j]
                if np.any(intersection):
                    new_body_idx[body_idx_n[i]] = False
                    new_body_idx[body_idx_n[j]] = False

    return new_body_idx


def measure_body(masks: np.ndarray, body_idx: np.ndarray):
    body_masks = masks[:, :, body_idx].astype(np.uint8)
    n = body_masks.shape[2]

    for i in range(n):
        mask = body_masks[:, :, i]


def main() -> None:
    files = [f for f in os.listdir(output_folder) if
             (os.path.isfile(os.path.join(output_folder, f)) and os.path.splitext(f)[1] != '.csv')]
    files.sort()

    for f in files:
        im = load_image(f)
        h, w = im.shape[0:2]

        with open(os.path.join(output_folder, f), 'rb') as file:
            nn_output = pickle.load(file)

        class_ids = nn_output['class_ids']
        rois = nn_output['rois']
        masks = nn_output['masks']

        body_idx, eye_idx, yolk_idx = check_rois(class_ids, rois, w)

        if np.any(body_idx):
            if np.any(eye_idx):
                im = draw_rois(im, rois, body_idx, eye_idx, yolk_idx)
                # measure_eyes(masks, eye_idx)
                masks = correct_body_masks(masks, body_idx)
                body_idx = remove_overlapping_body_masks(masks, body_idx)
                # measure_body(masks, body_idx)
                im = draw_masks(im, masks, body_idx, eye_idx, yolk_idx)

                cv.imshow('Neural Network output', im)
                cv.waitKey(0)


main()
