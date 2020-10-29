import os
import pickle
import numpy as np
import cv2 as cv
from fil_finder import FilFinder2D
import astropy.units as u
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


# %% Drawing
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

    eye_masks = masks[:, :, eye_idx].astype(np.uint8)
    n = eye_masks.shape[2]
    for i in range(n):
        mask = eye_masks[:, :, i].squeeze()
        eye_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[eye_outline != 0] = (255, 0, 0)

    yolk_masks = masks[:, :, yolk_idx].astype(np.uint8)
    n = yolk_masks.shape[2]
    for i in range(n):
        mask = yolk_masks[:, :, i].squeeze()
        yolk_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[yolk_outline != 0] = (0, 127, 0)

    return im


def draw_labels(im: np.ndarray, min_eye_diameters: list, body_measurements: list, yolk_measurements: list,
                scale: float) -> np.ndarray:
    font = cv.FONT_HERSHEY_PLAIN
    font_size = 2
    line_weight = 2

    padding_h = 0
    padding_w = 10

    for fish in min_eye_diameters:
        for eye in fish:
            diameter, pos_x, pos_y = eye[0], eye[1] + padding_w, eye[2] + padding_h
            im = cv.putText(im, '{:0.3f}mm'.format(diameter / scale), (pos_x, pos_y),
                            font, font_size, (255, 0, 0), line_weight, cv.LINE_AA)

    for fish, _ in enumerate(body_measurements):
        direction, area, pos_x, pos_y, skeleton, length = body_measurements[fish][0], body_measurements[fish][1],\
                                        body_measurements[fish][2] + padding_w, body_measurements[fish][3] + padding_h,\
                                        body_measurements[fish][4], body_measurements[fish][5]
        im = cv.putText(im, '{}'.format(direction),
                        (pos_x, pos_y), font, font_size, (0, 0, 255), line_weight, cv.LINE_AA)
        dy = int(cv.getTextSize('{}'.format(direction), font, font_size, line_weight)[1] * 2.5)
        im = cv.putText(im, 'Area: {:0.3f}mm^2'.format(area / (scale * scale)),
                        (pos_x, pos_y + dy), font, font_size, (0, 0, 255), line_weight, cv.LINE_AA)
        im = cv.putText(im, 'Length: {:0.3f}mm'.format(length / scale),
                        (pos_x - 50, pos_y - 70), font, font_size, (63, 192, 192), line_weight, cv.LINE_AA)

        im[skeleton != 0] = (0, 255, 255)

    for fish, _ in enumerate(yolk_measurements):
        area, pos_x, pos_y = yolk_measurements[fish][0], yolk_measurements[fish][1] + padding_w, yolk_measurements[fish][2] + padding_h
        im = cv.putText(im, '{:0.3f}mm^2'.format(area / (scale * scale)),
                        (pos_x, pos_y), font, font_size, (0, 127, 0), line_weight, cv.LINE_AA)

    return im


# %% RoIs
def check_rois(class_ids: np.ndarray, rois: np.ndarray, im_width: int) -> [np.ndarray, np.ndarray, np.ndarray]:
    body_idx, eye_idx, yolk_idx = get_idx_by_class(class_ids)

    # We check for eyes/yolks inside body roi FIRST, because even if we discard a body for crossing the image
    # boundaries the inner components might still be fine
    eye_idx = get_valid_inner_rois(rois, body_idx, eye_idx, im_width)
    yolk_idx = get_valid_inner_rois(rois, body_idx, yolk_idx, im_width)

    body_idx = get_valid_body_rois(rois, body_idx, im_width)

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


# %% Measurements
def measure_eyes(masks: np.ndarray, eye_idx: np.ndarray) -> list:
    min_eye_diameters = []

    eye_masks = masks[:, :, eye_idx].astype(np.uint8)
    n = eye_masks.shape[2]

    for i in range(n):
        mask = eye_masks[:, :, i].squeeze()
        lbl = sk_morph.label(mask > 0)
        region_props = sk_measure.regionprops(lbl, cache=False)

        temp_eye_diameters = []
        for region in region_props:
            binary_image = region.image.astype(np.uint8)
            # We take the minimum diameter of the convex hull of the eye because of cases where we have figure 8 eyes
            convex_hull = region.convex_image.astype(np.uint8)
            hull_props = sk_measure.regionprops(convex_hull)
            min_eye_diameter = hull_props[0].minor_axis_length
            # Eye centroid in image space with offset in x for placing eye labels
            eye_pos_x = int(hull_props[0].centroid[1] + (hull_props[0].bbox[3] / 2) + region.bbox[1])
            eye_pos_y = int(hull_props[0].centroid[0] + region.bbox[0])

            temp_eye_diameters.append([min_eye_diameter, eye_pos_x, eye_pos_y])

        min_eye_diameters.append(temp_eye_diameters)

    return min_eye_diameters


def measure_body(masks: np.ndarray, body_idx: np.ndarray):
    body_measurements = []
    body_masks = masks[:, :, body_idx].astype(np.uint8)
    n = body_masks.shape[2]

    for i in range(n):
        mask = body_masks[:, :, i]
        lbl = sk_morph.label(mask)
        region_props = sk_measure.regionprops(lbl, cache=False)[0]

        body_area = region_props.area
        body_pos_x = int(region_props.centroid[1])
        body_pos_y = int(region_props.centroid[0])

        # Find the end of the fish that is "heavier" - this should be the head end
        # Obviously won't work for C shaped fish, though...
        body_sum = np.sum(mask, axis=0)
        body_diff = np.abs(np.diff(body_sum))

        body_start = region_props.bbox[1]  # (body_sum != 0).argmax()
        body_stop = region_props.bbox[3]  # 2448 - (np.flip(body_sum) != 0).argmax()
        body_center = body_start + (body_stop - body_start) / 2
        left_body_sum = np.sum(body_sum[0:int(body_center)])
        right_body_sum = np.sum(body_sum[int(body_center): body_stop])

        if left_body_sum > right_body_sum:
            larva_direction = '<-'
        else:
            larva_direction = '->'

        # Myotome length
        skeleton = sk_morph.skeletonize(mask).astype(np.uint8)
        pruned_skeleton, body_length = prune_skeleton(skeleton)

        body_measurements.append([larva_direction, body_area, body_pos_x, body_pos_y, pruned_skeleton, body_length])

    return body_measurements


def prune_skeleton(skeleton):
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(skip_flatten=True)
    fil.create_mask(use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=100 * u.pix, skel_thresh=300 * u.pix, prune_criteria='length', max_prune_iter=5)

    pruned_skeleton = fil.skeleton_longpath
    length = fil.lengths(u.pix)[0].to_value()

    return pruned_skeleton, length
# def prune_skeleton(skeleton):
#     # Endpoints
#     endpoint1 = np.array(([-1, -1, -1],
#                           [-1,  1, -1],
#                           [-1,  1, -1]), dtype="int")
#
#     endpoint2 = np.array(([-1, -1, -1],
#                           [-1,  1, -1],
#                           [-1, -1,  1]), dtype="int")
#
#     endpoint3 = np.array(([-1, -1, -1],
#                           [-1,  1,  1],
#                           [-1, -1, -1]), dtype="int")
#
#     endpoint4 = np.array(([-1, -1,  1],
#                           [-1,  1, -1],
#                           [-1, -1, -1]), dtype="int")
#
#     endpoint5 = np.array(([-1,  1, -1],
#                           [-1,  1, -1],
#                           [-1, -1, -1]), dtype="int")
#
#     endpoint6 = np.array(([ 1, -1, -1],
#                           [-1,  1, -1],
#                           [-1, -1, -1]), dtype="int")
#
#     endpoint7 = np.array(([-1, -1, -1],
#                           [ 1,  1, -1],
#                           [-1, -1, -1]), dtype="int")
#
#     endpoint8 = np.array(([-1, -1, -1],
#                           [-1,  1, -1],
#                           [ 1, -1, -1]), dtype="int")
#
#     ep1 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint1)
#     ep2 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint2)
#     ep3 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint3)
#     ep4 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint4)
#     ep5 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint5)
#     ep6 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint6)
#     ep7 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint7)
#     ep8 = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, endpoint8)
#
#     pruned_skeleton = ep1 + ep2 + ep3 + ep4 + ep5 + ep6 + ep7 + ep8
#
#     return pruned_skeleton


def measure_yolk(masks: np.ndarray, yolk_idx: np.ndarray) -> list:
    yolk_measurements = []

    yolk_masks = masks[:, :, yolk_idx].astype(np.uint8)
    n = yolk_masks.shape[2]

    for i in range(n):
        mask = yolk_masks[:, :, i]
        lbl = sk_morph.label(mask)
        region_props = sk_measure.regionprops(lbl, cache=False)[0]

        yolk_area = region_props.area
        yolk_pos_x = int(region_props.centroid[1] + ((region_props.bbox[3] - region_props.bbox[1]) / 2))
        yolk_pos_y = int(region_props.centroid[0])

        yolk_measurements.append([yolk_area, yolk_pos_x, yolk_pos_y])

    return yolk_measurements


# %%
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

    for i, mask_1 in enumerate(body_masks):
        for j, mask_2 in enumerate(body_masks):
            if i != j:
                intersection = mask_1 * mask_2
                if np.any(intersection):
                    new_body_idx[body_idx_n[i]] = False
                    new_body_idx[body_idx_n[j]] = False

    return new_body_idx


# %%
def get_idx_by_class(class_ids: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    body_idx = eye_idx = yolk_idx = np.array([False] * class_ids.size)
    if 2 in class_ids:
        body_idx = (class_ids == 2)
    if 3 in class_ids:
        eye_idx = (class_ids == 3)
    if 1 in class_ids:
        yolk_idx = (class_ids == 1)

    return body_idx, eye_idx, yolk_idx


# %% Main
def main() -> None:
    scale = 287.0
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

                masks = correct_body_masks(masks, body_idx)
                body_idx = remove_overlapping_body_masks(masks, body_idx)
                body_measurements = measure_body(masks, body_idx)
                yolk_measurements = measure_yolk(masks, yolk_idx)
                min_eye_diameters = measure_eyes(masks, eye_idx)

                # im = draw_rois(im, rois, body_idx, eye_idx, yolk_idx)
                im = draw_masks(im, masks, body_idx, eye_idx, yolk_idx)
                im = draw_labels(im, min_eye_diameters, body_measurements, yolk_measurements, scale)

                cv.imshow('Neural Network output', im)
                cv.waitKey(0)


main()
