import os
import pickle
import numpy as np
import cv2 as cv
from progressbar import ProgressBar
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

def load_image(image_folder: str, f: str) -> np.ndarray:
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


def draw_labels(im: np.ndarray, eye_measurements: list, body_measurements: list, yolk_measurements: list,
                fish_parts: np.ndarray) -> np.ndarray:
    font = cv.FONT_HERSHEY_PLAIN
    font_size = 2
    line_weight = 2

    padding_h = 0
    padding_w = 10

    for fish in eye_measurements:
        if fish:
            for eye in fish:
                min_diameter, max_diameter, area, pos_x, pos_y = \
                    eye['Eye min diameter[mm]'], eye['Eye max diameter[mm]'], eye['Eye area[mm2]'], \
                    eye['Eye pos[px]'][0] + padding_w, eye['Eye pos[px]'][1] + padding_h
                im = cv.putText(im, '{:0.3f}mm'.format(min_diameter), (pos_x, pos_y),
                                font, font_size, (255, 0, 0), line_weight, cv.LINE_AA)

    for fish in body_measurements:
        if fish:
            direction, area, pos_x, pos_y, skeleton, length = \
                    fish['Facing direction'], fish['Body area[mm2]'], \
                    fish['Body pos[px]'][0] + padding_w, fish['Body pos[px]'][1] + padding_h, fish['Body skeleton'], \
                    fish['Myotome length[mm]']
            im = cv.putText(im, '{}'.format(direction),
                            (pos_x, pos_y), font, font_size, (0, 0, 255), line_weight, cv.LINE_AA)
            dy = int(cv.getTextSize('{}'.format(direction), font, font_size, line_weight)[1] * 2.5)
            im = cv.putText(im, 'Area: {:0.3f}mm2'.format(area),
                            (pos_x, pos_y + dy), font, font_size, (0, 0, 255), line_weight, cv.LINE_AA)
            im = cv.putText(im, 'Length: {:0.3f}mm'.format(length),
                            (pos_x - 50, pos_y - 70), font, font_size, (63, 192, 192), line_weight, cv.LINE_AA)

            im[skeleton != 0] = (0, 255, 255)

    for fish in yolk_measurements:
        if fish:
            area, pos_x, pos_y = fish['Yolk area[mm2]'], \
                                 fish['Yolk pos[px]'][0] + padding_w, fish['Yolk pos[px]'][1] + padding_h
            im = cv.putText(im, '{:0.3f}mm2'.format(area),
                            (pos_x, pos_y), font, font_size, (0, 127, 0), line_weight, cv.LINE_AA)

    return im


# %% Write output
def write_cod_csv_header(log_path: str) -> None:
    csv_header = 'Image ID,Date,Treatment,Dev.,Fish ID,' \
                 'Body area[mm2],Myotome length[mm],Myotome height[mm],' \
                 'Eye area[mm2],Eye min diameter[mm],Eye max diameter[mm],' \
                 'Yolk area[mm2],Yolk length[mm],Yolk height[mm],Yolk fraction' \
                 '\r\n'

    with open(log_path, 'a+') as log_file:
        log_file.write(csv_header)


def write_frame_to_csv(log_path: str, folder: str, file_name: str, fish: np.ndarray, body_measurements, eye_measurements, yolk_measurements, body_idx, eye_idx, yolk_idx):
    folder_split = folder.split('/')
    treatment = folder_split[-1]
    date = folder_split[-2]
    dev_stage = 'Larvae'

    # Here we go through the valid bodies and all the parts connected to them, writing out their measurements
    # Fish ID uses the internal number of the body (i.e., its position index) which is a bit messy but at least it
    # should be unique to each body, doesn't require any extra counting or work, and makes it easy to go back and check
    for n, body in enumerate(body_measurements):
        if body:
            eyes = [eye_measurements[part] for part in fish[n] if eye_measurements[part]]
            yolk = [yolk_measurements[part] for part in fish[n] if yolk_measurements[part]]
            fish_id = n
            if len(yolk) == 0:
                yolk = None
            else:
                yolk = yolk[0]
            if len(eyes) == 1:
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, body, eyes[0][0], yolk)
            elif len(eyes) == 2:
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, body, eyes[0][0], yolk)
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, None, eyes[1][0], None)

    # The above code only deals with parts that are connected to a body
    # Here we find the eyes and yolks that have been orphaned by body removal (for instance overlapping or edge masks)
    body_connected_parts = []
    for n, part in enumerate(fish):
        # If the part is a body
        if type(part) == np.ndarray:
            body_connected_parts.append(n)
            # If the body is actually included in our measurements
            if body_measurements[n]:
                for p in part:
                    body_connected_parts.append(p)
    if len(body_connected_parts) < len(fish):
        # Then some parts are not connected to a body
        unconnected_parts = [f for f in range(len(fish)) if f not in body_connected_parts]
        for p in unconnected_parts:
            # We still have a body listed in fish, even if it isn't valid any more, so fish ID still works + is unique
            fish_id = fish[p]
            if eye_measurements[p]:
                eye = eye_measurements[p]
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, None, eye[0], None)
            elif yolk_measurements[p]:
                yolk = yolk_measurements[p]
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, None, None, yolk)


def write_fish_to_csv(log_path: str, file_name: str, date: str, treatment: str, dev_stage: str, fish_num: int,
                      body_measurements: dict, eye_measurements: dict, yolk_measurements: dict) \
        -> None:

    with open(log_path, 'a+') as log_file:
        log_file.write('{},{},"{}",{},{:d},'.format(
                       file_name, date, treatment, dev_stage, fish_num,))
        if body_measurements:
            log_file.write('{:0.4f},{:0.4f},{:0.4f},'.format(
                       body_measurements['Body area[mm2]'],
                       body_measurements['Myotome length[mm]'],
                       body_measurements['Myotome height[mm]']))
        else:
            log_file.write(',,,')
        if eye_measurements:
            log_file.write('{:0.4f},{:0.4f},{:0.4f},'.format(
                       eye_measurements['Eye area[mm2]'],
                       eye_measurements['Eye min diameter[mm]'],
                       eye_measurements['Eye max diameter[mm]']))
        else:
            log_file.write(',,,')
        if yolk_measurements:
            if body_measurements:
                yolk_fraction = yolk_measurements['Yolk area[mm2]'] / body_measurements['Body area[mm2]']
            else:
                yolk_fraction = np.nan
            log_file.write('{:0.4f},{:0.4f},{:0.4f},{:0.4f}'.format(
                       yolk_measurements['Yolk area[mm2]'],
                       yolk_measurements['Yolk length[mm]'],
                       yolk_measurements['Yolk height[mm]'],
                       yolk_fraction))
        else:
            log_file.write(',,,')
        log_file.write('\r\n')


# %% RoIs
def check_rois(body_idx: np.ndarray, eye_idx: np.ndarray, yolk_idx: np.ndarray, rois: np.ndarray, im_width: int)\
               -> [np.ndarray, np.ndarray, np.ndarray]:
    # We check for eyes/yolks inside body roi FIRST, because even if we discard a body for crossing the image
    # boundaries the inner components might still be fine
    eye_idx = get_valid_inner_rois(rois, body_idx, eye_idx, im_width, 3)
    yolk_idx = get_valid_inner_rois(rois, body_idx, yolk_idx, im_width, 3)

    body_idx = get_valid_body_rois(rois, body_idx, im_width, 3)

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
def measure_eyes(masks: np.ndarray, eye_idx: np.ndarray, scale: float) -> list:
    eye_measurements = [None] * eye_idx.size

    for i in range(len(eye_idx)):
        if eye_idx[i]:
            mask = masks[:, :, i].astype(np.uint8).squeeze()
            lbl = sk_morph.label(mask > 0)
            region_props = sk_measure.regionprops(lbl, cache=False)

            temp_measurements = []
            for region in region_props:
                max_eye_diameter = region.major_axis_length / scale
                eye_area = region.area / (scale * scale)
                # We take the minimum diameter of the convex hull of the eye because of cases where we have figure 8 eyes
                convex_hull = region.convex_image.astype(np.uint8)
                hull_props = sk_measure.regionprops(convex_hull)
                min_eye_diameter = hull_props[0].minor_axis_length / scale

                # Eye centroid in image space with offset in x for placing eye labels
                eye_pos = (int(hull_props[0].centroid[1] + (hull_props[0].bbox[3] / 2) + region.bbox[1]),
                                            int(hull_props[0].centroid[0] + region.bbox[0]))

                temp_measurements.append({'Eye max diameter[mm]': max_eye_diameter,
                                          'Eye min diameter[mm]': min_eye_diameter,
                                          'Eye area[mm2]': eye_area,
                                          'Eye pos[px]': eye_pos})

            eye_measurements[i] = temp_measurements

    return eye_measurements


def measure_body(masks: np.ndarray, body_idx: np.ndarray, scale: float) -> list:
    body_measurements = [None] * body_idx.size

    for i in range(len(body_idx)):
        if body_idx[i]:
            mask = masks[:, :, i].astype(np.uint8)
            lbl = sk_morph.label(mask)
            region_props = sk_measure.regionprops(lbl, cache=False)[0]

            body_area = region_props.area / (scale * scale)
            body_pos = (int(region_props.centroid[1]), int(region_props.centroid[0]))

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
            pruned_skeleton, body_length = prune_skeleton(skeleton, mask)
            body_length /= scale
            # body_skeleton, body_length = extend_skeleton(pruned_skeleton, body_length, mask)

            body_skeleton = pruned_skeleton

            # Myotome height
            body_height = np.nan

            body_measurements[i] = {'Body area[mm2]': body_area,
                                    'Myotome length[mm]': body_length,
                                    'Myotome height[mm]': body_height,
                                    'Facing direction': larva_direction,
                                    'Body skeleton': body_skeleton,
                                    'Body pos[px]': body_pos}

    return body_measurements


def prune_skeleton(skeleton, mask: np.ndarray) -> [np.ndarray, float]:
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(skip_flatten=True)
    fil.create_mask(use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=100 * u.pix, skel_thresh=300 * u.pix, prune_criteria='length', max_prune_iter=5)

    pruned_skeleton = fil.skeleton_longpath
    if len(fil.end_pts) == 0:
        # We have no skeleton for some reason
        return pruned_skeleton, np.nan
    length = fil.lengths(u.pix)[0].to_value()

    skel_extension, extra_length = extend_skeleton(fil, mask)

    pruned_skeleton[skel_extension != 0] = 1
    length += extra_length

    return pruned_skeleton, length


def extend_skeleton(fil, mask: np.ndarray) -> [np.ndarray, float]:
    pruned_skeleton = fil.skeleton_longpath
    h, w = pruned_skeleton.shape
    skel_extension = np.zeros((h, w))
    extra_length = 0

    # Build a list of coordinates of the skeleton
    # I don't trust fil.branch_properties['pixels'][0][0] for this as it doesn't seem to match the claimed offset
    skel_coords = np.nonzero(pruned_skeleton)
    skel_coords = zip(skel_coords[0], skel_coords[1])
    skel_coords = np.array([c for c in skel_coords])

    # We want to get the direction the skeleton is more or less headed in at its ends
    # Tried various methods for this such as np.mean(np.diff(skel_coords[0:10, :], axis=0), axis=0)) and
    # slope_start, _, _, _, _ = scipy.stats.linregress(skel_coords[0:10])
    # start_grad = np.nan_to_num(np.array([1 / slope_start, 1], dtype='float64'))
    # They all seemed to work about the same and lingress gives us nasty NaN and near-inf outputs
    skel_gradient = np.gradient(skel_coords, axis=0)
    start_grad = np.array([np.mean(skel_gradient[0:10][:, 0]), np.mean(skel_gradient[0:10][:, 1])])
    end_grad = np.array([np.mean(skel_gradient[-11:-1][:, 0]), np.mean(skel_gradient[-11:-1][:, 1])])

    # Extend the skeleton from both ends in the direction it was headed until we reach the end of the body mask
    start_pos = np.array(fil.end_pts[0][0], dtype='float64')
    min_length = 99999
    line = np.zeros((h, w))
    # It's pretty gross to try all four possible directions, but we don't know which end is which (left/right)
    # We check the distance to the mask edge for each case and choose the shortest
    # It's not perfect but the difference shouldn't be too much
    for direction in [start_grad, -start_grad, [start_grad[0], -start_grad[1]], [-start_grad[0], start_grad[1]]]:
        curr_pos = start_pos.copy()
        inside_mask = True
        temp_line = np.zeros((h, w))
        while inside_mask:
            curr_pos += direction
            round_pos = np.round(curr_pos).astype(np.int64)
            # Check that we aren't going off the edge of the image
            if (round_pos != round_pos.clip([0, 0], [h, w])).any():
                inside_mask = False
                break
            inside_mask = mask[round_pos[0], round_pos[1]]
            if inside_mask:
                temp_line[round_pos[0], round_pos[1]] = 1
            else:
                line_length = np.linalg.norm(round_pos - start_pos)
                if line_length < min_length:
                    min_length = line_length
                    line = temp_line
    skel_extension[line != 0] = 1
    extra_length += min_length

    end_pos = np.array(fil.end_pts[0][1], dtype='float64')
    min_length = 99999
    line = np.zeros((h, w))
    for direction in [end_grad, -end_grad, [end_grad[0], -end_grad[1]], [-end_grad[0], end_grad[1]]]:
        curr_pos = end_pos.copy()
        inside_mask = True
        temp_line = np.zeros((h, w))
        while inside_mask:
            curr_pos += direction
            round_pos = np.round(curr_pos).astype(np.int64)
            # Check that we aren't going off the edge of the image
            if (round_pos != round_pos.clip([0, 0], [h, w])).any():
                inside_mask = False
                break
            inside_mask = mask[round_pos[0], round_pos[1]]
            if inside_mask:
                temp_line[round_pos[0], round_pos[1]] = 1
            else:
                line_length = np.linalg.norm(round_pos - end_pos)
                if line_length < min_length:
                    min_length = line_length
                    line = temp_line
    skel_extension[line != 0] = 1
    extra_length += min_length

    return skel_extension, extra_length


def measure_yolk(masks: np.ndarray, yolk_idx: np.ndarray, scale: float) -> list:
    yolk_measurements = [None] * yolk_idx.size

    for i in range(len(yolk_idx)):
        if yolk_idx[i]:
            mask = masks[:, :, i].astype(np.uint8)
            lbl = sk_morph.label(mask)
            region_props = sk_measure.regionprops(lbl, cache=False)[0]

            yolk_area = region_props.area / (scale * scale)
            yolk_pos_x = int(region_props.centroid[1])
            yolk_pos_y = int(region_props.centroid[0])

            # Assuming for now that these are good estimates of height & width
            yolk_width = region_props.major_axis_length / scale
            yolk_height = region_props.minor_axis_length / scale

            yolk_pos = (int(yolk_pos_x + ((region_props.bbox[3] - region_props.bbox[1]) / 2)),
                        int(yolk_pos_y))

            yolk_measurements[i] = {'Yolk area[mm2]': yolk_area,
                                    'Yolk length[mm]': yolk_width,
                                    'Yolk height[mm]': yolk_height,
                                    'Yolk pos[px]': yolk_pos}

    return yolk_measurements


# %%
def build_fish_associations(masks: np.ndarray, body_idx: np.ndarray, eye_idx: np.ndarray, yolk_idx: np.ndarray) \
                            -> np.ndarray:
    """
    We build an array of the same shape as the index arrays (body_idx, eye_idx, yolk_idx) where each element is:
        if it matches a body, a list of the indices of eyes and yolks that body contains
        if it matches an eye or a yolk, the index of the body containing that eye or yolk.
    If a body cointains no other parts it will have an empty list, if a part belongs to no body its index will be None.
    """
    n = body_idx.size
    # Portion of the part that has to be inside the body to be counted
    eye_threshold = yolk_threshold = 0.8

    fish = np.empty((n), dtype=np.object)
    for i in range(n):
        if body_idx[i]:
            fish[i] = []
            body_mask = masks[:, :, i].squeeze()
            # We treate eyes and yolks the same at the moment so could combine these two if statements, but could be
            # that at some point we'll want to treat them differently, different thresholds for instance
            for j in range(n):
                if eye_idx[j]:
                    eye_mask = masks[:, :, j].squeeze()
                    eye_pixels = np.sum(eye_mask)
                    overlapping_pixels = np.sum(body_mask * eye_mask)
                    if overlapping_pixels / eye_pixels > eye_threshold:
                        fish[i].append(j)
                        fish[j] = i
                elif yolk_idx[j]:
                    yolk_mask = masks[:, :, j].squeeze()
                    yolk_pixels = np.sum(yolk_mask)
                    overlapping_pixels = np.sum(body_mask * yolk_mask)
                    if overlapping_pixels / yolk_pixels > yolk_threshold:
                        fish[i].append(j)
                        fish[j] = i
            fish[i] = np.array(fish[i])

    return fish


def remove_orphans(fish: np.ndarray, body_idx: np.ndarray, eye_idx: np.ndarray, yolk_idx: np.ndarray) \
                   -> [np.ndarray, np.ndarray, np.ndarray]:
    """
        We remove orphan body parts *after* correcting masks but *before* doing overlap and image edge checks.
        This is because while we want masks to be accurate, we might for instance include eyes that we know had
        an associated body, even if that body was removed for crossing an image edge.
    """
    for f, T in enumerate(body_idx):
        if T:
            body_contents = fish[f]
            # Remove bodies that contain nothing
            if len(body_contents) == 0:
                body_idx[f] = False
            else:
                eyes = eye_idx[body_contents]
                n_eyes = np.sum(eyes)
                # Remove bodies with no eyes
                if n_eyes == 0:
                    body_idx[f] = False
                # Fish should have at most 2 eyes. If more, deal with that
                elif n_eyes > 2:
                    # print ('Too many eyes: {}'.format(n_eyes))
                    # ...by deleting all the eyes?
                    for e in body_contents:
                        if eye_idx[e]:
                            eye_idx[e] = False
                yolks = yolk_idx[body_contents]
                n_yolks = np.sum(yolks)
                # Fish should have at most 1 yolk. If more, deal with that
                if n_yolks > 1:
                    # print ('Too many yolks: {}'.format(n_yolks))
                    # ...by deleting all the yolks?
                    for y in body_contents:
                        if yolk_idx[y]:
                            yolk_idx[y] = False

    # Remove eyes and yolks lacking a corresponding body
    for e, T in enumerate(eye_idx):
        if T:
            body = fish[e]
            if body is None:
                eye_idx[e] = False
    for y, T in enumerate(yolk_idx):
        if T:
            body = fish[y]
            if body is None:
                yolk_idx[y] = False

    return body_idx, eye_idx, yolk_idx


# %% Mask correction
def correct_body_masks(masks: np.ndarray, body_idx: np.ndarray) -> np.ndarray:
    """ Discard all but largest region of each body mask and fill in holes."""
    corrected_masks = masks.copy()
    body_masks = masks[:, :, body_idx]
    h, w, n = body_masks.shape

    for i in range(n):
        mask = body_masks[:, :, i].squeeze()
        lbl = sk_morph.label(mask)
        region_props = sk_measure.regionprops(lbl, cache=False)

        # Find the largest region and take that as the body mask - assume other regions are artefacts
        areas = np.array([region.area for region in region_props])
        largest_region_idx = areas.argmax()
        body = region_props[largest_region_idx]

        new_mask = np.zeros((h, w), dtype=bool)
        new_mask[body.bbox[0]:body.bbox[2], body.bbox[1]:body.bbox[3]] = body.filled_image

        body_masks[:, :, i] = new_mask

    corrected_masks[:, :, body_idx] = body_masks

    return corrected_masks


def correct_eye_masks(masks: np.ndarray, eye_idx: np.ndarray) -> np.ndarray:
    corrected_masks = masks.copy()
    eye_masks = masks[:, :, eye_idx]
    h, w, n = eye_masks.shape

    for i in range(n):
        mask = eye_masks[:, :, i].squeeze()
        lbl = sk_morph.label(mask)
        region_props = sk_measure.regionprops(lbl, cache=False)

        # Find the largest region and take that, filled in, as the body mask - assume other regions are artefacts
        areas = np.array([region.area for region in region_props])
        largest_region_idx = areas.argmax()
        eye = region_props[largest_region_idx]

        new_mask = np.zeros((h, w), dtype=bool)
        new_mask[eye.bbox[0]:eye.bbox[2], eye.bbox[1]:eye.bbox[3]] = eye.filled_image

        eye_masks[:, :, i] = new_mask

    corrected_masks[:, :, eye_idx] = eye_masks

    return corrected_masks


def correct_yolk_masks(masks: np.ndarray, yolk_idx: np.ndarray) -> np.ndarray:
    corrected_masks = masks.copy()
    yolk_masks = masks[:, :, yolk_idx]
    h, w, n = yolk_masks.shape

    for i in range(n):
        mask = yolk_masks[:, :, i].squeeze()
        lbl = sk_morph.label(mask)
        region_props = sk_measure.regionprops(lbl, cache=False)

        # Discard all but the largest region and fill holes in yolk mask
        areas = np.array([region.area for region in region_props])
        largest_region_idx = areas.argmax()
        yolk = region_props[largest_region_idx]

        new_mask = np.zeros((h, w), dtype=bool)
        new_mask[yolk.bbox[0]:yolk.bbox[2], yolk.bbox[1]:yolk.bbox[3]] = yolk.filled_image

        yolk_masks[:, :, i] = new_mask

    corrected_masks[:, :, yolk_idx] = yolk_masks

    return corrected_masks


def remove_overlapping_masks(masks: np.ndarray, idx: np.ndarray) -> np.ndarray:
    new_idx = idx.copy()
    if np.sum(idx) <= 1:
        return idx

    idx_n = [j for j in range(idx.size) if idx[j]]
    chosen_masks = [masks[:, :, i] for i in idx_n]

    for i, mask_1 in enumerate(chosen_masks):
        for j, mask_2 in enumerate(chosen_masks):
            if i != j:
                intersection = mask_1 * mask_2
                if np.any(intersection):
                    new_idx[idx_n[i]] = False
                    new_idx[idx_n[j]] = False

    return new_idx


# %%
def get_idx_by_class(class_ids: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    body_idx = eye_idx = yolk_idx = np.array([False] * class_ids.size, dtype=np.bool)
    if 2 in class_ids:
        body_idx = (class_ids == 2)
    if 3 in class_ids:
        eye_idx = (class_ids == 3)
    if 1 in class_ids:
        yolk_idx = (class_ids == 1)

    return body_idx, eye_idx, yolk_idx


# %% Main
def analyse_folder(folder: str, image_folder: str, show_images: bool, write_csv: bool) -> None:
    scale = 287.0
    date, treatment = folder.split('/')[-2:]
    log_path = os.path.join(folder, '{}_{}_measurements_log.csv'.format(date, treatment))

    files = [f for f in os.listdir(folder) if
             (os.path.isfile(os.path.join(folder, f))
             and (os.stat(os.path.join(folder, f)).st_size > 408)
             and (os.path.splitext(f)[1] != '.csv'))]
    files.sort()

    if write_csv:
        write_cod_csv_header(log_path)

    pbar = ProgressBar()
    for f in pbar(files):
        im = load_image(image_folder, f)
        h, w = im.shape[0:2]

        with open(os.path.join(folder, f), 'rb') as file:
            nn_output = pickle.load(file)

        class_ids = nn_output['class_ids']
        rois = nn_output['rois']
        masks = nn_output['masks']

        body_idx, eye_idx, yolk_idx = get_idx_by_class(class_ids)

        if np.any(body_idx) and np.any(eye_idx):
            masks = correct_body_masks(masks, body_idx)
            masks = correct_eye_masks(masks, body_idx)
            masks = correct_yolk_masks(masks, yolk_idx)
            fish = build_fish_associations(masks, body_idx, eye_idx, yolk_idx)
            body_idx, eye_idx, yolk_idx = remove_orphans(fish, body_idx, eye_idx, yolk_idx)
            body_idx, eye_idx, yolk_idx = check_rois(body_idx, eye_idx, yolk_idx, rois, w)

            body_idx = remove_overlapping_masks(masks, body_idx)
            eye_idx = remove_overlapping_masks(masks, eye_idx)

            # Check that we still have anything left to measure at this point!
            if np.any(body_idx) or np.any(eye_idx):
                body_measurements = measure_body(masks, body_idx, scale)
                eye_measurements = measure_eyes(masks, eye_idx, scale)
                yolk_measurements = measure_yolk(masks, yolk_idx, scale)

                if show_images:
                    # im = draw_rois(im, rois, body_idx, eye_idx, yolk_idx)
                    im = draw_masks(im, masks, body_idx, eye_idx, yolk_idx)
                    im = draw_labels(im, eye_measurements, body_measurements, yolk_measurements, fish)

                    cv.imshow('Neural Network output', im)
                    cv.waitKey(0)

                if write_csv:
                    write_frame_to_csv(log_path, folder, f, fish, body_measurements, eye_measurements, yolk_measurements,
                                   body_idx, eye_idx, yolk_idx)


def main():
    show_images = False
    write_csv = True

    image_root_folder = '/media/dave/SINTEF Polar Night D/Easter cod experiments/Bernard/'
    nn_output_root_folder = '/home/dave/cod_results/uncropped/1246/'

    dates = ['20200413', '20200414', '20200415', '20200416', '20200417']
    treatments = ['1', '2', '3', 'DCA-ctrl', 'DCA-ctrl-2', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
    done = []

    for date in dates:
        for treatment in treatments:
            nn_output_folder = os.path.join(nn_output_root_folder, date, treatment)
            image_folder = os.path.join(image_root_folder, date, treatment)

            if os.path.isdir(nn_output_folder) and os.path.isdir(image_folder):
                if os.path.join(date, treatment) in done:
                    print('Skipping previously analysed folder {}'.format(nn_output_folder))
                else:
                    print('Analysing {}'.format(nn_output_folder))
                    analyse_folder(nn_output_folder, image_folder, show_images, write_csv)
                    print('    ...done')


main()
