import os
import pickle
import numpy as np
import cv2 as cv
from PIL import Image
# from fil_finder import FilFinder2D
# import astropy.units as u
import pandas as pd
from skimage import morphology as sk_morph
from skimage import measure as sk_measure
from skimage import transform as sk_transform
from progressbar import ProgressBar, Percentage, Bar, Timer, AdaptiveETA


def load_image(image_folder: str, f: str) -> np.ndarray:
    # im = np.load(os.path.join(image_folder, f + '.silc_bayer')).astype(np.uint8).squeeze()
    im = np.load(os.path.join(image_folder, f + '.silc')).astype(np.uint8).squeeze()
    im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)

    return im
    # if os.path.splitext(f)[-1] in ['.silc']:
    #     im = np.load(os.path.join(image_folder, f + '.silc')).astype(np.uint8).squeeze()
    #     im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)
    # elif os.path.splitext(f)[-1] in ['.silc_bayer']:
    #     im = np.load(os.path.join(image_folder, f + '.silc_bayer')).astype(np.uint8).squeeze()
    #     im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)
    # elif os.path.splitext(f)[-1] == '.png':
    #     im = Image.open(os.path.join(image_folder, f + '.png'))
    #     im = np.asarray(im)
    #     im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    # else:
    #     raise Exception('No valid image extension found')

    # return im


# %% Drawing
def draw_rois(im: np.ndarray, rois: np.ndarray, egg_idx: np.ndarray, embryo_idx: np.ndarray, yolk_idx: np.ndarray,
              eye_idx: np.ndarray) -> np.ndarray:
    for roi in rois[egg_idx]:
        # NN outputs roi in form T, L, B, R
        # Remember that we're in BGR!
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (0, 0, 255), 2)  # Egg ROI in red
    for roi in rois[embryo_idx]:  # Embryo ROI in dark green
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (0, 127, 0), 2)
    for roi in rois[yolk_idx]:  # Yolk ROI in dark blue
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (127, 0, 0), 2)
    for roi in rois[eye_idx]:  # Eye ROI in
        cv.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), (0, 0, 0), 2)
    return im


def draw_masks(im: np.ndarray, masks: np.ndarray, egg_idx: np.ndarray, embryo_idx: np.ndarray, yolk_idx: np.ndarray,
               eye_idx: np.ndarray) -> np.ndarray:
    egg_masks = masks[:, :, egg_idx].astype(np.uint8)
    n = egg_masks.shape[2]
    for i in range(n):
        mask = egg_masks[:, :, i].squeeze()
        egg_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[egg_outline != 0] = (255, 0, 255)  # Egg in magenta

    embryo_masks = masks[:, :, embryo_idx].astype(np.uint8)
    n = embryo_masks.shape[2]
    for i in range(n):
        mask = embryo_masks[:, :, i].squeeze()
        embryo_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[embryo_outline != 0] = (0, 255, 0)  # Embryo in green

    yolk_masks = masks[:, :, yolk_idx].astype(np.uint8)
    n = yolk_masks.shape[2]
    for i in range(n):
        mask = yolk_masks[:, :, i].squeeze()
        yolk_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[yolk_outline != 0] = (255, 255, 0)  # Yolk in cyan

    eye_masks = masks[:, :, eye_idx].astype(np.uint8)
    n = eye_masks.shape[2]
    for i in range(n):
        mask = eye_masks[:, :, i].squeeze()
        eye_outline = cv.dilate(mask, np.ones([5, 5])) - mask

        im[eye_outline != 0] = (255, 0, 0)  # Eye in blue

    return im


def draw_labels(im: np.ndarray, egg_measurements: list, yolk_measurements: list, eye_measurements: list) -> np.ndarray:
    font = cv.FONT_HERSHEY_PLAIN
    font_size = 2
    line_weight = 2

    padding_h = 0
    padding_w = 10

    for egg in egg_measurements:
        if egg:
            area, diameter, pos_x, pos_y = \
                egg['Egg area[mm2]'], egg['Egg min diameter[mm]'], \
                egg['Egg pos[px]'][0] + padding_w, egg['Egg pos[px]'][1] + padding_h
            im = cv.putText(im, '{:0.3f}mm'.format(diameter), (pos_x, pos_y),
                            font, font_size, (255, 0, 255), line_weight, cv.LINE_AA)
    #
    #     for fish in body_measurements:
    #         if fish:
    #             direction, area, pos_x, pos_y, skeleton, length = \
    #                     fish['Facing direction'], fish['Body area[mm2]'], \
    #                     fish['Body pos[px]'][0] + padding_w, fish['Body pos[px]'][1] + padding_h, fish['Body skeleton'], \
    #                     fish['Myotome length[mm]']
    #             im = cv.putText(im, '{}'.format(direction),
    #                             (pos_x, pos_y), font, font_size, (0, 0, 255), line_weight, cv.LINE_AA)
    #             dy = int(cv.getTextSize('{}'.format(direction), font, font_size, line_weight)[1] * 2.5)
    #             im = cv.putText(im, 'Area: {:0.3f}mm2'.format(area),
    #                             (pos_x, pos_y + dy), font, font_size, (0, 0, 255), line_weight, cv.LINE_AA)
    #             im = cv.putText(im, 'Length: {:0.3f}mm'.format(length),
    #                             (pos_x - 50, pos_y - 70), font, font_size, (63, 192, 192), line_weight, cv.LINE_AA)
    #
    #             im[skeleton != 0] = (0, 255, 255)
    #
    for fish in yolk_measurements:
        if fish:
            area, pos_x, pos_y = fish['Yolk area[mm2]'], \
                                 fish['Yolk pos[px]'][0] + padding_w, fish['Yolk pos[px]'][1] + padding_h
            im = cv.putText(im, '{:0.3f}mm2'.format(area),
                            (pos_x, pos_y), font, font_size, (255, 255, 0), line_weight, cv.LINE_AA)

    for fish in eye_measurements:
        if fish:
            area, pos_x, pos_y = fish[0]['Eye area[mm2]'], \
                                 fish[0]['Eye pos[px]'][0] + padding_w, fish[0]['Eye pos[px]'][1] + padding_h
            im = cv.putText(im, '{:0.4f}'.format(area),
                            (pos_x, pos_y), font, font_size, (255, 255, 0), line_weight, cv.LINE_AA)
            # print('\n{}'.format(fish))
    #
    return im


# %% Write output
def write_cod_egg_csv_header(log_path: str) -> None:
    csv_header = 'Image ID,Date,Treatment,Dev.,Fish ID,' \
                 'Egg area[mm2],Egg min diameter[mm],Egg max diameter[mm],' \
                 'Eye area[mm2],Eye min diameter[mm],Eye max diameter[mm],' \
                 'Yolk area[mm2],Yolk length[mm],Yolk height[mm],' \
                 'Embryo area[mm2]' \
                 '\r\n'

    with open(log_path, 'w') as log_file:
        log_file.write(csv_header)


def write_frame_to_csv(log_path: str, folder: str, file_name: str, fish, egg_measurements, yolk_measurements,
                       eye_measurements, embryo_measurements) -> None:
    folder_split = folder.split('/')
    treatment = folder_split[-1]
    date = folder_split[-2]
    dev_stage = 'Eggs'

    # Here we go through the valid eggs, writing out their measurements
    for n, egg in enumerate(egg_measurements):
        if egg:
            fish_id = n
            # write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, egg)

            eyes = [eye_measurements[part] for part in fish[n] if eye_measurements[part]]
            yolk = [yolk_measurements[part] for part in fish[n] if yolk_measurements[part]]
            embryo = [embryo_measurements[part] for part in fish[n] if embryo_measurements[part]]

            if len(yolk) == 0:
                yolk = None
            else:
                yolk = yolk[0]

            if len(embryo) == 0:
                # This should never happen, since we only look at eggs that have both "egg" and "embryo"
                # Update for 2020 reanalysis - now we look at eggs without embryos too, so this can happen just fine
                embryo = None
            else:
                embryo = embryo[0]

            if len(eyes) == 1:
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, egg, eyes[0][0], yolk,
                                  embryo)
            elif len(eyes) == 2:
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, egg, eyes[0][0], yolk,
                                  embryo)
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, None, eyes[1][0], None,
                                  None)
            else:
                write_fish_to_csv(log_path, file_name, date, treatment, dev_stage, fish_id, egg, None, yolk, embryo)


# def write_fish_to_csv(log_path: str, file_name: str, date: str, treatment: str, dev_stage: str, fish_num: int,
#                       egg_measurements: dict, yolk_measurements: dict) \
def write_fish_to_csv(log_path: str, file_name: str, date: str, treatment: str, dev_stage: str, fish_num: int,
                      egg_measurements: dict, eye_measurements: dict, yolk_measurements: dict,
                      embryo_measurements: dict) -> None:
    with open(log_path, 'a+') as log_file:
        log_file.write('{},{},"{}",{},{:d},'.format(
            file_name, date, treatment, dev_stage, fish_num, ))
        if egg_measurements:
            log_file.write('{:0.6f},{:0.6f},{:0.6f},'.format(
                egg_measurements['Egg area[mm2]'],
                egg_measurements['Egg min diameter[mm]'],
                egg_measurements['Egg max diameter[mm]']))
        else:
            log_file.write(',,,')
        if eye_measurements:
            log_file.write('{:0.6f},{:0.6f},{:0.6f},'.format(
                eye_measurements['Eye area[mm2]'],
                eye_measurements['Eye min diameter[mm]'],
                eye_measurements['Eye max diameter[mm]'], ))
        else:
            log_file.write(',,,')
        if yolk_measurements:
            log_file.write('{:0.6f},{:0.6f},{:0.6f},'.format(
                yolk_measurements['Yolk area[mm2]'],
                yolk_measurements['Yolk length[mm]'],
                yolk_measurements['Yolk height[mm]']))
        else:
            log_file.write(',,,')
        if embryo_measurements:
            log_file.write('{:0.6f}'.format(
                embryo_measurements['Embryo area[mm2]']))
        else:
            log_file.write('')
        log_file.write('\r\n')


# %% RoIs
def check_rois(egg_idx: np.ndarray, embryo_idx: np.ndarray, yolk_idx: np.ndarray, eye_idx: np.ndarray, rois: np.ndarray,
               im_width: int) \
        -> [np.ndarray, np.ndarray, np.ndarray]:
    tolerance = 10
    margin = 5

    embryo_idx = get_valid_inner_rois(rois, egg_idx, embryo_idx, im_width, margin, tolerance)
    yolk_idx = get_valid_inner_rois(rois, egg_idx, yolk_idx, im_width, margin, tolerance)
    eye_idx = get_valid_inner_rois(rois, egg_idx, eye_idx, im_width, margin, tolerance)
    # Check eggs LAST, because even eggs that are partly off-screen can have good parts inside
    egg_idx = get_valid_egg_rois(rois, egg_idx, im_width, margin)

    return egg_idx, embryo_idx, yolk_idx, eye_idx


def get_valid_egg_rois(rois: np.ndarray, egg_idx: np.ndarray, im_width: int, margin: int = 0) -> np.ndarray:
    """ Return egg rois that aren't touching or close to (margin > 0) the left and right image edges."""
    good_roi_idx = egg_idx

    # Check that rois aren't at edge of screen
    for i in range(egg_idx.size):
        if egg_idx[i]:
            if (rois[i][1] <= 0 + margin) or (rois[i][3] >= im_width - margin):
                good_roi_idx[i] = False

    return good_roi_idx


def get_valid_inner_rois(rois: np.ndarray, egg_idx: np.ndarray, inner_idx: np.ndarray,
                         im_width: int, margin: int = 0, tolerance: int = 10) -> np.ndarray:
    """ Check that inner ROIs are inside the body rois, within some tolerance.
        Check also that they are not crossing the image boundary. """
    good_roi_idx = inner_idx

    for i in range(inner_idx.size):
        if inner_idx[i]:
            # Inside body check
            in_at_least_one_body = False
            for j in range(egg_idx.size):
                if egg_idx[j]:
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


def measure_egg(masks: np.ndarray, egg_idx: np.ndarray, scale: float) -> list:
    egg_measurements = [None] * egg_idx.size

    for i in range(len(egg_idx)):
        if egg_idx[i]:
            mask = masks[:, :, i].astype(np.uint8)
            lbl = sk_morph.label(mask)
            region_props = sk_measure.regionprops(lbl, cache=False)[0]

            egg_area = region_props.area / (scale * scale)
            egg_pos = (int(region_props.centroid[1]), int(region_props.centroid[0]))

            # Egg diameter we get just by averaging the two axes (since it should be a sphere)
            egg_min_axis = region_props.minor_axis_length / scale
            egg_maj_axis = region_props.major_axis_length / scale
            egg_diameter = (egg_min_axis + egg_maj_axis) / 2.0
            r = egg_diameter / 2.
            egg_volume = (4. / 3.) * np.pi * r * r * r

            egg_measurements[i] = {'Egg area[mm2]': egg_area,
                                   'Egg min diameter[mm]': egg_min_axis,
                                   'Egg max diameter[mm]': egg_maj_axis,
                                   'Egg pos[px]': egg_pos}

    return egg_measurements


def measure_yolk(masks: np.ndarray, yolk_idx: np.ndarray, scale: float) -> list:
    yolk_measurements = [None] * yolk_idx.size

    for i in range(len(yolk_idx)):
        if yolk_idx[i]:
            mask = masks[:, :, i].astype(np.uint8)
            lbl = sk_morph.label(mask)
            try:
                region_props = sk_measure.regionprops(lbl, cache=False)[0]
            except IndexError:
                yolk_measurements[i] = {'Yolk area[mm2]': -1,
                                        'Yolk length[mm]': -1,
                                        'Yolk height[mm]': -1,
                                        'Yolk pos[px]': -1}
                continue

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


def measure_embryo(masks: np.ndarray, embryo_idx: np.ndarray, scale: float) -> list:
    embryo_measurements = [None] * embryo_idx.size

    for i in range(len(embryo_idx)):
        if embryo_idx[i]:
            mask = masks[:, :, i].astype(np.uint8)
            lbl = sk_morph.label(mask)
            region_props = sk_measure.regionprops(lbl, cache=False)[0]

            embryo_area = region_props.area / (scale * scale)

            embryo_measurements[i] = {'Embryo area[mm2]': embryo_area}

    return embryo_measurements


# %%
def build_fish_associations(masks: np.ndarray, egg_idx: np.ndarray, embryo_idx: np.ndarray, yolk_idx: np.ndarray,
                            eye_idx: np.ndarray) \
        -> np.ndarray:
    """
    We build an array of the same shape as the index arrays (body_idx, eye_idx, yolk_idx) where each element is:
        if it matches a body, a list of the indices of eyes and yolks that body contains
        if it matches an eye or a yolk, the index of the body containing that eye or yolk.
    If a body cointains no other parts it will have an empty list, if a part belongs to no body its index will be None.
    """
    n = egg_idx.size
    # Portion of the part that has to be inside the body to be counted
    embryo_threshold = yolk_threshold = eye_threshold = 0.8

    fish = np.empty((n), dtype=object)
    for i in range(n):
        if egg_idx[i]:
            fish[i] = []
            egg_mask = masks[:, :, i].squeeze()
            # We treat embryos and yolks the same at the moment so could combine these two if statements, but could be
            # that at some point we'll want to treat them differently, different thresholds for instance
            for j in range(n):
                if embryo_idx[j]:
                    embryo_mask = masks[:, :, j].squeeze()
                    embryo_pixels = np.sum(embryo_mask)
                    overlapping_pixels = np.sum(egg_mask * embryo_mask)
                    if overlapping_pixels / embryo_pixels > embryo_threshold:
                        fish[i].append(j)
                        fish[j] = i
                elif yolk_idx[j]:
                    yolk_mask = masks[:, :, j].squeeze()
                    yolk_pixels = np.sum(yolk_mask)
                    overlapping_pixels = np.sum(egg_mask * yolk_mask)
                    if overlapping_pixels / yolk_pixels > yolk_threshold:
                        fish[i].append(j)
                        fish[j] = i
                elif eye_idx[j]:
                    eye_mask = masks[:, :, j].squeeze()
                    eye_pixels = np.sum(eye_mask)
                    overlapping_pixels = np.sum(egg_mask * eye_mask)
                    if overlapping_pixels / eye_pixels > eye_threshold:
                        fish[i].append(j)
                        fish[j] = i
            fish[i] = np.array(fish[i])

    return fish


def remove_orphans(fish: np.ndarray, egg_idx: np.ndarray, embryo_idx: np.ndarray, yolk_idx: np.ndarray,
                   eye_idx: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
        Unlike with larvae we don't want yolk or embryo body parts without an associated egg so we do this check later.
    """
    for f, T in enumerate(egg_idx):
        if T:
            egg_contents = fish[f]
            # Remove bodies that contain nothing
            if len(egg_contents) == 0:
                egg_idx[f] = False
            else:
                embryos = embryo_idx[egg_contents]
                n_embryos = np.sum(embryos)
                # Eggs should have exactly 1 embryo. If not, deal with that
                if n_embryos != 1:
                    egg_idx[f] = False
                    for e in egg_contents:
                        if embryo_idx[e]:
                            embryo_idx[e] = False

                yolks = yolk_idx[egg_contents]
                n_yolks = np.sum(yolks)
                # Fish should have at most 1 yolk. If more, deal with that
                if n_yolks > 1:
                    # print ('Too many yolks: {}'.format(n_yolks))
                    # ...by deleting all the yolks?
                    for y in egg_contents:
                        if yolk_idx[y]:
                            yolk_idx[y] = False

    # Remove embryos, yolks and eyes lacking a corresponding body
    for e, T in enumerate(embryo_idx):
        if T:
            egg = fish[e]
            if egg is None:
                embryo_idx[e] = False
    for y, T in enumerate(yolk_idx):
        if T:
            egg = fish[y]
            if egg is None:
                yolk_idx[y] = False
    for y, T in enumerate(eye_idx):
        if T:
            egg = fish[y]
            if egg is None:
                eye_idx[y] = False

    return egg_idx, embryo_idx, yolk_idx, eye_idx


# %% Mask correction
def correct_masks(masks: np.ndarray) -> np.ndarray:
    """ Discard all but largest region of each mask and fill in holes.
    We can treat egg, embryo and yolk in the same way."""
    corrected_masks = masks.copy()
    h, w, n = masks.shape

    for i in range(n):
        mask = masks[:, :, i].squeeze()
        # On the rare occasion the mask is entirely empty for some reason, just skip this and return the empty mask
        if mask.any():
            lbl = sk_morph.label(mask)
            region_props = sk_measure.regionprops(lbl, cache=False)

            # Find the largest region and take that as the body mask - assume other regions are artefacts
            areas = np.array([region.area for region in region_props])
            largest_region_idx = areas.argmax()
            part = region_props[largest_region_idx]

            new_mask = np.zeros((h, w), dtype=bool)
            new_mask[part.bbox[0]:part.bbox[2], part.bbox[1]:part.bbox[3]] = part.filled_image

            corrected_masks[:, :, i] = new_mask
        # else:
        #     print ('No masks!?')

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
    egg_idx = embryo_idx = yolk_idx = eye_idx = np.array([False] * class_ids.size, dtype=bool)
    if 2 in class_ids:
        egg_idx = (class_ids == 2)
    if 1 in class_ids:
        embryo_idx = (class_ids == 1)
    if 3 in class_ids:
        yolk_idx = (class_ids == 3)
    if 4 in class_ids:
        eye_idx = (class_ids == 4)
    # else:
    #     egg_idx = None

    # return egg_idx, embryo_idx, yolk_idx, eye_idx
    return egg_idx, embryo_idx, yolk_idx, eye_idx


# %% Main
def analyse_folder(folder: str, image_folder: str, image_out_folder: str, show_images: bool, write_images: bool, write_csv: bool) -> None:
    scale = 287.0
    date, treatment = folder.split('/')[-2:]
    image_out_folder = os.path.join(image_out_folder, date, treatment)
    log_path = os.path.join(folder, '{}_{}_measurements_log.csv'.format(date, treatment))
    # log_path = os.path.join(folder, 'measurements_log.csv')

    files = [f for f in os.listdir(folder) if
             (os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1] != '.csv') and os.path.getsize(os.path.join(folder, f)) > 408]
    files.sort()

    if write_csv:
        write_cod_egg_csv_header(log_path)

    pbar = ProgressBar(widgets=[Percentage(), ' ', Bar(), '   ', Timer(), '   ', AdaptiveETA()])
    for f in pbar(files):
        im = load_image(image_folder, f)
        h, w = im.shape[0:2]

        with open(os.path.join(folder, f), 'rb') as file:
            nn_output = pickle.load(file)

        class_ids = nn_output['class_ids']
        rois = nn_output['rois']
        masks = nn_output['masks']

        egg_idx, embryo_idx, yolk_idx, eye_idx = get_idx_by_class(class_ids)
        # egg_idx, embryo_idx, yolk_idx = get_idx_by_class(class_ids)

        if np.any(egg_idx):
            masks = correct_masks(masks)
            fish = build_fish_associations(masks, egg_idx, embryo_idx, yolk_idx, eye_idx)
            egg_idx, embryo_idx, yolk_idx, eye_idx = check_rois(egg_idx, embryo_idx, yolk_idx, eye_idx, rois, w)
            # egg_idx = remove_overlapping_masks(masks, egg_idx)
            # egg_idx, embryo_idx, yolk_idx, eye_idx = remove_orphans(fish, egg_idx, embryo_idx, yolk_idx, eye_idx)

            # Check that we still have anything left to measure at this point!
            if np.any(egg_idx):
                egg_measurements = measure_egg(masks, egg_idx, scale)
                eye_measurements = measure_eyes(masks, eye_idx, scale)
                yolk_measurements = measure_yolk(masks, yolk_idx, scale)
                embryo_measurements = measure_embryo(masks, embryo_idx, scale)

                if show_images:
                    # im = draw_rois(im, rois, egg_idx, embryo_idx, yolk_idx, eye_idx)
                    im = draw_masks(im, masks, egg_idx, embryo_idx, yolk_idx, eye_idx)
                    im = draw_labels(im, egg_measurements, yolk_measurements,
                                     eye_measurements)  # , body_measurements, yolk_measurements, fish)

                    cv.imshow('Neural Network output: {}'.format(os.path.join(image_folder, f)), im)
                    k = cv.waitKey(0)
                    if k == 98:  # b
                        print('\nEmbryo: {}'.format(os.path.join(image_folder, f)))
                    if k == 103:  # g
                        print('\nEgg: {}'.format(os.path.join(image_folder, f)))
                    if k == 101:  # e
                        print('\nEye: {}'.format(os.path.join(image_folder, f)))
                    if k == 121:  # y
                        print('\nYolk: {}'.format(os.path.join(image_folder, f)))
                    cv.destroyAllWindows()

                if write_images:
                    im = draw_masks(im, masks, egg_idx, embryo_idx, yolk_idx, eye_idx)
                    im = draw_labels(im, egg_measurements, yolk_measurements,
                                     eye_measurements)
                    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

                    im = Image.fromarray(im)
                    if not os.path.isdir(image_out_folder):
                        os.makedirs(image_out_folder)
                    im.save(os.path.join(image_out_folder, f) + '.png')

                if write_csv:
                    write_frame_to_csv(log_path, folder, f, fish, egg_measurements, yolk_measurements, eye_measurements,
                                       embryo_measurements)


def main():
    show_images = False
    write_images = True
    write_csv = True

    image_root_folder = '/media/dave/dave_8tb/2021'
    nn_output_root_folder = '/mnt/6TB_Media/PhD Work/2021_cod/egg_results/egg_results_2188/'

    # dates = ['20210411', '20210412', '20210413', '20210414', '20210415', '20210416', '20210417', '20210418']
    dates = ['20210408', '20210409', '20210410', '20210411', '20210412', '20210413', '20210414', '20210415', '20210416',
             '20210417', '20210418']
    treatments = ['1',
                  '2']  # [, '3', 'DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']

    done = []

    # 2020 reanalysis
    image_root_folder = '/media/dave/dave_8tb/Easter_2020/Bernard/'
    nn_output_root_folder = '/media/dave/DATA/2020_reanalysis/eggs/1151/'
    image_out_folder = '/media/dave/DATA/2020_reanalysis/eggs/1151/images/'

    # dates = ['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410', '20200411', '20200412']
    dates = ['20200408', '20200409', '20200410', '20200411', '20200412', '20200404', '20200405', '20200406', '20200407']
    treatments = ['1', '2', '3', 'DCA-ctrl', 'DCA-ctrl-2', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
    done = ['20200408/1',
        '20200410/DCA-2,50', '20200411/DCA-0,15']

    for date in dates:
        for treatment in treatments:
            nn_output_folder = os.path.join(nn_output_root_folder, date, treatment)
            image_folder = os.path.join(image_root_folder, date, treatment)

            if os.path.isdir(nn_output_folder) and os.path.isdir(image_folder):
                if os.path.join(date, treatment) in done:
                    print('Skipping previously analysed folder {}'.format(nn_output_folder))
                else:
                    print('Analysing {}'.format(nn_output_folder))
                    analyse_folder(nn_output_folder, image_folder, image_out_folder, show_images, write_images, write_csv)
                    print('    ...done')


main()
