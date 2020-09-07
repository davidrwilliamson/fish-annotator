## Script to crop annotations and raw images based on annotation RoIs

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def main(annotation_folder, image_folder):
    ann_files = [os.path.join(annotation_folder, f) for f in sorted(os.listdir(annotation_folder)) if f.endswith('.png')]
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith('.png')]

    fnames = {}

    for imf in image_files:
        fname = os.path.basename(imf).split('_')[0]
        fnames[fname] = []

    for anf in ann_files:
        fname = os.path.basename(anf).split('_')[0]
        fnames[fname].append(anf)

    for fname in fnames:
        bbox = find_bb(fname, fnames)
        # draw_preview(fname, fnames, bbox)
        export_crop(fname, fnames, bbox)


def export_crop(fname, fnames, bbox):
    ann_crop_folder = '/home/dave/PycharmProjects/fish-annotator/data/cod/annotations_cropped/'
    im_crop_folder = '/home/dave/PycharmProjects/fish-annotator/data/cod/images_cropped/'

    for im_file in fnames[fname]:
        im = cv.imread(im_file, cv.IMREAD_UNCHANGED)
        cropped_im = im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        new_fname = os.path.join(ann_crop_folder, os.path.basename(im_file))
        cv.imwrite(new_fname, cropped_im)

    raw_im_file = os.path.join(image_folder, '{}{}'.format(fname, '_full.png'))
    raw_im = cv.imread(raw_im_file, cv.IMREAD_UNCHANGED)
    cropped_raw_im = raw_im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    new_raw_fname = os.path.join(im_crop_folder, os.path.basename(raw_im_file))
    cv.imwrite(new_raw_fname, cropped_raw_im)
    # plt.imshow(cropped_im)
    # plt.show()


def find_bb(fname, fnames):
    x0_i = y0_i = x1_i = y1_i = -1
    max_x = max_y = 0

    for im_file in fnames[fname]:
        im = cv.imread(im_file)
        grey = cv.cvtColor(im, cv.COLOR_RGBA2GRAY)
        max_x = grey.shape[1]
        max_y = grey.shape[0]
        _, contours, _ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contours_poly = [None] * len(contours)
        bb = [None] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            bb[i] = cv.boundingRect(contours_poly[i])

        # Get largest bounding box of current image
        x0_j = y0_j = x1_j = y1_j = -1
        for b in bb:
            if x0_j == -1:
                x0_j = b[0]
            if y0_j == -1:
                y0_j = b[1]
            if b[0] < x0_j:
                x0_j = b[0]
            if b[1] < y0_j:
                y0_j = b[1]
            if b[0] + b[2] > x1_j:
                x1_j = b[0] + b[2]
            if b[1] + b[3] > y1_j:
                y1_j = b[1] + b[3]
        # print('Annotation {}\n    TL: ({}, {}), BR: ({}, {})'.format(os.path.basename(im_file), x0_j, y0_j, x1_j, y1_j))

        # Expand bounding box to cover all annotations for this image
        if x0_i == -1:
            x0_i = x0_j
        if y0_i == -1:
            y0_i = y0_j
        if x0_j < x0_i:
            x0_i = x0_j
        if y0_j < y0_i:
            y0_i = y0_j
        if x1_j > x1_i:
            x1_i = x1_j
        if y1_j > y1_i:
            y1_i = y1_j

    # print ('Largest bounding box\n    TL: ({}, {}), BR: ({}, {})'.format(x0_i, y0_i, x1_i, y1_i))

    margin = 20
    x0_i = max(x0_i - margin, 0)
    y0_i = max(y0_i - margin, 0)
    x1_i = min(x1_i + margin, max_x)
    y1_i = min(y1_i + margin, max_y)

    bbox = [x0_i, y0_i, x1_i, y1_i]

    return bbox


def draw_preview(images, fnames, bbox):
    im = cv.imread(fnames[images][0])
    drawing = np.zeros((im.shape[0], im.shape[1]))

    for im_file in fnames[images]:
        im = cv.imread(im_file)
        grey = cv.cvtColor(im, cv.COLOR_RGBA2GRAY)
        drawing[grey > 0] = (255)

    cv.rectangle(drawing, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    plt.imshow(drawing)
    plt.show()


annotation_folder = '/home/dave/PycharmProjects/fish-annotator/data/cod/annotations'
image_folder = '/home/dave/PycharmProjects/fish-annotator/data/cod/images'

main(annotation_folder, image_folder)
