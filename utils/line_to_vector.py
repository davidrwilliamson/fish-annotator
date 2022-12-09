import numpy as np
import cv2 as cv
import os


# This and the other utils files are extremely dirty.
# They can be expected to break and/or cause irreparable data loss without warning.
# Use with caution or, better yet, not at all.

# Note that this works with opencv 4.3, not 3.4 !
def load_image(path) -> np.ndarray:
    im = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return im


def im_fill(im: np.ndarray) -> np.ndarray:
    # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    thresh, im_th = cv.threshold(im, 10, 255, cv.THRESH_BINARY)

    # Dilate/erode to get fill small gaps in lines
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(im_th, element)
    eroded = cv.erode(dilated, element)
    im_th = eroded

    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Erode/dilate to get rid of any small specks
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eroded = cv.erode(im_out, element)
    dilated = cv.dilate(eroded, element)
    im_out = dilated

    # Display images.
    # cv.imshow("Thresholded Image", im_th)
    # cv.imshow("Floodfilled Image", im_floodfill)
    # cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    # cv.imshow("Foreground", im_out)
    # cv.imshow("Dilated", dilated)
    # cv.waitKey(0)

    return im_out


def outline_points(im: np.ndarray) -> list:
    points = []
    contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # '[0][:-1])
    if len(contours) >= 1:
        for cnt in contours:
            epsilon = 0.001 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            points.append(approx)
    # hull = cv.convexHull(cnt)

    # Display contours
    # result_borders = np.zeros(im.shape, np.uint8)
    # cv.drawContours(result_borders, contours, -1, 255, 1)
    # cv.drawContours(result_borders, approx, -1, 255, 3)
    # cv.imshow("Points on boundary", result_borders)
    # cv.waitKey(0)

    return points


def output_nn_labels(labels: list):
    first_line = 'filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n'
    out_str = first_line
    for label in labels:
        if len(label) > 1:
            for l in label:
                out_str = out_str + l
        else:
            out_str = out_str + label[0]

    # Need to remove trailing newlines at this point or the NN trainer has problems
    return out_str


def labels_cod(path: str, feature: int, points):
    if feature == '0':
        feature_string = 'body'
    elif feature == '1':
        feature_string = 'eye'
    elif feature == '2':
        feature_string = 'yolk'
    elif feature == '3':
        feature_string = 'body'
    elif feature == '4':
        feature_string = 'egg'
    else:
        raise

    region_count = len(points)
    if region_count > 1:
        regions_strings = []
        for region_id in range(len(points)):
            r_s = format_region_string(path, region_count, region_id, points[region_id], feature_string)
            regions_strings.append(r_s)
    else:
        if len(points) == 0:
            # Empty annotation file for some reason
            print(path)
        else:
            regions_strings = [format_region_string(path, 1, 0, points[0], feature_string)]

    return regions_strings


def format_region_string(path: str, region_count: int, region_id: int, polygon: np.ndarray, region_type: str) -> str:
    str_file_info = format_file_info_string(path)
    str_polygon = format_polygon_string(polygon)
    str_region = '"{{""region"": ""{}""}}"'.format(region_type)

    str_out = '{},{},{},{},{}\n'.format(str_file_info, region_count, region_id, str_polygon, str_region)

    return str_out


def format_file_info_string(path: str) -> str:
    filename = os.path.split(path)[1]
    filename = '{}_full.png'.format(
        filename.split('_')[0])  # Hack because filenames of image don't match annotation filename
    file_size = os.stat(path).st_size
    str_out = '{},{},{{}}'.format(filename, file_size)

    return str_out


def format_polygon_string(polygon: np.ndarray) -> str:
    str_start = '"{""name"": ""polygon"", '
    str_end = '}"'
    polygon = polygon.reshape(polygon.shape[0], 2)
    x = polygon[:, 0].tolist()
    y = polygon[:, 1].tolist()
    str_coords = '""all_points_x"": {}, ""all_points_y"": {}'.format(x, y)
    str_out = '{}{}{}'.format(str_start, str_coords, str_end)

    return str_out


def main():
    labels = []
    root = '/media/dave/dave_8tb/2021/'
    # dates = ['202104{:02d}'.format(i) for i in range(8, 26)]
    egg_folders = ['20210408/2', '20210408/3', '20210409/1', '20210410/1', '20210410/2', '20210410/3', '20210411/1',
                   '20210411/2', '20210412/1', '20210413/1', '20210414/1', '20210415/1', '20210415/2', '20210416/1',
                   '20210416/2', '20210416/3', '20210417/1', '20210417/2', '20210417/3', '20210418/1', '20210419/1',
                   '20210420/1']
    larva_folders = ['20210418/2', '20210421/1', '20210420/1', '20210422/1', '20210422/2', '20210422/sw1_1',
                     '20210422/sw1_2', '20210422/sw3_1', '20210422/sw3_2', '20210422/sw3_3', '20210422/ulsfo-28-1_1',
                     '20210422/ulsfo-28-1_2', '20210422/ulsfo-28-1_3', '20210423/1', '20210423/2', '20210423/3',
                     '20210423/statfjord-14d-4_2', '20210423/statfjord-14d-4', '20210423/statfjord-21d-2_2',
                     '20210423/statfjord-21d-2', '20210423/statfjord-40d-4_2', '20210423/statfjord-40d-4',
                     '20210423/statfjord-4d-1_2', '20210423/statfjord-4d-1', '20210423/statfjord-4d-3_2',
                     '20210423/statfjord-4d-3', '20210423/statfjord-60d-2_2', '20210423/statfjord-60d-2',
                     '20210423/sw-4d-3', '20210423/sw-60d-2_2', '20210423/sw-60d-2', '20210423/sw-60d-3',
                     '20210423/sw-60d-4_2', '20210423/sw-60d-4', '20210423/ulsfo-28d-1_2', '20210423/ulsfo-28d-1',
                     '20210423/ulsfo-28d-2', '20210423/ulsfo-28d-4_2', '20210423/ulsfo-28d-4', '20210423/ulsfo-60d-1_2',
                     '20210423/ulsfo-60d-1', '20210423/ulsfo-60d-2_2', '20210423/ulsfo-60d-2', '20210424/1',
                     '20210424/2', '20210425/1', '20210425/2', '20210425/statfjord-28d-3', '20210425/statfjord-60d-3',
                     '20210425/sw-4d-1', '20210425/sw-60d-1', '20210425/sw3', '20210425/sw4', '20210425/ulsfo-4d-3']

    # 2020
    root = '/media/dave/dave_8tb/Easter_2020/Bernard/'
    egg_folders = ['20200404/3', '20200405/1', '20200406/1', '20200407/1',
                   '20200408/1', '20200408/DCA-ctrl', '20200408/DCA-0,15', '20200408/DCA-0,31', '20200408/DCA-0,62', '20200408/DCA-1,25', '20200408/DCA-2,50', '20200408/DCA-5,00',
                   '20200409/1', '20200409/DCA-ctrl', '20200409/DCA-0,31', '20200409/DCA-1,25', '20200409/DCA-5,00',
                   '20200410/1', '20200410/DCA-ctrl', '20200410/DCA-0,31', '20200410/DCA-1,25', '20200410/DCA-5,00',
                   '20200411/DCA-ctrl',
                   '20200412/DCA-ctrl-2']

    # for ff in larva_folders:
    for ff in egg_folders:
        folder = os.path.join(root, ff, 'analysis/annotations')
        if os.path.exists(folder):
            # folder = os.path.join('/home/dave/Desktop/more_annotations/still_more', ff)
            # folder = os.path.join('/home/dave/PycharmProjects/fish-annotator/data/cod_eggs/cropped/', ff)
            # folder = os.path.join('/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-2,50/analysis/annotations')
            files = sorted([file for file in os.listdir(folder) if os.path.splitext(file)[1] == '.png'])

            for f in files:
                feature = os.path.splitext(f.split('_')[-1])[0]
                path = os.path.join(folder, f)
                im = load_image(path)
                im_filled = im_fill(im)
                points = outline_points(im_filled)
                label = labels_cod(path, feature, points)
                labels.append(label)

                # # Show processing steps outputs
                # cv.imshow('Raw outline', im)
                # cv.imshow('Filled area of interest', im_filled)
                # points_im = np.zeros(im.shape, np.uint8)
                # for p in points:
                #     cv.drawContours(points_im, p, -1, 255, 3)
                # cv.imshow("Points on boundary", points_im)
                # cv.waitKey()
    out_text = output_nn_labels(labels)
    _ = None


main()
