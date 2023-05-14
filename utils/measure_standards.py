import os
import cv2 as cv
import numpy as np


def list_files(folder) -> list:
    files = [f for f in os.listdir(folder) if
             (os.path.isfile(os.path.join(folder, f))
             and (os.path.splitext(f)[1] == '.silc'))]

    return files


def load_image(image_folder: str, f: str) -> np.ndarray:
    if os.path.splitext(f)[-1] in ['.silc']:
        im = np.load(os.path.join(image_folder, f + '.silc')).astype(np.uint8).squeeze()
        im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)

    return im


def find_circles(im_file: str) -> np.ndarray:
    try:
        im = np.load(im_file).astype('uint8').squeeze()
    except ValueError:
        print ('Error loading {}'.format(im_file))
        return None
    im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)
    im = cv.medianBlur(im, 7)
    g_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(g_im, cv.HOUGH_GRADIENT, 1, minDist=100,
                              param1=70, param2=50, minRadius=20, maxRadius=50)
    return circles


def preview_measurements(circles: np.ndarray, im_file: str) -> None:
    im = np.load(im_file).astype('uint8').squeeze()
    im = cv.cvtColor(im, cv.COLOR_BAYER_BG2BGR)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(im, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(im, (i[0], i[1]), 2, (0, 0, 255), 3)
            # label the circle with size
            cv.putText(im, text=str(i[2]), org=(i[0], i[1]), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))

    cv.imshow(im_file, im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def circle_stats(detected_circs: dict):
    c_list = []
    for value in detected_circs.values():
        for c in value:
            c_list.append(c)
    c_arr = np.asarray(c_list)
    # Mean and std dev. before pixel size conversion
    mean = c_arr.mean()
    std = c_arr.std()

    outliers = []
    for key, value in detected_circs.items():
        for c in value:
            if abs(mean - c) > 2 * std:
                outliers.append(key)

    c_arr *= 2.  # Radius to diameter
    c_arr /= 287.  # Pixel size

    return c_arr.mean(), c_arr.std(), c_arr.min(), c_arr.max(), outliers


def exclude_bad_measurements(detected_circs: dict) -> dict:
    # Manually checked outliers. Seems like a bit of dirt got caught in the tube :(
    bad = {'D20200407T132711.570214.silc': 0,
           'D20200407T132651.516578.silc': 1,
           'D20200407T132625.099503.silc': 0,
           'D20200407T132650.638763.silc': 4,
           'D20200407T132707.064384.silc': 1,
           'D20200407T132710.199448.silc': 0,
           'D20200407T132702.583352.silc': 0,
           'D20200407T132712.200099.silc': 4,
           'D20200407T132648.899996.silc': 0,
           'D20200407T132706.187204.silc': 0,
           'D20200407T132701.827936.silc': 1,
           'D20200407T132702.081063.silc': 5,
           'D20200407T132654.828681.silc': 8,
           'D20200407T132657.455660.silc': 5,
           'D20200407T132707.190169.silc': 0,
           'D20200407T132715.065650.silc': 0,
           'D20200407T132649.861030.silc': 0,
           'D20200407T132658.458443.silc': 2,
           'D20200407T132705.192714.silc': 0,
           'D20200407T132709.975272.silc': 0,
           'D20200407T132652.645264.silc': 2,
           'D20200407T132711.697215.silc': 0,
           'D20200407T132706.312793.silc': 2,
           'D20200407T132654.576930.silc': 4,
           'D20200407T132658.836530.silc': 3,
           'D20200407T132712.326139.silc': 2}
    for key, value in bad.items():
        try:
            if len(detected_circs[key]) == 1:
                del detected_circs[key]
            else:
                old = detected_circs[key]
                new = np.delete(old, [value])
                detected_circs[key] = new
        except KeyError:
            continue

    return detected_circs


def view_outliers():
    # folder = '/media/dave/dave_8tb1/Easter_2020/Bernard/20200404/standard/'
    # outliers = ['D20200404T140314.619119.silc', 'D20200404T140314.619119.silc', 'D20200404T140312.811351.silc',
    #             'D20200404T140315.163131.silc', 'D20200404T140314.727931.silc', 'D20200404T140314.050178.silc',
    #             'D20200404T140308.007814.silc', 'D20200404T140315.925848.silc', 'D20200404T140315.817094.silc',
    #             'D20200404T140314.510231.silc', 'D20200404T140312.920281.silc']

    folder = '/media/dave/dave_8tb1/Easter_2020/Bernard/20200407/230um_standard/'
    outliers = ['D20200407T132711.570214.silc', 'D20200407T132651.516578.silc', 'D20200407T132625.099503.silc',
                'D20200407T132650.638763.silc', 'D20200407T132652.142663.silc', 'D20200407T132700.712859.silc',
                'D20200407T132707.064384.silc', 'D20200407T132710.199448.silc', 'D20200407T132702.583352.silc',
                'D20200407T132712.200099.silc', 'D20200407T132712.200099.silc', 'D20200407T132648.899996.silc',
                'D20200407T132706.187204.silc', 'D20200407T132701.827936.silc', 'D20200407T132702.081063.silc',
                'D20200407T132654.828681.silc', 'D20200407T132657.455660.silc', 'D20200407T132707.190169.silc',
                'D20200407T132715.065650.silc', 'D20200407T132653.138392.silc', 'D20200407T132649.861030.silc',
                'D20200407T132658.458443.silc', 'D20200407T132705.067388.silc', 'D20200407T132705.192714.silc',
                'D20200407T132709.975272.silc', 'D20200407T132652.645264.silc', 'D20200407T132711.697215.silc',
                'D20200407T132706.312793.silc', 'D20200407T132653.015552.silc', 'D20200407T132654.576930.silc',
                'D20200407T132658.836530.silc', 'D20200407T132712.326139.silc', 'D20200407T132706.560124.silc']

    for f in outliers:
        im_file = os.path.join(folder, f)
        circles = find_circles(im_file)
        if circles is not None:
            print('{}: {}'.format(f, circles[0, :][:, 2]))
            preview_measurements(circles, im_file)


def main():
    # folders = ['/media/dave/dave_8tb1/Easter_2020/Bernard/20200404/standard/', '/media/dave/dave_8tb1/Easter_2020/Bernard/20200407/230um_standard/']
    folders = ['/media/dave/dave_8tb1/Easter_2020/Bernard/20200407/230um_standard/']
    for folder in folders:
        files = list_files(folder)
        detected_circs = {}
        for f in files:
            im_file = os.path.join(folder, f)
            circles = find_circles(im_file)
            if circles is not None:
                detected_circs[f] = circles[0, :][:, 2]
                # preview_measurements(circles, im_file)
        detected_circs = exclude_bad_measurements(detected_circs)
        stats = circle_stats(detected_circs)
        print('{}: {}'.format(folder, stats[:4]))
        print('{} outliers: {}'.format(len(stats[4]), stats[4]))


main()
# view_outliers()
