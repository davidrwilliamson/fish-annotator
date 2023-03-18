import os
import cv2 as cv
import numpy as np


def load_image(file: str) -> np.ndarray:
    im = np.load(file).astype('uint8').squeeze()
    im = cv.cvtColor(im, cv.COLOR_BAYER_RG2RGB)  # cv2.COLOR_BAYER_BG2RGB = 48

    return im


def save_image(im: np.ndarray, out_folder: str, im_file: str):
    cv.imwrite(os.path.join(out_folder, im_file) + '_full.png', im)


def main():
    # root = '/media/dave/dave_8tb/2021/'
    root = '/media/dave/dave_8tb/Easter_2020/Bernard'
    # dates = ['202104{:02d}'.format(i) for i in range(8, 26)]
    egg_folders = []
    larva_folders = []
    # '20210408/1', '20210422/2', '20210423/sw-60d-2', '20210408/2', '20210422/sw1_1', '20210423/sw-60d-3', '20210408/3',
    # folders = ['20210422/sw1_2', '20210423/sw-60d-4_2', '20210409/1', '20210422/sw3_1',
    #            '20210423/sw-60d-4', '20210410/1', '20210422/sw3_2', '20210423/ulsfo-28d-1_2', '20210410/2',
    #            '20210422/sw3_3', '20210423/ulsfo-28d-1', '20210410/3', '20210422/ulsfo-28-1_1', '20210423/ulsfo-28d-2',
    #            '20210411/1', '20210422/ulsfo-28-1_2', '20210423/ulsfo-28d-4_2', '20210411/2', '20210422/ulsfo-28-1_3',
    #            '20210423/ulsfo-28d-4', '20210412/1', '20210423/1', '20210423/ulsfo-60d-1_2', '20210413/1', '20210423/2',
    #            '20210423/ulsfo-60d-1', '20210414/1', '20210423/3', '20210423/ulsfo-60d-2_2', '20210415/1',
    #            '20210423/statfjord-14d-4_2', '20210423/ulsfo-60d-2', '20210415/2', '20210423/statfjord-14d-4',
    #            '20210424/1', '20210416/1', '20210423/statfjord-21d-2_2', '20210424/2', '20210416/2',
    #            '20210423/statfjord-21d-2', '20210425/1', '20210416/3', '20210423/statfjord-40d-4_2', '20210425/2',
    #            '20210417/1', '20210423/statfjord-40d-4', '20210425/statfjord-28d-3', '20210417/2',
    #            '20210423/statfjord-4d-1_2', '20210425/statfjord-60d-3', '20210417/3', '20210423/statfjord-4d-1',
    #            '20210425/sw3', '20210418/1', '20210423/statfjord-4d-3_2', '20210425/sw4', '20210418/2',
    #            '20210423/statfjord-4d-3', '20210425/sw-4d-1', '20210419/1', '20210423/statfjord-60d-2_2',
    #            '20210425/sw-60d-1', '20210420/1', '20210423/statfjord-60d-2', '20210425/ulsfo-4d-3', '20210421/1',
    #            '20210423/sw-4d-3', '20210422/1',
    #            '20210423/sw-60d-2_2']
    folders = ['20200404/3', '20200405/1', '20200406/1', '20200407/1',]

    for ff in folders:
        folder = os.path.join(root, ff)
        out_folder = os.path.join(root, ff, 'png_exports/full')

        if os.path.exists(folder):
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            # files = sorted([file for file in os.listdir(folder) if os.path.splitext(file)[1] == '.silc_bayer'])
            files = sorted([file for file in os.listdir(folder) if os.path.splitext(file)[1] == '.silc'])
            print("Writing {} images to {}".format(len(files), out_folder))

            for im_file in files:
                im = load_image(os.path.join(folder, im_file))
                save_image(im, out_folder, im_file)


main()
