import cv2 as cv
from tqdm import tqdm, trange
import os
import numpy as np
from skimage.measure import label, regionprops
from typing import List, Tuple

start, end = 11500, 11533


class ImFolder:

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.num_images = 0
        self._list_files()

        self.current_im = 0

    def _list_files(self) -> None:
        self.files = [f for f in os.listdir(self.folder) if (os.path.splitext(f)[1] == '.bayer_silc'
                                                             or os.path.splitext(f)[1] == '.silc_bayer')]
        self.num_images = len(self.files)

    def load_rgb_im(self, idx: int) -> np.ndarray:
        f = os.path.join(self.folder, self.files[idx])
        bayer_im = np.load(f, allow_pickle=False)
        rgb_im = cv.cvtColor(bayer_im, cv.COLOR_BayerBG2BGR)

        return rgb_im

    def __next__(self) -> Tuple[np.ndarray, str]:
        if self.num_images > self.current_im:
            self.current_im += 1
        else:
            self.current_im = 0

        return self.load_rgb_im(self.current_im), self.files[self.current_im]


class EggExtractor:

    def __init__(self, im_folder: ImFolder) -> None:
        self.im_folder = im_folder
        self.bg_frames = self._background_subtraction()

        # self.min_blob_size = 20000
        self.min_blob_size = 20000
        self.max_blob_size = 80000

    def _background_subtraction(self) -> List[np.ndarray]:
        bg_mog = cv.createBackgroundSubtractorMOG2()
        bg_mog.setDetectShadows(False)

        n = self.im_folder.num_images

        bg_frames = []

        print("Doing background subtraction...")
        for idx in trange(n):
            frame = self.im_folder.load_rgb_im(idx)
            fg_mask = bg_mog.apply(frame, learningRate=0.3)
            bg_frames.append(fg_mask)

        return bg_frames

    def _background_blobber(self, fg_im: np.ndarray) -> list:
        label_im = label(fg_im)
        regions = regionprops(label_im)
        good_regions = []

        if len(regions) > 1:
            for r in regions:
                if self.min_blob_size < r.area: # < self.max_blob_size:
                    good_regions.append(r)

        return good_regions

    def extract_interesting_images(self) -> Tuple[list, list]:
        egg_indices = []
        egg_blobs = []

        print ("Extracting interesting images...")
        for idx in trange(self.im_folder.num_images):
            blobs = self._background_blobber(self.bg_frames[idx])

            if len(blobs) > 0:
                # for blob in blobs:
                #     frame = cv.rectangle(frame, (blob.bbox[1], blob.bbox[0]), (blob.bbox[3], blob.bbox[2]),
                #                          (255, 0, 0), thickness=5)
                # cv.imshow('Blobs: {}'.format(len(blobs)), frame)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                egg_indices.append(idx)
                egg_blobs.append(blobs)

        return egg_indices, egg_blobs


def write_rgb_images(root_folder: str, date_folders: List[str], sub_folders: List[str], im_write: bool) -> None:
    if im_write:
        # Just write out all the images
        print ("Writing out all images...")
        for date_folder in date_folders:
            for sub_folder in sub_folders:
                f = os.path.join(root_folder, date_folder, sub_folder)
                if os.path.exists(f):
                    im_folder = ImFolder(f)
                    for i in trange(im_folder.num_images):
                        im = im_folder.load_rgb_im(i)
                        filename = os.path.splitext(im_folder.files[i])[0]

                        if not os.path.isdir(os.path.join(root_folder, date_folder, 'all_images', sub_folder)):
                            os.makedirs(os.path.join(root_folder, date_folder, 'all_images', sub_folder))

                        cv.imwrite(os.path.join(root_folder, date_folder, 'all_images', sub_folder, filename + '.png'), im)


def process_images(root_folder: str, date_folders: List[str], sub_folders: List[str], im_write: bool, im_show: bool) -> None:
    # Do background subtraction etc.
    for date_folder in date_folders:
        for sub_folder in sub_folders:
            f = os.path.join(root_folder, date_folder, sub_folder)
            if os.path.exists(f):
                print("\nProcessing {}".format(os.path.join(root_folder, date_folder, sub_folder)))
                im_folder = ImFolder(f)

                egg_extractor = EggExtractor(im_folder)
                indices, blobs = egg_extractor.extract_interesting_images()

                if im_write:
                    print("Writing out image files...")

                for i, e in zip(indices, blobs):
                    im = im_folder.load_rgb_im(i)
                    filename = os.path.splitext(im_folder.files[i])[0]
                    # for blob in e:
                    #     im = cv.rectangle(im, (blob.bbox[1], blob.bbox[0]), (blob.bbox[3], blob.bbox[2]), (255, 0, 0), thickness=5)

                    if not os.path.isdir(os.path.join(root_folder, date_folder, 'images', sub_folder)):
                        os.makedirs(os.path.join(root_folder, date_folder, 'images', sub_folder))

                    if im_write:
                        cv.imwrite(os.path.join(root_folder, date_folder, 'images', sub_folder, filename + '.png'), im)

                    if im_show:
                        cv.imshow('{}'.format(filename), im)
                        cv.waitKey(0)
                        cv.destroyAllWindows()

                del im_folder, egg_extractor, indices, blobs


def main(im_write: bool = False, im_show: bool = True) -> None:
    # root_folder = '/media/dave/dave_8tb/2021/'
    root_folder = '/media/dave/dave_8tb/2021'
    # date_folders = ['20210409', '20210410', '20210411', '20210412', '20210413', '20210414', '20210415']
    date_folders = ['20210425']

    # sub_folders = ['1', '2', '3']
    # sub_folders = ['{0:02d}'.format(i) for i in range(11, 16)]
    # sub_folders = ['1', '2', '3', '4', 'sw1_1', 'sw1_2', 'sw3_1', 'sw3_2', 'sw3_3', 'ulsfo-28-1_1', 'ulsfo-28-1_2', 'ulsfo-28-1_3']

    sub_folders = []
    for date_folder in date_folders:
        contents = os.listdir(os.path.join(root_folder, date_folder))
        for c in contents:
            if c not in ['images', 'all_images']:
                sub_folders.append(c)

    sub_folders = list(set(sub_folders))

    process_images(root_folder, date_folders, sub_folders, im_write, im_show)
    # write_rgb_images(root_folder, date_folders, sub_folders, im_write)


main(im_write=True, im_show=False)
