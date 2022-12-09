import cv2 as cv
from tqdm import tqdm, trange
import os
import numpy as np
from skimage.measure import label, regionprops
from typing import List, Tuple


class ImFolder:

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.num_images = 0

        self.errors = []

        self.batch_size = 2000
        self.batches = []
        self.num_batches = 0
        self.current_batch_num = -1
        self.current_batch = ()

        self._list_files()

        self.current_im = 0

    def _check_errors(self, files) -> None:
        print('Checking for errors...')
        error_file = os.path.join(self.folder, 'errors.txt')
        if os.path.exists(error_file):
            with open(error_file, 'r') as err_file:
                for line in err_file.readlines():
                    self.errors.append(line.strip())
        else:
            with open(error_file, 'w') as err_file:
                for f in tqdm(files):
                    try:
                        np.load(os.path.join(self.folder, f), allow_pickle=False)
                    except ValueError:
                        self.errors.append(f)
                        err_file.write('{}\n'.format(f))

    def _list_files(self) -> None:
        self.files = [f for f in os.listdir(self.folder) if (os.path.splitext(f)[1] == '.silc'
                                                             or os.path.splitext(f)[1] == '.bayer_silc'
                                                             or os.path.splitext(f)[1] == '.silc_bayer')]
        self._check_errors(self.files)
        self.files = [f for f in self.files if not f in self.errors]
        self.num_images = len(self.files)
        self._batches()

    def _batches(self) -> None:
        n_batches = self.num_images // self.batch_size
        self.batches = [(i * self.batch_size, (i+1) * self.batch_size) for i in range(n_batches)]
        r = self.num_images % self.batch_size
        if r != 0:
            self.batches.append((n_batches * self.batch_size, n_batches * self.batch_size + r))
            n_batches += 1

        self.num_batches = n_batches

    def next_batch(self) -> None:
        self.current_batch_num += 1
        self.current_batch = self.batches[self.current_batch_num]

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
        self.min_blob_size = 1000
        self.max_blob_size = 80000

    def _background_subtraction(self) -> List[np.ndarray]:
        bg_mog = cv.createBackgroundSubtractorMOG2()
        bg_mog.setDetectShadows(False)

        n = self.im_folder.num_images

        bg_frames = []

        print("Doing background subtraction...")
        for idx in trange(self.im_folder.current_batch[0], self.im_folder.current_batch[1]):
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
        for idx in trange(self.im_folder.current_batch[1] - self.im_folder.current_batch[0]):
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
                    for _ in im_folder.batches:
                        im_folder.next_batch()
                        for i in trange(im_folder.current_batch[0], im_folder.current_batch[1]):
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

                for _ in im_folder.batches:
                    im_folder.next_batch()
                    egg_extractor = EggExtractor(im_folder)
                    indices, blobs = egg_extractor.extract_interesting_images()

                    if im_write:
                        print("Writing out image files...")

                    for i, e in tqdm(zip(indices, blobs)):
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
    root_folder = '/media/dave/dave_8tb/Svalbard/'
    date_folders = ['20210831']
    # date_folders = ['20210425']

    # sub_folders = ['1', '2', '3']
    # sub_folders = ['{0:02d}'.format(i) for i in range(11, 16)]
    # sub_folders = ['1', '2', '3', '4', 'sw1_1', 'sw1_2', 'sw3_1', 'sw3_2', 'sw3_3', 'ulsfo-28-1_1', 'ulsfo-28-1_2', 'ulsfo-28-1_3']

    sub_folders = ['Profiler']
    # for date_folder in date_folders:
    #     contents = os.listdir(os.path.join(root_folder, date_folder))
    #     for c in contents:
    #         if c not in ['images', 'all_images']:
    #             sub_folders.append(c)
    #
    # sub_folders = list(set(sub_folders))

    process_images(root_folder, date_folders, sub_folders, im_write, im_show)
    # write_rgb_images(root_folder, date_folders, sub_folders, im_write)


main(im_write=True, im_show=False)