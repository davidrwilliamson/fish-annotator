#!/usr/bin/env python
import numpy as np
import cv2
import re
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


class Annotation:

    def __init__(self):
        pass


class ImageSet:

    def __init__(self, folder):
        self.folder = folder
        self._im_files = []
        self._im_annotations = []
        self._current_file = 0

        self._list_image_files()
        self._list_annotations()

    # def load_folder(self):
    #     im_files = self.get_image_files(self.folder)
    #
    #     return im_files

    def _list_image_files(self):
        rois_filename = os.path.join(self.folder, 'RoIs')

        if os.path.isfile(rois_filename):
            rois_file = open(rois_filename, 'rt')
        else:
            print('{}: RoIs file missing.'.format(self.folder))
            return

        im_files = []
        for line in rois_file.readlines():
            f = re.match('^filename:', line)
            if f:
                fn = line.split(': ')[1].strip()
                fn = os.path.join(self.folder, fn)
                im_files.append(fn)

        self._im_files = im_files

    def _list_annotations(self):
        self._im_annotations = np.empty(self.num_images, dtype=Annotation)
        annotations_filename = os.path.join(self.folder, 'Annotations')

        if os.path.isfile(annotations_filename):
            annotations_file = open(annotations_filename, 'r')
        else:
            print('{}: Annotations file missing.'.format(self.folder))
            return

    @property
    def num_images(self):
        return len(self._im_files)

    @property
    def curr_frame_num(self):
        return self._current_file + 1

    @property
    def curr_image(self):
        return self._im_files[self._current_file]

    def next_image(self):
        self._current_file = (self._current_file + 1) % len(self._im_files)
        return self.curr_image

    def prev_image(self):
        self._current_file = (self._current_file - 1) % len(self._im_files)
        return self.curr_image


class PygView(object):

    def __init__(self, width=1080, height=720, fps=60):
        """Initialize pygame, window, background, font,...
        """
        pygame.init()
        pygame.display.set_caption("Press ESC to quit")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.image = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.playtime = 0.0
        self.font = pygame.font.SysFont('mono', 20, bold=True)

    def run(self):
        """The main loop
        """
        running = True

        folder = '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-0,31/'
        im_set = ImageSet(folder)
        self.load_image(im_set.curr_image)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_LEFT:
                        self.load_image(im_set.prev_image())
                    if event.key == pygame.K_RIGHT:
                        self.load_image(im_set.next_image())

            milliseconds = self.clock.tick(self.fps)
            self.playtime += milliseconds / 1000.0

            self.screen.blit(self.image, (0, 0))
            self.draw_text('FPS: {:6.3}, File: {} of {}'.format(
                self.clock.get_fps(), im_set.curr_frame_num, im_set.num_images))
            pygame.display.update()

        pygame.quit()

    def draw_text(self, text):
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render(text, True, (0, 255, 0))
        self.screen.blit(surface, (self.width - fw - fh, self.height - 2 * fh))

    def load_image(self, filename):
        im = np.load(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)

        im = pygame.surfarray.make_surface(np.uint8(im))
        im = pygame.transform.flip(im, False, True)
        im = pygame.transform.rotate(im, -90)
        im = pygame.transform.scale(im, (self.width, self.height))
        im = im.convert()

        self.image = im


if __name__ == '__main__':
    PygView().run()
