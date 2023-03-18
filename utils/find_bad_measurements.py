import os
import pandas as pd
import cv2 as cv
import numpy as np


def main():
    stage = 'eggs'
    if stage == 'larvae':
        root_folder = '/media/dave/DATA/2020_reanalysis/larvae/2115/'
        dates = ['20200412', '20200413', '20200414', '20200415', '20200416', '20200417']
        treatments = ['DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50']
        done = ['20200416/DCA-ctrl', '20200416/DCA-0,15', '20200416/DCA-0,31', '20200416/DCA-0,62']
    elif stage == 'eggs':
        root_folder = '/media/dave/DATA/2020_reanalysis/eggs/1151/'
        dates = ['20200412']
        # treatments = ['DCA-ctrl', 'DCA-ctrl-2', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
        treatments = ['DCA-2,50']
        done = ['']

    run(root_folder, stage, dates, treatments, done)
    # run_recheck(root_folder, stage, dates, treatments, done)


def run_recheck(root_folder, stage, dates, treatments, done):
    def display_im():
        # cv.destroyAllWindows()
        im = ims[current_im_num]
        im_file = os.path.join(root_folder, 'images', date, treatment, im + '.png')
        img = cv.imread(im_file)
        info_string = '{}/{}/{}, {} of {}'.format(date, treatment, im, current_im_num + 1, num_images)
        img = update_text(img, info_string)
        try:
            cv.imshow('Measurement Checker', img)
        except Exception as err:
            print('Problem showing {}\n Error: {}'.format(os.path.join(root_folder, date, treatment, im), err))

    def update_text(img, info_string):
        font = cv.FONT_HERSHEY_PLAIN
        font_size = 2
        font_colour = (127, 127, 127)
        line_weight = 2
        pos_x = 2000
        pos_y = 850
        line_offset = 25

        this_frame = df[df['Image ID'] == ims[current_im_num]]
        # num_to_check = this_frame.loc[:, ['Body area[mm2]', 'Eye min diameter[mm]', 'Yolk area[mm2]']].count().sum()
        current_check = 0
        text_lines = []
        for n, fish_id in enumerate(this_frame['Fish ID'].unique()):
            if stage == 'larvae':
                bodies = this_frame[this_frame['Fish ID'] == fish_id]['Body area[mm2]'].dropna()
            elif stage == 'eggs':
                embryos = this_frame[this_frame['Fish ID'] == fish_id]['Embryo area[mm2]'].dropna()
                eggs = this_frame[this_frame['Fish ID'] == fish_id]['Egg min diameter[mm]'].dropna()
            else:
                raise RuntimeWarning('Invalid life stage selected')
            eyes = this_frame[this_frame['Fish ID'] == fish_id]['Eye area[mm2]'].dropna()
            yolks = this_frame[this_frame['Fish ID'] == fish_id]['Yolk area[mm2]'].dropna()

            text_lines.append('Fish {}'.format(n + 1))
            if stage == 'larvae':
                for body in zip(bodies.index, bodies):
                    text_lines.append('  Body {:.3f} ({}): {}'.format(body[1], chr(97 + current_check), df[df.index == body[0]]['Body bad'].values[0]))
                    current_check += 1
            elif stage == 'eggs':
                for embryo in zip(embryos.index, embryos):
                    text_lines.append('  Embryo {:.3f} ({}): {}'.format(embryo[1], chr(97 + current_check), df[df.index == embryo[0]]['Embryo bad'].values[0]))
                    current_check += 1
                for egg in zip(eggs.index, eggs):
                    text_lines.append('  Egg {:.3f} ({}): {}'.format(egg[1], chr(97 + current_check), df[df.index == egg[0]]['Egg bad'].values[0]))
                    current_check += 1
            else:
                raise RuntimeWarning('Invalid life stage selected')
            for eye in zip(eyes.index, eyes):
                text_lines.append('  Eye {:.4f} ({}): {}'.format(eye[1], chr(97 + current_check), df[df.index == eye[0]]['Eye bad'].values[0]))
                current_check += 1
            for yolk in zip(yolks.index, yolks):
                text_lines.append('  Yolk {:.3f} ({}): {}'.format(yolk[1], chr(97 + current_check), df[df.index == yolk[0]]['Yolk bad'].values[0]))
                current_check += 1

        img = cv.putText(img, info_string,
                         (20, 30),
                         font, font_size, font_colour, line_weight, cv.LINE_AA)
        pos_y -= line_offset * len(text_lines)
        for i, line in enumerate(text_lines):
            img = cv.putText(img, line,
                             (pos_x, pos_y + (i * line_offset)),
                             font, font_size, font_colour, line_weight, cv.LINE_AA)

        return img

    def write_to_file(df):
        df = df.drop(columns=['Egg max diameter[mm]', 'Egg area[mm2]',
                              'Eye max diameter[mm]', 'Eye area[mm2]',
                              'Yolk length[mm]', 'Yolk height[mm]'])
        type_dict = {'Image ID': str,
                     'Fish ID': int,
                     'Embryo area[mm2]': float,
                     'Embryo bad': bool,
                     'Eye min diameter[mm]': float,
                     'Eye bad': bool,
                     'Yolk area[mm2]': float,
                     'Yolk bad': bool,
                     'Egg min diameter[mm]': float,
                     'Egg bad': bool}
        df = df.astype(type_dict)
        df.to_csv(os.path.join(root_folder, date, treatment, 'rechecked_measurements.csv'), sep=',', index=False, )

    for date in dates:
        for treatment in treatments:
            if os.path.join(date, treatment) not in done:
                dir_path = os.path.join(root_folder, date, treatment)
                if os.path.isdir(dir_path):
                    csv_path = os.path.join(dir_path, '{}_{}_{}'.format(date, treatment, 'measurements_log.csv'))
                    df = pd.read_csv(os.path.join(dir_path, csv_path))
                    df = df[df['Treatment'] == treatment]
                    if stage == 'larvae':
                        df = df.drop(columns=['Date', 'Treatment', 'Dev.', 'Myotome length[mm]', 'Myotome height[mm]', 'Eye max diameter[mm]',
                                              'Eye area[mm2]', 'Yolk length[mm]', 'Yolk height[mm]', 'Yolk fraction'])
                        df['Body bad'] = False
                        df['Eye bad'] = False
                        df['Yolk bad'] = False

                        type_dict = {'Image ID': str,
                                     'Fish ID': int,
                                     'Body area[mm2]': float,
                                     'Body bad': bool,
                                     'Eye min diameter[mm]': float,
                                     'Eye bad': bool,
                                     'Yolk area[mm2]': float,
                                     'Yolk bad': bool}
                        df = df.astype(type_dict)
                    elif stage == 'eggs':
                        df = df.drop(columns=['Date', 'Treatment', 'Dev.'])#,
                                              # 'Egg max diameter[mm]', 'Egg area[mm2]',
                                              # 'Eye max diameter[mm]', 'Eye area[mm2]',
                                              # 'Yolk length[mm]', 'Yolk height[mm]'])
                        df['Embryo bad'] = False
                        df['Eye bad'] = False
                        df['Yolk bad'] = False
                        df['Egg bad'] = False

                        type_dict = {'Image ID': str,
                                     'Fish ID': int,
                                     'Embryo area[mm2]': float,
                                     'Embryo bad': bool,
                                     'Eye min diameter[mm]': float,
                                     'Eye bad': bool,
                                     'Yolk area[mm2]': float,
                                     'Yolk bad': bool,
                                     'Egg min diameter[mm]': float,
                                     'Egg bad': bool}
                        # df = df.astype(type_dict)
                    else:
                        raise RuntimeWarning('Invalid life stage selected')
                    note_file = os.path.join(dir_path, 'bad_measurements.csv')
                    if os.path.isfile(note_file):
                        notes = np.loadtxt(note_file, delimiter=', ', skiprows=1, dtype=str) #, dtype={'names': ('Image ID', 'eye', 'body', 'yolk', 'recheck'), 'formats': (str, bool, bool, bool, bool)})

                        to_recheck = []
                        for note in notes:
                            image_id = note[0]
                            if note[4].astype(bool):
                                to_recheck.append(image_id)

                        df = df[df['Image ID'].apply(lambda x: x in to_recheck)]

                        current_im_num = 0
                        ims = list(set(df['Image ID'].values))
                        num_images = len(ims)

                        if len(ims) > 0:
                            loop = True
                            display_im()

                            while loop:
                                k = cv.waitKey(0)

                                if k in [81, 83]:
                                    if k == 81:  # left
                                        current_im_num = 0 if num_images == 1 else (current_im_num - 1) % num_images
                                    elif k == 83:  # right
                                        current_im_num = 0 if num_images == 1 else (current_im_num + 1) % num_images
                                    display_im()

                                elif k in range(ord('a'), ord('z') + 1):
                                    this_frame = df[df['Image ID'] == ims[current_im_num]]
                                    # num_to_check = this_frame.loc[:, ['Body area[mm2]', 'Eye min diameter[mm]',
                                    #                                   'Yolk area[mm2]']].count().sum()
                                    current_check = 0
                                    for n, fish_id in enumerate(this_frame['Fish ID'].unique()):
                                        if stage == 'larvae':
                                            bodies = this_frame[this_frame['Fish ID'] == fish_id]['Body area[mm2]'].dropna()
                                        elif stage == 'eggs':
                                            embryos = this_frame[this_frame['Fish ID'] == fish_id][
                                                'Embryo area[mm2]'].dropna()
                                            eggs = this_frame[this_frame['Fish ID'] == fish_id][
                                                'Egg min diameter[mm]'].dropna()
                                        else:
                                            raise RuntimeWarning('Invalid life stage selected')
                                        eyes = this_frame[this_frame['Fish ID'] == fish_id]['Eye min diameter[mm]'].dropna()
                                        yolks = this_frame[this_frame['Fish ID'] == fish_id]['Yolk area[mm2]'].dropna()

                                        if stage == 'larvae':
                                            for body in zip(bodies.index, bodies):
                                                if k == 97 + current_check:
                                                    df.loc[df.index == body[0], 'Body bad'] = ~df.loc[df.index == body[0], 'Body bad']
                                                current_check += 1
                                        elif stage == 'eggs':
                                            for embryo in zip(embryos.index, embryos):
                                                if k == 97 + current_check:
                                                    df.loc[df.index == embryo[0], 'Embryo bad'] = ~df.loc[
                                                        df.index == embryo[0], 'Embryo bad']
                                                current_check += 1
                                            for egg in zip(eggs.index, eggs):
                                                if k == 97 + current_check:
                                                    df.loc[df.index == egg[0], 'Egg bad'] = ~df.loc[
                                                        df.index == egg[0], 'Egg bad']
                                                current_check += 1
                                        else:
                                            raise RuntimeWarning('Invalid life stage selected')
                                        for eye in zip(eyes.index, eyes):
                                            if k == 97 + current_check:
                                                df.loc[df.index == eye[0], 'Eye bad'] = ~df.loc[df.index == eye[0], 'Eye bad']
                                            current_check += 1
                                        for yolk in zip(yolks.index, yolks):
                                            if k == 97 + current_check:
                                                df.loc[df.index == yolk[0], 'Yolk bad'] = ~df.loc[df.index == yolk[0], 'Yolk bad']
                                            current_check += 1
                                    display_im()

                                else:
                                    if k == 27:  # escape
                                        quit()
                                    elif k == 13:  # enter
                                        write_to_file(df)
                                        loop = False
                                write_to_file(df)

                    else:
                        raise RuntimeWarning('No notes file for {}'.format(root_folder))


def run(root_folder, stage, dates, treatments, done):
    def display_im(bad):
        # cv.destroyAllWindows()
        im = ims[current_im_num]
        im_file = os.path.join(root_folder, 'images', date, treatment, im + '.png')
        img = cv.imread(im_file)
        info_string = '{}/{}/{}, {} of {}'.format(date, treatment, im, current_im_num + 1, num_images)
        img = update_text(img, bad, info_string)
        try:
            cv.imshow('Measurement Checker', img)
        except Exception as err:
            print('Problem showing {}\n Error: {}'.format(os.path.join(root_folder, date, treatment, im), err))

    def write_to_file():
        with open(os.path.join(root_folder, date, treatment, 'bad_measurements.csv'), 'w') as f:
            if stage == 'larvae':
                f.write('Image ID, eye, body, yolk, recheck\n')
            if stage == 'eggs':
                f.write('Image ID, eye, body, yolk, recheck, eggs\n')
            for count, note in enumerate(notes):
                f.write('{}, '.format(ims[count]))
                f.write('{}, '.format(note[0].astype(int)))

                f.write('{}, '.format(note[1].astype(int)))
                f.write('{}, '.format(note[2].astype(int)))
                if stage == 'eggs':
                    f.write('{}, '.format(note[3].astype(int)))
                    f.write('{}\n'.format(note[4].astype(int)))
                else:
                    f.write('{}\n'.format(note[3].astype(int)))
        f.close()

    def update_text(img, bad, info_string):
        font = cv.FONT_HERSHEY_PLAIN
        font_size = 2
        font_colour = (127, 127, 127)
        line_weight = 2
        pos_x = 2200
        pos_y = 735
        line_offset = 25
        egg_offset = 0

        img = cv.putText(img, info_string,
                         (20, 30),
                         font, font_size, font_colour, line_weight, cv.LINE_AA)
        img = cv.putText(img, 'Eye: {}'.format(bad[0]),
                         (pos_x, pos_y),
                         font, font_size, font_colour, line_weight, cv.LINE_AA)
        if stage == 'eggs':
            img = cv.putText(img, 'Egg: {}'.format(bad[4]),
                             (pos_x, pos_y + line_offset),
                             font, font_size, font_colour, line_weight, cv.LINE_AA)
            egg_offset = line_offset
        img = cv.putText(img, 'Body: {}'.format(bad[1]),
                         (pos_x, pos_y + line_offset + egg_offset),
                         font, font_size, font_colour, line_weight, cv.LINE_AA)
        img = cv.putText(img, 'Yolk: {}'.format(bad[2]),
                         (pos_x, pos_y + 2 * line_offset + egg_offset),
                         font, font_size, font_colour, line_weight, cv.LINE_AA)
        img = cv.putText(img, 'Recheck: {}'.format(bad[3]),
                         (pos_x, pos_y + 3 * line_offset + egg_offset),
                         font, font_size, font_colour, line_weight, cv.LINE_AA)

        return img

    for date in dates:
        for treatment in treatments:
            if os.path.join(date, treatment) not in done:
                dir_path = os.path.join(root_folder, date, treatment)
                if os.path.isdir(dir_path):
                    csv_path = os.path.join(dir_path, '{}_{}_{}'.format(date, treatment, 'measurements_log.csv'))
                    df = pd.read_csv(os.path.join(dir_path, csv_path))
                    df = df[df['Treatment'] == treatment]
                    # We only care about showing each image once, drop duplicates
                    df = df.drop_duplicates(subset='Image ID', keep='first')

                    current_im_num = 0
                    ims = df['Image ID'].values
                    num_images = df.shape[0]

                    note_file = os.path.join(dir_path, 'bad_measurements.csv')
                    if os.path.isfile(note_file):
                        if stage == 'larvae':
                            notes = np.loadtxt(note_file, usecols=(1, 2, 3, 4), delimiter=', ', skiprows=1, dtype=bool)
                        elif stage == 'eggs':
                            notes = np.loadtxt(note_file, usecols=(1, 2, 3, 4, 5), delimiter=', ', skiprows=1, dtype=bool)
                    else:
                        if stage == 'larvae':
                            notes = np.zeros((num_images, 4), dtype=bool)
                        elif stage == 'eggs':
                            notes = np.zeros((num_images, 5), dtype=bool)

                    loop = True
                    display_im(notes[current_im_num])

                    while loop:
                        k = cv.waitKey(0)

                        if k in [101, 103, 98, 121, 114, 97, 81, 83]:
                            if k == 101:  # e
                                notes[current_im_num][0] = 1 - notes[current_im_num][0]
                            elif k == 98:  # b
                                notes[current_im_num][1] = 1 - notes[current_im_num][1]
                            elif k == 121:  # y
                                notes[current_im_num][2] = 1 - notes[current_im_num][2]
                            elif k == 103:  # g
                                notes[current_im_num][4] = 1 - notes[current_im_num][4]
                            elif k == 114:  # r
                                for i in range(3):
                                    notes[current_im_num][i] = 0
                                if stage == 'eggs':
                                    notes[current_im_num][4] = 0
                                notes[current_im_num][3] = 1 - notes[current_im_num][3]
                            elif k == 97:  # a
                                notes[current_im_num][0:3] = 1 - notes[current_im_num][0:3]
                                if stage == 'eggs':
                                    notes[current_im_num][4] = 1 - notes[current_im_num][4]
                                notes[current_im_num][3] = 0
                                current_im_num = (current_im_num + 1) % num_images
                            elif k == 81:  # left
                                current_im_num = (current_im_num - 1) % num_images
                            elif k == 83:  # right
                                current_im_num = (current_im_num + 1) % num_images
                            display_im(notes[current_im_num])
                        else:
                            if k == 27:  # escape
                                quit()
                            elif k == 13:  # enter
                                write_to_file()
                                loop = False
                        write_to_file()


main()
