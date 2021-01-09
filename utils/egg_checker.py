import pandas as pd
import numpy as np
import os
import cv2 as cv


def prepare_dataframe(csv_file):
    sep = ';'
    col_names = ['folder', 'file', 'circles']

    df = pd.read_csv(csv_file, sep=sep, names=col_names)
    # Remove brackets & whitespace
    df['circles'] = df['circles'].str.replace('[', '')
    df['circles'] = df['circles'].str.replace(']', '')
    df['circles'] = df['circles'].str.replace(' ', '')
    df['folder'] = df['folder'].str.replace(' ', '')
    df['file'] = df['file'].str.replace(' ', '')

    # Convert to list of floats
    df['circles'] = df['circles'].str.split(',')
    df['circles'] = df['circles'].apply(lambda x: list(map(float, x)))
    # Put each circle in its own list and then put lists on separate lines
    df['circles'] = df['circles'].apply(lambda x: [x[i:i + 3] for i in range(0, len(x), 3)])
    df = df.explode('circles')

    # Separate out into circle centre position and radius
    df['pos'] = df['circles'].apply(lambda x: x[0:2])
    df['radius'] = df['circles'].apply(lambda x: x[2])
    df = df.drop(columns=['circles'])

    # Create Date and Treatment columns based on folder
    df['Date'] = df['folder'].apply(lambda x: x.split('/')[0])
    df['Treatment'] = df['folder'].apply(lambda x: x.split('/')[1])
    # Collapse identical treatments with different names into one
    df.loc[df['Treatment'] == '2', ['Treatment']] = '1'
    df.loc[df['Treatment'] == '3', ['Treatment']] = '1'
    df.loc[df['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl'

    df = drop_bad_measurements(df)

    return df


def drop_bad_measurements(df):
    # Hand checked all egs with radius > 205
    df = df[df['radius'] <= 205]
    # Hand checked all egs with radius < 185
    df = df[df['radius'] >= 180] # These are all bad
    # These are 180 < radius < 185 that are bad
    checked_bad = [6, 58, 74, 207, 229, 363, 1750, 2516, 2750, 2876, 3266, 3319, 3399, 3401, 3464, 3523, 3526, 3552,
                   3573, 3593, 3619, 3629, 3632, 3637, 4522, 4705, 4881, 4883, 5254, 5316, 5413, 5437, 5568, 5594, 5600,
                   5988, 7690, 7778, 8304, 8344, 8648, 9167, 10531, 10933, 11978, 12009, 12018, 12046, 12137, 12574,
                   13144, 13242, 13387, 13774, 13936, 14021, 14114, 14188, 14298]
    # These are 185 <= radius < 187 that are bad
    checked_bad.extend([44, 71, 172, 190, 927, 1144, 1751, 2376, 2381, 2834, 2957, 3318, 3352, 3361, 3388, 3422, 3463,
                        3478, 3609, 3736, 3753, 3807, 4097, 4288, 4520, 4521, 5039, 5068, 5737, 5796, 5938, 6269, 7082,
                        7341, 7852, 8114, 8560, 8649, 8835, 9190, 9589, 9772, 9783, 10245, 10295, 10599, 10793, 11181,
                        11474, 11534, 11655, 11861, 11969, 11973, 12038, 12576, 12786, 12864, 13567, 13664, 13710])
    # Bad radius > 203
    checked_bad.extend([164, 1326, 1456, 1846, 2031, 2611, 2801, 3455, 3507, 3655, 3672, 4079, 4145, 4547, 5237, 5237,
                        5604, 6140, 7095, 7589, 7946, 8302, 8462, 8591, 9066, 10069, 10123, 10701, 10736, 11759, 11897,
                        11979, 12004, 12344, 13507, 14161, 14288])
    # Bad 203 >= radius > 197 in 20200404-20200407
    checked_bad.extend([34, 48, 65, 107, 114, 123, 211, 233, 242, 243, 249, 266, 329])
    # Bad 190 < radius <= 187 in 20200404-20200407
    checked_bad.extend([53, 57, 79, 148, 150, 185, 188, 191, 209, 260, 262, 268, 294, 355, 387, 399, 499, 524, 573, 604])
    df = df.drop(index=checked_bad)

    # Exclude circles where centre is less than a radius from the screen edge
    # (i.e., the circle extends off the edge of the screen)
    df = df.loc[(df['pos'].apply(lambda x: x[0]) - df['radius'] >= 0) &
           (df['pos'].apply(lambda x: x[0]) + df['radius'] <= 2448)]

    return df


def run_checks(df):
    good_circs = []
    bad_circs = []

    root_folder = '/media/dave/SINTEF Polar Night D/Easter cod experiments/Bernard'
    png_folder = 'png_exports/full'

    colour = (0, 0, 255)

    counter = 1
    n_rows = df.shape[0]
    for idx, row in df.iterrows():
        im_path = os.path.join(root_folder, row['folder'], png_folder, row['file'])
        img = cv.imread(im_path)
        x, y = int(row['pos'][0]), int(row['pos'][1])
        r = int(row['radius'])
        cv.circle(img, (x, y), r, colour, 2)

        cv.imshow('{}/{}'.format(counter, n_rows), img)
        k = cv.waitKey(0)

        if k == 81:  # swipe left
            good_circs.append(idx)
        elif k == 83:  # swipe right
            bad_circs.append(idx)

        cv.destroyAllWindows()
        counter += 1

    return good_circs, bad_circs


def main():
    circle_file = '/home/dave/Desktop/circle_stats.csv'
    df = prepare_dataframe(circle_file)

    check = df[((df['Date'] == '20200404') |
                (df['Date'] == '20200405') |
                (df['Date'] == '20200406') |
                (df['Date'] == '20200407')) &
               (df['radius'] < 190) &
               (df['radius'] >= 187)
               ]

    # hi_good, hi_bad = run_checks(hi_check)
    good, bad = run_checks(check)

    # run_checks(df)

    foo = -1


main()
