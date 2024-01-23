import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from oil_deplete_plots import OilDepletePlots as odp
# import seaborn as sns


def hatch_survival():
    treatments = ['Statfjord-4d-1', 'Statfjord-4d-3',
                  'Statfjord-14d-4',
                  'Statfjord-21d-2',
                  'Statfjord-40d-4',
                  'Statfjord-60d-2',
                  'SW-4d-3',
                  'SW-60d-2', 'SW-60d-3', 'SW-60d-4',
                  'ULSFO-28d-1', 'ULSFO-28d-2',
                  'ULSFO-28d-4',
                  'ULSFO-60d-1', 'ULSFO-60d-2']

    df = pd.read_csv('/mnt/Media/bhh_data/OD_survival and hatch.csv')
    df = df[df['Name'].isin(treatments)]
    df = df.rename(columns={'Name': 'Treatment'})

    old_names = ['Statfjord-4d-1', 'Statfjord-4d-3',
                 'Statfjord-14d-4',
                 'Statfjord-21d-2',
                 'Statfjord-40d-4',
                 'Statfjord-60d-2',
                 'SW-4d-3',
                 'SW-60d-2', 'SW-60d-3', 'SW-60d-4',
                 'ULSFO-28d-1', 'ULSFO-28d-2',
                 'ULSFO-28d-4',
                 'ULSFO-60d-1', 'ULSFO-60d-2']
    new_names = ['Statfjord 4d-1', 'Statfjord 4d-3',
                 'Statfjord 14d-4',
                 'Statfjord 21d-2',
                 'Statfjord 40d-4',
                 'Statfjord 60d-2',
                 'SW 4d (ctrl)',
                 'SW 60d-2', 'SW 60d-3', 'SW 60d-4',
                 'ULSFO 28d-1', 'ULSFO 28d-2',
                 'ULSFO 28d-4',
                 'ULSFO 60d-1', 'ULSFO 60d-2']

    for i in range(len(old_names)):
        df.loc[df['Treatment'] == old_names[i], ['Treatment']] = new_names[i]

    odp.survival_hatching_plots(df, 'Survival')
    odp.survival_hatching_plots(df, 'Hatch')

def load_csvs():
    measurements = pd.read_csv('/mnt/Media/bhh_data/results/2115/20210422/bar/20210422_bar_measurements_log.csv')
    filenames = pd.read_csv('/mnt/Media/bhh_data/Oildeplete_files.csv')

    merged = filenames.merge(measurements, left_on='File_ID', right_on='Image ID')
    merged = merged[merged['Age[DPF]'] == 17]
    merged.drop(columns=['#', 'Date', 'Dev.', 'File_ID', 'Species', 'Age[DPF]', 'Treatment_y', 'Myotome height[mm]'], inplace=True)
    merged.rename(columns={'Treatment_x': 'Treatment'}, inplace=True)
    ## Some images taken at a different scale
    bad_scale = ['ARG_0273',
                 'ARG_0550',
                 'OIL_0170',
                 'OIL_0175',
                 'OIL_0176',
                 'OIL_0177',
                 'OIL_0183',
                 'OIL_0214',
                 'OIL_0215',
                 'OIL_0216',
                 'OIL_0301',
                 'OIL_0388']
    idx = merged[merged['Image ID'].isin(bad_scale)].index
    merged.drop(merged.index[idx], inplace=True)

    manual_results = pd.read_csv('/mnt/Media/bhh_data/Oildeplete_manual_measurements.csv')
    manual_results.drop(columns=['#', 'Species', 'Age[DPF]', 'Myotome height[mm]', 'Eye to front[mm2]'], inplace=True)
    manual_results.rename(columns={'File_ID': 'Image ID'}, inplace=True)

    return merged, manual_results

def compare_individuals(joined, attribute):
    plt.scatter(x='{}_m'.format(attribute), y='{}_a'.format(attribute), s=2, data=joined)

    ax = plt.gca()
    minimum = np.min((ax.get_xlim(), ax.get_ylim()))
    maximum = np.max((ax.get_xlim(), ax.get_ylim()))

    ax.set_xlim(minimum, maximum * 1.1)
    ax.set_ylim(minimum, maximum * 1.1)
    # plt.axis('square')
    plt.plot([minimum, maximum], [minimum, maximum], linestyle='--', c='gray')

    plt.title(attribute)
    plt.xlabel('Manual method')
    plt.ylabel('Automated method')
    plt.tight_layout()
    plt.show()
def main():
    auto, manual = load_csvs()

    joined = auto.join(manual, lsuffix="_a", rsuffix="_m")

    for attribute in ['Myotome length[mm]', 'Eye area[mm2]', 'Body area[mm2]', 'Yolk area[mm2]', 'Yolk fraction']:
        compare_individuals(joined, attribute)
    foo = -1

main()
# hatch_survival()
