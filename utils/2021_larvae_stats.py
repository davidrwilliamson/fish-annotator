import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import ptitprince as pt


def load_csv_files(root_folder, dates, treatments) -> pd.DataFrame:
    master_df = None
    csv_file = 'measurements_log.csv'

    for d in dates:
        for t in treatments:
            dir_path = os.path.join(root_folder, d, t)
            csv_path = os.path.join(dir_path, '{}_{}_{}'.format(d, t, csv_file))
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                df['Date'] = df['Date'].astype(str)

                if master_df is None:
                    master_df = df
                else:
                    master_df = pd.concat([master_df, df], ignore_index=True)

    return master_df


def prepare_dataframe(df, root_folder, dates, treatments) -> pd.DataFrame:
    for d in dates:
        for t in treatments:
            dir_path = os.path.join(root_folder, d, t)
            if os.path.isdir(dir_path):
                df = exclude_bad_measurements(df, dir_path)

    # Set datetime from Image ID
    df['Date'] = df['Image ID'].apply(
        lambda x: '{}-{}-{}{}:{}:{}'.format(x[1:5], x[5:7], x[7:9], x[9:12], x[12:14], x[14:]))
    df.rename(columns={'Date': 'DateTime'}, inplace=True)
    # Drop rows that are just the column names (from concat in load_csv_files) and reindex
    df = df.drop(df.index[df['Image ID'] == 'Image ID'].tolist()).reset_index(drop=True)
    # Drop unused columns
    df = df.drop(columns=['Dev.', 'Myotome height[mm]'])
    # df = df.drop(columns=['Image ID', 'Dev.', 'Myotome height[mm]'])
    # Collapse multiple runs of the same treatment together
    df.loc[df['Treatment'] == 2, ['Treatment']] = 1
    df.loc[df['Treatment'] == 3, ['Treatment']] = 1
    df['Treatment'].replace(to_replace='_[0-9]$', value='', regex=True, inplace=True)

    df = additional_calcs(df)
    df = df.rename(columns={'Yolk fraction': 'Yolk fraction (area)'})

    # Change column types from object to appropriate numericals (required for plotting)
    type_dict = {'DateTime': 'datetime64',
                 'Treatment': str,
                 'Fish ID': int,
                 'Body area[mm2]': float,
                 'Total body volume[mm3]': float,
                 'Structural body volume[mm3]': float,
                 'Myotome length[mm]': float,
                 'Eye area[mm2]': float,
                 'Eye min diameter[mm]': float,
                 'Eye max diameter[mm]': float,
                 'Eye volume[mm3]': float,
                 'Yolk area[mm2]': float,
                 'Yolk length[mm]': float,
                 'Yolk height[mm]': float,
                 'Yolk volume[mm3]': float,
                 'Yolk fraction (area)': float,
                 'Yolk fraction (volume)': float}
    df = df.astype(type_dict)

    return df


def additional_calcs(df):
    df['Total body volume[mm3]'] = ((df['Body area[mm2]'] * df['Body area[mm2]']) / df['Myotome length[mm]']) \
        .apply(lambda x: x * np.pi * 0.25)
    # Ellipsoid: 4/3 pi a^2 c
    a = df['Yolk height[mm]'] * 0.5
    c = df['Yolk length[mm]'] * 0.5
    df['Yolk volume[mm3]'] = np.pi * (4. / 3.) * a * a * c
    # Structural volume
    df['Structural body volume[mm3]'] = df['Total body volume[mm3]'] - df['Yolk volume[mm3]']
    # Yolk fraction by volume
    df['Yolk fraction (volume)'] = df['Yolk volume[mm3]'] / df['Total body volume[mm3]']
    # Eye volume
    estimated_r = (df['Eye min diameter[mm]'] + df['Eye max diameter[mm]']) / 2.
    df['Eye volume[mm3]'] = np.pi * (4. / 3.) * (estimated_r ** 3)

    return df


def exclude_bad_measurements(df, folder_path) -> pd.DataFrame:
    note_file = os.path.join(folder_path, 'bad_measurements.csv')
    if os.path.isfile(note_file):
        notes = np.loadtxt(note_file, delimiter=', ', skiprows=1, dtype=str)

        for note in notes:
            idx = df[df['Image ID'] == note[0]].index
            if note[1].astype(bool):  # eye
                df.loc[idx, 'Eye area[mm2]'] = np.nan
                df.loc[idx, 'Eye min diameter[mm]'] = np.nan
                df.loc[idx, 'Eye max diameter[mm]'] = np.nan
            if note[2].astype(bool):  # body
                df.loc[idx, 'Body area[mm2]'] = np.nan
                df.loc[idx, 'Myotome length[mm]'] = np.nan
                df.loc[idx, 'Yolk fraction'] = np.nan
            if note[3].astype(bool):  # yolk
                df.loc[idx, 'Yolk area[mm2]'] = np.nan
                df.loc[idx, 'Yolk length[mm]'] = np.nan
                df.loc[idx, 'Yolk height[mm]'] = np.nan
                df.loc[idx, 'Yolk fraction'] = np.nan
            if note[4].astype(bool):  # recheck
                # df.drop(index=idx)
                pass

    recheck_file = os.path.join(folder_path, 'rechecked_measurements.csv')
    if os.path.isfile(recheck_file):
        rc_df = pd.read_csv(recheck_file, dtype={'Image ID': str, 'Fish ID': int, 'Body area[mm]': float, 'Eye min diameter[mm]': float, 'Body bad': bool, 'Eye bad': bool, 'Yolk bad': bool})
        bad_bodies = rc_df.loc[rc_df[rc_df['Body bad'] == True].index]
        for i in bad_bodies.index:
            rows = df[(df['Image ID'] == bad_bodies['Image ID'][i]) &
                      (df['Fish ID'] == bad_bodies['Fish ID'][i]) &
                      (df['Body area[mm2]'] == bad_bodies['Body area[mm2]'][i])]
            df.loc[rows.index, 'Body area[mm2]'] = np.nan
            df.loc[rows.index, 'Myotome length[mm]'] = np.nan
            df.loc[rows.index, 'Yolk fraction[mm2]'] = np.nan

        bad_eyes = rc_df.loc[rc_df[rc_df['Eye bad'] == True].index]
        for i in bad_eyes.index:
            rows = df[(df['Image ID'] == bad_eyes['Image ID'][i]) &
                      (df['Fish ID'] == bad_eyes['Fish ID'][i]) &
                      (df['Eye min diameter[mm]'] == bad_eyes['Eye min diameter[mm]'][i])]
            df.loc[rows.index, 'Eye area[mm2]'] = np.nan
            df.loc[rows.index, 'Eye min diameter[mm]'] = np.nan
            df.loc[rows.index, 'Eye max diameter[mm]'] = np.nan
        bad_yolks = rc_df.loc[rc_df[rc_df['Yolk bad'] == True].index]
        for i in bad_yolks.index:
            rows = df[(df['Image ID'] == bad_yolks['Image ID'][i]) &
                      (df['Fish ID'] == bad_yolks['Fish ID'][i]) &
                      (df['Yolk area[mm2]'] == bad_yolks['Yolk area[mm2]'][i])]
            df.loc[rows.index, 'Yolk area[mm2]'] = np.nan
            df.loc[rows.index, 'Yolk length[mm]'] = np.nan
            df.loc[rows.index, 'Yolk height[mm]'] = np.nan
            df.loc[rows.index, 'Yolk fraction'] = np.nan

    else:
        print('No bad_measurements.csv for {}'.format(folder_path))

    return df


# def plot_attribute_treatment_date(df: pd.DataFrame, date: str, treatment: str, attribute: str):
#     var = df.loc[(df['DateTime'].dt.date == date) & (df['Treatment'] == treatment)]
#
#     figsize = (9, 9)
#     f, ax = plt.subplots(figsize=figsize)
#
#     ax = sns.swarmplot(y='Myotome length[mm]')
#
#     plt.show()


def plot_date_attribute(df: pd.DataFrame, date: str, attribute: str):
    var = df.loc[(df['DateTime'].dt.date == pd.to_datetime(date))]

    figsize = (12, 12)
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.swarmplot(x='Treatment', y=attribute, data=var, orient='v')

    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()


def plot_treatment_attribute(df: pd.DataFrame, treatment: str, attribute: str):
    var = df.loc[(df['Treatment'] == treatment)]
    dates = df['DateTime'].dt.date

    figsize = (12, 12)
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.swarmplot(x=dates, y=attribute, data=var, orient='v')

    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()


def boxplot_by_treatment(df: pd.DataFrame, date: str, attribute: str):
    data = df.loc[(df['DateTime'].dt.date == pd.to_datetime(date))]
    x = 'Treatment'
    y = attribute

    ax = sns.boxplot(x=x, y=y, data=data)
    pairs = [('DCA-ctrl', 'DCA-0,15'), ('DCA-ctrl', 'DCA-0,31'), ('DCA-ctrl', 'DCA-0,62'), ('DCA-ctrl', 'DCA-1,25'), ('DCA-ctrl', 'DCA-2,50')]

    annotator = Annotator(ax, pairs, data=data, x=x, y=y)
    annotator.configure(test='Kruskal', text_format='star', loc='inside')
    _, results = annotator.apply_and_annotate()

    plt.tight_layout()
    plt.show()


def swarmplot_by_treatment_all_dates(df: pd.DataFrame, attribute: str):
    data = df.loc[(df['DateTime'].dt.date <= pd.to_datetime('20200417'))]
    x = 'Treatment'
    y = attribute
    hue = df['DateTime'].dt.date

    figsize = (12, 12)
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, dodge=True)

    ax.set_title('Larvae - {}'.format(attribute))

    plt.tight_layout()
    plt.show()


def boxplot_by_treatment_all_dates(df: pd.DataFrame, attribute: str):
    data = df.loc[(df['DateTime'].dt.date <= pd.to_datetime('20200417'))]
    x = 'Treatment'
    y = attribute
    hue = df['DateTime'].dt.date

    figsize = (12, 12)
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data)
    pairs = [('DCA-ctrl', 'DCA-0,15'), ('DCA-ctrl', 'DCA-0,31'), ('DCA-ctrl', 'DCA-0,62'), ('DCA-ctrl', 'DCA-1,25'), ('DCA-ctrl', 'DCA-2,50')]

    annotator = Annotator(ax, pairs, data=data, x=x, y=y)
    annotator.configure(test='Kruskal', text_format='star', loc='inside', comparisons_correction='bonferroni')
    _, results = annotator.apply_and_annotate()

    # Show number of observations
    n_obs = data.loc[data[attribute].notna()].groupby([x, hue]).size().values
    n_obs = [str(x) for x in n_obs.tolist()]

    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    i = 0
    for median in lines[4:len(lines)-1:lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # 'n: ' label on left side
        if i == 0:
            ax.text(x - 0.2,
                    -0.02,
                    'n:',
                    ha='center', va='center',
                    size='x-small', fontweight='semibold', color='k')
        # Number of observations below each bar
        ax.text(x,
                -0.02,
                n_obs[i],
                ha='center', va='center',
                size='x-small', fontweight='semibold', color='k')
        i += 1

    ax.set_title('Larvae - {}'.format(attribute))

    plt.tight_layout()
    plt.show()


def main():
    root_folder, dates, treatments = reprocess_2020()

    df = load_csv_files(root_folder, dates, treatments)
    df = prepare_dataframe(df, root_folder, dates, treatments)

    # plot_date_attribute(df, '20200413', 'Eye area[mm2]')

    # boxplot_by_treatment_all_dates(df, 'Eye area[mm2]')
    # boxplot_by_treatment_all_dates(df, 'Yolk fraction (area)')
    # boxplot_by_treatment_all_dates(df, 'Yolk area[mm2]')
    # boxplot_by_treatment_all_dates(df, 'Body area[mm2]')

    swarmplot_by_treatment_all_dates(df, 'Eye area[mm2]')
    swarmplot_by_treatment_all_dates(df, 'Yolk fraction (area)')
    swarmplot_by_treatment_all_dates(df, 'Yolk area[mm2]')
    swarmplot_by_treatment_all_dates(df, 'Body area[mm2]')

    foo = -1


def process_2021():
    # root_folder = '/mnt/6TB_Media/PhD Work/2021_cod/larvae_results'
    root_folder = '/media/dave/DATA/2115'
    dates = ['20210419', '20210420', '20210421', '20210422', '20210423', '20210424', '20210425']
    treatments = ['1', '2', '3', 'sw1_1', 'sw1_2', 'sw3_1', 'sw3_2', 'sw3_3', 'ulsfo-28d-1_1', 'ulsfo-28d-1_2',
                  'ulsfo-28d-1_3', 'statfjord-4d-1', 'statfjord-4d-1_2', 'statfjord-4d-3', 'statfjord-4d-3_2',
                  'statfjord-14d-4', 'statfjord-14d-4_2', 'statfjord-21d-2', 'statfjord-21d-2_2', 'statfjord-40d-4',
                  'statfjord-40d-4_2', 'statfjord-60d-2', 'statfjord-60d-2_2', 'sw-4d-3', 'sw-60d-2', 'sw-60d-2_2',
                  'sw-60d-3', 'sw-60d-4', 'sw-60d-4_2', 'ulsfo-28d-1', 'ulsfo-28d-1_2', 'ulsfo-28d-2', 'ulsfo-28d-2_2',
                  'ulsfo-28d-4', 'ulsfo-28d-4_2', 'ulsfo-60d-1', 'ulsfo-60d-1_2', 'ulsfo-60d-2', 'ulsfo-60d-2_2',
                  'statfjord-28d-3', 'statfjord-60d-3', 'sw3', 'sw4', 'sw-4d-1', 'sw-60d-1', 'ulsfo-4d-3',
                  'ulsfo-60d-3']

    return root_folder, dates, treatments


def reprocess_2020():
    root_folder = '/media/dave/DATA/2020_reanalysis/larvae/2115'
    dates = ['20200412', '20200413', '20200414', '20200415', '20200416', '20200417']
    # treatments = ['1', '2', 'DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
    treatments = ['DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50']

    return root_folder, dates, treatments


main()