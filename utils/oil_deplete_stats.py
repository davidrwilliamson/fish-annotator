import pandas as pd
import numpy as np
import os
from oil_deplete_plots import OilDepletePlots as odp


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


def prepare_dataframe(df: pd.DataFrame, root_folder: str, dates, treatments) -> pd.DataFrame:
    for d in dates:
        for t in treatments:
            dir_path = os.path.join(root_folder, d, t)
            if os.path.isdir(dir_path):
                df = exclude_bad_measurements(df, dir_path)
                pass

    df = manual_exclusions(df)  # Stuff that I missed in checking and added after

    # Set datetime from Image ID
    df['Date'] = df['Image ID'].apply(
        lambda x: '{}-{}-{}{}:{}:{}'.format(x[1:5], x[5:7], x[7:9], x[9:12], x[12:14], x[14:]))
    df.rename(columns={'Date': 'DateTime'}, inplace=True)
    # Drop rows that are just the column names (from concat in load_csv_files) and reindex
    df = df.drop(df.index[df['Image ID'] == 'Image ID'].tolist()).reset_index(drop=True)
    # Drop unused columns
    # df = df.drop(columns=['Dev.', 'Myotome height[mm]', 'Image ID', 'DateTime', 'Fish ID'])
    df = df.drop(columns=['Dev.', 'Myotome height[mm]', 'DateTime'])
    # Collapse multiple runs of the same treatment together
    # df.loc[df['Treatment'] == 2, ['Treatment']] = 1
    # df.loc[df['Treatment'] == 3, ['Treatment']] = 1
    df['Treatment'].replace(to_replace='_[0-9]$', value='', regex=True, inplace=True)

    df = additional_calcs(df)
    df = df.rename(columns={'Body area[mm2]': 'Total body area[mm2]', 'Myotome length[mm]': 'Standard length[mm]'})

    # Change column types from object to appropriate numericals (required for plotting)
    type_dict = {'Treatment': str,
                 'Total body area[mm2]': float,
                 'Structural body area[mm2]': float,
                 'Standard length[mm]': float,
                 'Eye area[mm2]': float,
                 'Eye min diameter[mm]': float,
                 'Eye max diameter[mm]': float,
                 'Yolk area[mm2]': float,
                 'Yolk length[mm]': float,
                 'Yolk height[mm]': float,
                 'Yolk fraction': float
                 }
    df = df.astype(type_dict)

    return df


def additional_calcs(df: pd.DataFrame) -> pd.DataFrame:
    df['Body area[mm2]'] = df['Body area[mm2]'].astype(float)
    df['Yolk area[mm2]'] = df['Yolk area[mm2]'].astype(float)
    df['Structural body area[mm2]'] = df['Body area[mm2]'] - df['Yolk area[mm2]']

    return df


def exclude_bad_measurements(df: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    note_file = os.path.join(folder_path, 'bad_measurements.csv')
    if os.path.isfile(note_file):
        notes = np.loadtxt(note_file, delimiter=',', skiprows=1, dtype=str)

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
                df.loc[rows.index, 'Yolk fraction'] = np.nan

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


def manual_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    # Exclude 20210425/sw3 due to huge bubble sitting in images messing up measurements

    return df


def prepare_manual_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df.loc[df['Treatment'] == 'SW-1'].index)
    df = df.drop(df.loc[df['Treatment'] == 'SW-3'].index)
    # df = df.drop(columns=['#', 'Species', 'Age[DPF]', 'Myotome height[mm]', 'File_ID', 'Eye to front[mm2]'])
    df = df.drop(columns=['#', 'Species', 'Age[DPF]', 'Myotome height[mm]', 'Eye to front[mm2]'])
    df = additional_calcs(df)
    df.rename(columns={'Myotome length[mm]': 'Standard length[mm]', 'Body area[mm2]': 'Total body area[mm2]'}, inplace=True)

    return df


def process_2021() -> tuple:
    # root_folder = '/mnt/6TB_Media/PhD Work/2021_cod/larvae_results'
    root_folder = '/media/dave/DATA/2115'
    dates = ['20210423']#, '20210425']
    treatments = ['statfjord-4d-1', 'statfjord-4d-1_2', 'statfjord-4d-3', 'statfjord-4d-3_2',
                  'statfjord-14d-4', 'statfjord-14d-4_2',
                  'statfjord-21d-2', 'statfjord-21d-2_2',
                  'statfjord-40d-4', 'statfjord-40d-4_2',
                  'statfjord-60d-2', 'statfjord-60d-2_2',
                  'sw-4d-3',
                  'sw-60d-2', 'sw-60d-2_2', 'sw-60d-3', 'sw-60d-4', 'sw-60d-4_2',
                  'ulsfo-28d-1', 'ulsfo-28d-1_2', 'ulsfo-28d-2', 'ulsfo-28d-2_2',
                  'ulsfo-28d-4', 'ulsfo-28d-4_2',
                  'ulsfo-60d-1', 'ulsfo-60d-1_2', 'ulsfo-60d-2', 'ulsfo-60d-2_2',
                  # 'statford-28d-3',
                  # 'statford-60d-3',
                  # 'sw4',
                  # 'sw-4d-1',
                  # 'sw-60d-1',
                  # 'ulsfo-4d-3',
                  # 'ulsfo-60d-3'
                  ]

    return root_folder, dates, treatments


def rename_treatments(df: pd.DataFrame) -> pd.DataFrame:
    old_names = ['statfjord-4d-1', 'statfjord-4d-3',
                 'statfjord-14d-4',
                 'statfjord-21d-2',
                 'statfjord-40d-4',
                 'statfjord-60d-2',
                 'sw-4d-3',
                 'sw-60d-2', 'sw-60d-3', 'sw-60d-4',
                 'ulsfo-28d-1', 'ulsfo-28d-2',
                 'ulsfo-28d-4',
                 'ulsfo-60d-1', 'ulsfo-60d-2']
    new_names = ['Statfjord 4d-1', 'Statfjord 4d-3',
                 'Statfjord 14d-4',
                 'Statfjord 21d-2',
                 'Statfjord 40d-4',
                 'Statfjord 60d-2',
                 'SW 4d-3',
                 'SW 60d-2', 'SW 60d-3', 'SW 60d-4',
                 'ULSFO 28d-1', 'ULSFO 28d-2',
                 'ULSFO 28d-4',
                 'ULSFO 60d-1', 'ULSFO 60d-2']

    for i in range(len(old_names)):
        df.loc[df['Treatment'] == old_names[i], ['Treatment']] = new_names[i]

    return df


def rename_replicates(df: pd.DataFrame) -> pd.DataFrame:
    old_names = ['Statfjord 4d-1', 'Statfjord 4d-3',
                 'Statfjord 14d-4',
                 'Statfjord 21d-2',
                 'Statfjord 40d-4',
                 'Statfjord 60d-2',
                 'SW 4d-3',
                 'SW 60d-2', 'SW 60d-3', 'SW 60d-4',
                 'ULSFO 28d-1', 'ULSFO 28d-2', 'ULSFO 28d-4',
                 'ULSFO 60d-1', 'ULSFO 60d-2']
    new_names = ['Statfjord 4d-1', 'Statfjord 4d-2',
                 'Statfjord 14d',
                 'Statfjord 21d',
                 'Statfjord 40d',
                 'Statfjord 60d',
                 'SW 4d (ctrl)',
                 'SW 60d-1', 'SW 60d-2', 'SW 60d-3',
                 'ULSFO 28d-1', 'ULSFO 28d-2', 'ULSFO 28d-3',
                 'ULSFO 60d-1', 'ULSFO 60d-2']

    for i in range(len(old_names)):
        df.loc[df['Treatment'] == old_names[i], ['Treatment']] = new_names[i]

    return df


def main() -> None:
    root_folder, dates, treatments = process_2021()
    df = load_csv_files(root_folder, dates, treatments)
    df = prepare_dataframe(df, root_folder, dates, treatments)
    df = rename_treatments(df)

    df_bhh = pd.read_csv('/media/dave/dave_8tb1/2021/Oildeplete - biometry.csv')
    df_bhh = prepare_manual_dataframe(df_bhh)

    df['Method'] = 'Automated'
    df_bhh['Method'] = 'Manual'

    attributes = ['Eye area[mm2]', 'Total body area[mm2]', 'Structural body area[mm2]', 'Standard length[mm]', 'Yolk area[mm2]', 'Yolk fraction']
    treatments = df['Treatment'].unique()
    df_bhh = df_bhh[df_bhh['Treatment'].apply(lambda x: x in treatments)]

    df = rename_replicates(df)
    df_bhh = rename_replicates(df_bhh)

    df_combined = pd.concat([df, df_bhh])

    # odp.count_fish(df)

    # table = odp.prepare_latex_table(df_bhh)
    #
    normalize = False
    # difference_table = odp.prepare_method_difference_table(df, df_bhh, attributes, normalize)
    #
    for attribute in attributes:
    #     pass
    #     # odp.anova(df_bhh, attribute)
    #     # odp.test_replicates(df, attribute)
        odp.plot_measurements_both_methods(df_combined, attribute)
        # odp.plot_cross_compare(df, df_bhh, attribute)
        # odp.plot_compare_difference_from_mean(df, df_bhh, attribute, normalize)
    foo = -1


main()
