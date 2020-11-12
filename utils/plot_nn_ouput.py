import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

                if master_df is None:
                    master_df = df
                else:
                    master_df = pd.concat([master_df, df], ignore_index=True)

    return master_df

def plot_attribute(df, dx, dy, title):
    figsize = (9, 9)
    ort = 'v'
    pal = sns.color_palette('Set2')
    # pal = pal[0]
    sigma = 0.2

    f, ax = plt.subplots(figsize=figsize)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=sigma, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)
    ax = pt.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                      size=3, jitter=1, zorder=0, orient=ort, move=0.25)
    # ax = sns.swarmplot(x=dx, y=dy, data=df, color='gray', zorder=15, orient=ort)
    ax = sns.boxplot(x=dx, y=dy, data=df, palette=pal, width=.15, zorder=10,
                     showcaps=True, boxprops={'linewidth': 1, "zorder": 10},
                     showfliers=False, whiskerprops={'linewidth': 1, "zorder": 10},
                     saturation=1, orient=ort)
    ax.set_title(title)
    plt.show()

def larvae_grid(df: pd.DataFrame):
    dx = 'Myotome length[mm]'
    pal = sns.color_palette('Set2')
    sigma = 0.2

    g = sns.FacetGrid(df, row="Date", col="Treatment", margin_titles=True)
    g.map_dataframe(pt.half_violinplot, x=dx, data=df, palette=pal, bw=sigma, cut=0.,
                            scale="area", width=.6, inner=None)
    g.map(pt.stripplot, x=dx, data=df, palette=pal, edgecolor="white",
                      size=3, jitter=1, zorder=0,  move=0.25)
    g.map(sns.boxplot, x=dx, data=df, palette=pal, width=.15, zorder=10,
                     showcaps=True, boxprops={'linewidth': 1, "zorder": 10},
                     showfliers=False, whiskerprops={'linewidth': 1, "zorder": 10},
                     saturation=1)
    g.set_axis_labels(dx, '')
    g.fig.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.show()

def main():
    bhh = pd.read_csv('/home/dave/cod_results/BHH/bhh_larvae.csv')

    treatments = ['1', '2', '3', 'DCA-ctrl', 'DCA-ctrl-2', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']

    root_folder_larvae = '/home/dave/cod_results/uncropped/1246/'
    dates_larvae = ['20200413', '20200414', '20200415', '20200416', '20200417']

    df = load_csv_files(root_folder_larvae, dates_larvae, treatments)

    df = remove_bad_measurements(df)

    small_lengths = df[df['Myotome length[mm]'] < 2]
    short_eyes = df[df['Eye min diameter[mm]'] < 0.2]

    # with open('/home/dave/cod_results/uncropped/1246/outliers', 'w') as file:
    #     for i,  row in short_eyes.iterrows():
    #         im_path = os.path.join(str(row['Date']), str(row['Treatment']), str(row['Image ID']))
    #         file.write('{}\n'.format(im_path))

    df.loc[df['Treatment'] == 2, ['Treatment']] = 1
    df.loc[df['Treatment'] == 3, ['Treatment']] = 1
    df.loc[df['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl'

    plot_attribute(df, 'Date', 'Eye area[mm2]', '')
    plot_attribute(df, 'Date', 'Eye min diameter[mm]', '')
    # plot_attribute(df, 'Date', 'Yolk area[mm2]', '')
    # plot_attribute(df, 'Date', 'Yolk length[mm]', '')
    # plot_attribute(df, 'Date', 'Yolk height[mm]', '')
    # plot_attribute(df, 'Date', 'Myotome length[mm]', '')
    # plot_attribute(df, 'Date', 'Body area[mm2]', '')

    # larvae_grid(df)

    # root_folder_eggs = '/home/dave/cod_results/cod_eggs/0735/'
    # dates_eggs = ['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410']

    # eggs = load_csv_files(root_folder_eggs, dates_eggs, treatments)
    # eggs.loc[eggs['Treatment'] == 2, ['Treatment']] = 1
    # eggs.loc[eggs['Treatment'] == 3, ['Treatment']] = 1
    # eggs.loc[eggs['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl
    #
    # eggs = eggs[eggs['Egg diameter[mm]'] > 1.25]
    # eggs = eggs[eggs['Egg diameter[mm]'] < 1.45]
    # # plot_attribute(eggs[eggs['Treatment'] == 1], 'Date', 'Egg diameter[mm]', 'Egg diameters with treatment 1')
    # plot_attribute(eggs[eggs['Date'] == 20200408], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200408')
    # plot_attribute(eggs[eggs['Date'] == 20200409], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200409')
    # plot_attribute(eggs[eggs['Date'] == 20200410], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200410')

def remove_bad_measurements(df):
    # Body areas larger than 2 are all spurious
    bad_areas = df[df['Body area[mm2]'] > 2]
    df = set_body_nan(df, bad_areas.index)

    small_area_good = ['D20200414T142011.993450',
                       'D20200413T142801.406841',
                       'D20200413T160635.374999',
                       'D20200413T160924.212674',
                       'D20200413T160952.505343',
                       'D20200414T144735.181881',
                       'D20200414T144904.507240',
                       'D20200414T150506.006251',
                       'D20200415T163748.466583',
                       'D20200415T171436.764387',
                       'D20200416T134037.555950'
                       ]

    small_areas = df[df['Body area[mm2]'] < 0.9]
    bad_rows = [i for i, row in small_areas.iterrows() if row['Image ID'] not in small_area_good]
    df = set_body_nan(df, bad_rows)

    large_area_good = ['D20200413T132121.816669',
            'D20200413T132332.825405',
            'D20200413T134031.665507',
            'D20200413T134038.051070',
            'D20200413T134056.319119',
            'D20200413T134233.675140',
            'D20200413T134328.603213',
            'D20200413T134414.357287',
            'D20200413T142457.237521',
            'D20200413T142512.612858',
            'D20200413T142720.258141',
            'D20200413T151207.639213',
            'D20200413T151209.813316',
            'D20200413T155125.505895',
            'D20200413T155136.710633',
            'D20200413T155243.068021',
            'D20200414T140026.362917',
            'D20200415T154651.422208',
            'D20200417T144818.260228'
            ]

    large_areas = df[df['Body area[mm2]'] > 1.8]
    bad_rows = [i for i, row in large_areas.iterrows() if row['Image ID'] not in large_area_good]
    df = set_body_nan(df, bad_rows)

    # Body lengths larger than 7 are all spurious
    bad_areas = df[df['Myotome length[mm]'] > 7]
    df = set_body_nan(df, bad_areas.index)

    large_lengths_good = ['D20200414T150328.756238',
            'D20200415T154649.792826',
            'D20200415T160856.307877',
            'D20200416T134046.817531',
            'D20200417T152522.705000',
            'D20200417T154418.597554'
            ]

    large_lengths = df[df['Myotome length[mm]'] > 5.5]
    bad_rows = [i for i, row in large_lengths.iterrows() if row['Image ID'] not in large_lengths_good]
    df = set_body_nan(df, bad_rows)

    large_eye_diams_good = ['D20200413T151122.819588',
            'D20200414T144922.581310',
            'D20200417T151106.196059',
            ]

    large_eye_diams = df[df['Eye min diameter[mm]'] > 0.4]
    bad_rows = [i for i, row in large_eye_diams.iterrows() if row['Image ID'] not in large_eye_diams_good]
    df = set_eye_nan(df, bad_rows)

    # Min eye diameters smaller than 0.05 are all spurious
    bad_areas = df[df['Eye min diameter[mm]'] < 0.05]
    df = set_eye_nan(df, bad_areas.index)

    small_eye_diams_good = ['D20200413T134054.631847',
                    'D20200413T134054.757834',
                    'D20200413T134215.673546',
                    'D20200413T134249.318879',
                    'D20200413T134257.220115',
                    'D20200413T134320.569562',
                    'D20200413T134343.642767',
                    'D20200413T140354.366839',
                    'D20200413T140447.424789',
                    'D20200413T140510.812547',
                    'D20200413T142426.202982',
                    'D20200413T142638.739164',
                    'D20200413T160735.071641',
                    'D20200413T160822.567087',
                    'D20200413T160829.464362',
                    'D20200413T160924.212674',
                    'D20200413T160925.467889',
                    'D20200414T140029.474823',
                    'D20200414T142009.244003',
                    'D20200414T142011.993450',
                    'D20200414T142046.850848',
                    'D20200414T142203.599432',
                    'D20200414T144910.031380',
                    'D20200414T150330.385760',
                    'D20200414T150347.864872',
                    'D20200414T150406.551999',
                    'D20200414T150555.864218',
                    'D20200414T152649.172025',
                    'D20200414T152813.603940',
                    'D20200414T161131.454829',
                    'D20200414T163613.228017',
                    'D20200414T163953.800494',
                    'D20200415T160753.005830',
                    'D20200415T160953.077452',
                    'D20200415T165423.636035',
                    'D20200415T171436.764387',
                    'D20200415T181026.453764',
                    'D20200416T135736.905135',
                    'D20200416T135829.335952',
                    'D20200416T145524.346384',
                    'D20200416T151046.944428',
                    'D20200416T151350.752313',
                    'D20200417T150941.006051',
                    'D20200417T152616.908227',
                    'D20200417T164723.535032',
                    'D20200417T164933.276895',
                    'D20200417T170359.590828',
                    'D20200417T170421.169757',
                    'D20200417T170437.148486',
                    'D20200417T170440.661903'
                    ]

    small_eye_diams = df[df['Eye min diameter[mm]'] < 0.2]
    bad_rows = [i for i, row in small_eye_diams.iterrows() if row['Image ID'] not in small_eye_diams_good]
    df = set_eye_nan(df, bad_rows)

    return df

def set_body_nan(df, idx):
    df.loc[idx, 'Body area[mm2]'] = np.nan
    df.loc[idx, 'Myotome length[mm]'] = np.nan
    df.loc[idx, 'Yolk fraction'] = np.nan

    return df

def set_eye_nan(df, idx):
    df.loc[idx, 'Eye min diameter[mm]'] = np.nan
    df.loc[idx, 'Eye max diameter[mm]'] = np.nan
    df.loc[idx, 'Eye area[mm2]'] = np.nan

    return df

main()
