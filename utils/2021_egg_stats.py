import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ptitprince as pt
from statannotations.Annotator import Annotator


def load_csv_files(root_folder: str, dates: list, treatments: list) -> pd.DataFrame:
    master_df = None
    csv_file = 'measurements_log.csv'

    for d in dates:
        for t in treatments:
            dir_path = os.path.join(root_folder, d, t)
            if os.path.isdir(dir_path):
                csv_path = os.path.join(dir_path, '{}_{}_{}'.format(d, t, csv_file))
                if os.path.isfile(csv_path):
                    df = pd.read_csv(csv_path)
                    df['Date'] = df['Date'].astype(str)

                    if master_df is None:
                        master_df = df
                    else:
                        master_df = pd.concat([master_df, df], ignore_index=True)

    return master_df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Set datetime from Image ID
    df['Date'] = df['Image ID'].apply(
        lambda x: '{}-{}-{}{}:{}:{}'.format(x[1:5], x[5:7], x[7:9], x[9:12], x[12:14], x[14:]))
    df.rename(columns={'Date': 'DateTime'}, inplace=True)

    # Drop rows that are just the column names (from concat in load_csv_files) and reindex
    df = df.drop(df.index[df['Image ID'] == 'Image ID'].tolist()).reset_index(drop=True)

    # Combine multiple runs of same sample
    df.loc[df['Treatment'] == 2, ['Treatment']] = 1
    df.loc[df['Treatment'] == 3, ['Treatment']] = 1
    df.loc[df['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl'

    # Drop unused columns
    df = df.drop(columns=['Dev.'])

    # Change column types from object to appropriate numericals (required for plotting)
    type_dict = {'DateTime': 'datetime64',
                 'Treatment': str,
                 'Fish ID': int,
                 'Egg area[mm2]': float,
                 'Egg min diameter[mm]': float,
                 'Egg max diameter[mm]': float,
                 'Eye area[mm2]': float,
                 'Eye min diameter[mm]': float,
                 'Eye max diameter[mm]': float,
                 'Yolk area[mm2]': float,
                 'Yolk length[mm]': float,
                 'Yolk height[mm]': float,
                 'Embryo area[mm2]': float}
    df = df.astype(type_dict)

    # df = remove_bad_eggs(df)
    # df = remove_bad_yolks(df)
    # df = remove_bad_eyes(df)
    # df = remove_bad_embryos(df)

    # df = df.set_index('DateTime')
    # df = df.drop(columns=['Image ID'])

    return df


def additional_calcs(df: pd.DataFrame) -> pd.DataFrame:
    a = df['Eye min diameter[mm]'] / 2
    b = df['Eye max diameter[mm]'] / 2
    # Ellipse area
    df['Eye area calculated[mm2]'] = a * b * np.pi
    # Prolate spheroid for eye volume
    df['Eye volume[mm3]'] = (4 / 3) * np.pi * a * a * b

    # Invert yolk area because we actually measure the gaps around the yolk
    df['Yolk area[mm2]'] = df['Egg area[mm2]'] - df['Yolk area[mm2]']
    df['Yolk fraction (area)'] = df['Yolk area[mm2]'] / df['Egg area[mm2]']

    df['Egg diameter[mm]'] = df['Egg max diameter[mm]']
    return df


def remove_bad_eggs(df: pd.DataFrame) -> pd.DataFrame:
    bad_eggs = ['D20210408T215004.597206',
                'D20210408T215004.974115',
                'D20210410T192217.785397',
                'D20210410T192446.199663',
                'D20210413T161726.650199',
                'D20210415T171258.778696',
                'D20210415T171311.321749',
                'D20210415T171325.851014',
                'D20210415T171355.654837',
                'D20210415T171440.174480',
                'D20210415T171918.213895',
                'D20210416T140645.565762',
                'D20210416T140839.588810',
                'D20210417T183003.240531',
                'D20210418T182732.232767',
                'D20210418T183011.836572',
                'D20210418T183015.778917']

    # Finds rows in bad eggs, now we need to set them to nan for Egg area and Egg diameter
    bad_rows = df.loc[df.apply(lambda x: x['Image ID'] in bad_eggs, axis=1)].index
    df['Egg area[mm2]'][bad_rows] = np.nan
    # df['Egg diameter[mm]'][bad_rows] = np.nan

    return df


def remove_bad_yolks(df: pd.DataFrame) -> pd.DataFrame:
    bad_yolks = ['D20210408T215004.848409',
                 'D20210410T192217.785397',
                 'D20210415T171348.363716',
                 'D20210415T171431.188135',
                 'D20210416T135351.752026',
                 'D20210418T182831.351010',
                 'D20210418T182953.655459',
                 'D20210418T183009.894198']

    bad_rows = df.loc[df.apply(lambda x: x['Image ID'] in bad_yolks, axis=1)].index
    df['Yolk area[mm2]'][bad_rows] = np.nan
    df['Yolk length[mm]'][bad_rows] = np.nan
    df['Yolk height[mm]'][bad_rows] = np.nan

    return df


def remove_bad_eyes(df: pd.DataFrame) -> pd.DataFrame:
    bad_eyes = ['D20210415T171431.188135',
                'D20210410T194843.245594',
                'D20210410T194843.872315',
                'D20210410T194844.624692',
                'D20210410T194844.747920',
                'D20210410T194847.257758',
                'D20210410T194848.759615',
                'D20210410T194850.641981',
                'D20210410T194901.872564',
                'D20210410T194907.197515',
                'D20210410T194907.826620',
                'D20210411T171740.339895',
                'D20210411T171800.585303',
                'D20210411T171807.231209',
                'D20210411T171834.623184',
                'D20210411T171918.571826',
                'D20210411T171921.714277',
                'D20210411T171931.682152',
                'D20210411T171943.032944',
                'D20210411T172003.661196',
                'D20210412T161116.449301',
                'D20210412T161209.502832',
                'D20210412T161234.132549',
                'D20210412T161303.346441',
                'D20210412T161513.487042',
                'D20210412T161624.279357',
                'D20210412T161637.192775',
                'D20210412T161643.460041',
                'D20210413T162049.712483',
                'D20210413T162118.864150',
                'D20210413T162200.375770',
                'D20210413T162315.931881',
                'D20210413T162407.836272',
                'D20210413T162418.617040',
                'D20210414T162620.351269',
                'D20210414T162803.861335',
                'D20210414T162804.738577',
                'D20210414T162828.431679',
                'D20210414T162851.872488',
                'D20210414T162859.724650',
                'D20210414T162934.341137',
                'D20210414T163003.233569',
                'D20210415T171247.934631',
                'D20210415T171339.271451',
                'D20210415T171342.344452',
                'D20210415T171344.852577',
                'D20210415T171346.983919',
                'D20210415T171406.421224',
                'D20210415T171406.547045',
                'D20210415T171414.955896',
                'D20210415T171415.329807',
                'D20210415T171424.606171',
                'D20210415T171428.428861',
                'D20210415T171443.227462',
                'D20210415T171443.728670',
                'D20210415T171449.751520',
                'D20210415T171501.850541',
                'D20210415T171502.978774',
                'D20210415T171515.334696',
                'D20210415T171522.230949',
                'D20210415T171528.620842',
                'D20210415T171545.425450',
                'D20210415T171623.609517',
                'D20210416T135609.350783',
                'D20210416T135942.631131',
                'D20210416T140533.268816',
                'D20210417T182912.920109',
                'D20210417T183321.353222',
                'D20210418T182828.462609']

    bad_single_eye = [['D20210411T171242.962132', 0.014787116512280105],
                      ['D20210411T172009.552943', 0.019206254780317837],
                      ['D20210412T161210.813555', 0.00625235221988855],
                      ['D20210412T161307.169766', 0.013876579781228375],
                      ['D20210412T161535.056411', 0.023334021294419017],
                      ['D20210414T162651.515444', 0.008644028700117762],
                      ['D20210414T162736.338955', 0.01473855455329068],
                      ['D20210414T162840.465592', 0.017761536500382424],
                      ['D20210414T162931.581786', 0.0167660163410992],
                      ['D20210414T163012.074430', 0.014520025737838265],
                      ['D20210415T171326.102599', 0.04045211183819155],
                      ['D20210415T171332.622427', 0.041119838774296154],
                      ['D20210415T171432.441263', 0.03553521349051221],
                      ['D20210415T171432.441263', 0.02187716252473625],
                      ['D20210415T171924.466051', 0.0007162888950940281],
                      ['D20210416T135419.204407', 0.022933385132756254],
                      ['D20210416T135838.701685', 0.03451541235173427],
                      ['D20210416T140607.620140', 0.005572484794036592],
                      ['D20210417T182912.165750', 0.000048561958989425635],
                      ['D20210417T182912.165750', 0.000024280979494712818],
                      ['D20210417T182939.705235', 0.03592370916242761],
                      ['D20210417T183346.434640', 0.00008498342823149486],
                      ['D20210417T183346.434640', 0.0240624506792604],
                      ['D20210417T183351.702383', 0.020019667593390716],
                      ['D20210417T183503.981379', 0.03472180067743933],
                      ['D20210418T182124.102393', 0.019764717308696233],
                      ['D20210418T182252.643319', 0.0400757566560235],
                      ['D20210418T182731.732613', 0.08125629787905644]]

    bad_rows = df.loc[df.apply(lambda x: x['Image ID'] in bad_eyes, axis=1)].index

    for eye in bad_single_eye:
        idx = df.loc[(df['Image ID'] == eye[0]) & (np.isclose(df['Eye area[mm2]'], eye[1], atol=1e-03))].index
        if idx.any():
            bad_rows = bad_rows.append(idx)
    df['Eye area[mm2]'][bad_rows] = np.nan
    # df['Eye min diameter[mm]'][bad_rows] = np.nan
    # df['Eye max diameter[mm]'][bad_rows] = np.nan
    return df


def remove_bad_embryos(df: pd.DataFrame) -> pd.DataFrame:
    bad_embryos = ['D20210408T215004.848409',
                   'D20210410T192338.702530',
                   'D20210418T182731.732613',
                   ]

    bad_rows = df.loc[df.apply(lambda x: x['Image ID'] in bad_embryos, axis=1)].index

    return df


def plot_attribute_by_date(df: pd.DataFrame, attribute: str):
    figsize = (18, 18)
    f, ax = plt.subplots(figsize=figsize)

    df['dpf'] = df['DateTime'].dt.day - 1
    ax = sns.swarmplot(x='dpf', y=attribute, data=df)

    ax.set_title('{}'.format(attribute.split('[')[0]))

    plt.tight_layout()
    plt.show()


def journal_plot(df: pd.DataFrame, attribute: str, treatment: str) -> None:
    df['dpf'] = df['DateTime'].dt.day - 6

    figsize = (18, 18)
    ort = 'v'
    linewidth = 2
    label_fontsize = 12
    title_fontsize = 20
    x = 'dpf'
    y = attribute
    # y_lims = y_lims

    pal = sns.color_palette('Set2')
    colour = pal[1]

    data = df[df['Treatment'] == treatment]

    means = [data[data['dpf'] == dpf].mean()[attribute] for dpf in range(2, 13)]
    std_errs = [data[data['dpf'] == dpf].std()[attribute] for dpf in range(2, 13)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(x=x, y=y, data=data, ax=ax, color='gray', size=10, zorder=-5)

    fig.set_size_inches(9, 9)
    # Linear regression
    sns.regplot(x=x, y=y, data=data, ax=ax, scatter=False, color=colour,
                truncate=False, line_kws={'zorder': 10})
    # sns.regplot(x=x, y=y, data=data_2, ax=ax, scatter=False, color=colour,
    #             truncate=False, line_kws={'zorder': 10})

    # Error bars
    for i in range(11):
        y_lo = means[i] - std_errs[i]
        y_hi = means[i] + std_errs[i]
        plt.plot([i, i], [y_lo, y_hi], linewidth=linewidth, color='black')
        # Mean bars
        plt.plot([i-0.25, i+0.25], [means[i], means[i]], linewidth=linewidth * 1.5, color='black')
        # Whisker caps
        plt.plot([i - 0.125, i + 0.125], [y_lo, y_lo], linewidth=linewidth * 1.5, color='black')
        plt.plot([i - 0.125, i + 0.125], [y_hi, y_hi], linewidth=linewidth * 1.5, color='black')

    # ax.set(ylim=y_lims)

    # Tick appearance
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(y_lims[0], y_lims[1], 0.01)))
    # ax.minorticks_on()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    # ax.tick_params(axis='y', which='minor', left=True)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=8, labelsize=label_fontsize)
    ax.tick_params(axis='both', which='both', width=linewidth)

    ax.set_xlabel('Days post fertilisation', fontsize=label_fontsize)
    ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # ax.set_title('{} - {}'.format(attribute.split('[')[0], treatment_names[treatment]), fontsize=title_fontsize)
    ax.set_title('{}, {}'.format(attribute.split('[')[0], treatment), fontsize=title_fontsize)


    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/dave/Desktop/tox_plots/oil/{}_{}.png'.format(attribute, treatment))


def plot_both_treatments(df: pd.DataFrame, attribute: str) -> None:
    df['dpf'] = df['DateTime'].dt.day - 6

    figsize = (18, 18)
    ort = 'v'
    linewidth = 2
    label_fontsize = 12
    title_fontsize = 20
    x = 'dpf'
    y = attribute
    # y_lims = y_lims

    pal = sns.color_palette('Set2')
    colour = pal[1]

    data_1 = df[df['Treatment'] == '1']
    data_2 = df[df['Treatment'] == '2']

    means_1 = [data_1[data_1['dpf'] == dpf].mean()[attribute] for dpf in range(2, 13)]
    std_errs_1 = [data_1[data_1['dpf'] == dpf].std()[attribute] for dpf in range(2, 13)]
    means_2 = [data_2[data_2['dpf'] == dpf].mean()[attribute] for dpf in range(2, 13)]
    std_errs_2 = [data_2[data_2['dpf'] == dpf].std()[attribute] for dpf in range(2, 13)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(x=x, y=y, data=df, ax=ax, color='gray', hue='Treatment', size=10, zorder=-5)

    fig.set_size_inches(9, 9)
    # Linear regression
    sns.regplot(x=x, y=y, data=data_1, ax=ax, scatter=False, color=colour,
                truncate=False, line_kws={'zorder': 10})
    sns.regplot(x=x, y=y, data=data_2, ax=ax, scatter=False, color=colour,
                truncate=False, line_kws={'zorder': 10})

    # Error bars
    for i in range(11):
        y_lo_1 = means_1[i] - std_errs_1[i]
        y_hi_1 = means_1[i] + std_errs_1[i]
        plt.plot([i, i], [y_lo_1, y_hi_1], linewidth=linewidth, color='black')
        # Mean bars
        plt.plot([i-0.25, i+0.25], [means_1[i], means_1[i]], linewidth=linewidth * 1.5, color='black')
        # Whisker caps
        plt.plot([i - 0.125, i + 0.125], [y_lo_1, y_lo_1], linewidth=linewidth * 1.5, color='black')
        plt.plot([i - 0.125, i + 0.125], [y_hi_1, y_hi_1], linewidth=linewidth * 1.5, color='black')

        y_lo_2 = means_2[i] - std_errs_2[i]
        y_hi_2 = means_2[i] + std_errs_2[i]
        plt.plot([i, i], [y_lo_2, y_hi_2], linewidth=linewidth, color='black')
        # Mean bars
        plt.plot([i-0.25, i+0.25], [means_2[i], means_2[i]], linewidth=linewidth * 1.5, color='black')
        # Whisker caps
        plt.plot([i - 0.125, i + 0.125], [y_lo_2, y_lo_2], linewidth=linewidth * 1.5, color='black')
        plt.plot([i - 0.125, i + 0.125], [y_hi_2, y_hi_2], linewidth=linewidth * 1.5, color='black')

    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=8, labelsize=label_fontsize)
    ax.tick_params(axis='both', which='both', width=linewidth)

    ax.set_xlabel('Days post fertilisation', fontsize=label_fontsize)
    ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # ax.set_title('{} - {}'.format(attribute.split('[')[0], treatment_names[treatment]), fontsize=title_fontsize)
    ax.set_title('{}'.format(attribute.split('[')[0]), fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

def boxplot_by_treatment_all_dates(df: pd.DataFrame, attribute: str):
    data = df.loc[(df['DateTime'].dt.date <= pd.to_datetime('20200412'))]
    x = 'Treatment'
    y = attribute
    hue = df['DateTime'].dt.date

    figsize = (12, 12)
    f, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data)
    pairs = [('DCA-ctrl', 'DCA-0,15'), ('DCA-ctrl', 'DCA-0,31'), ('DCA-ctrl', 'DCA-0,62'), ('DCA-ctrl', 'DCA-1,25'), ('DCA-ctrl', 'DCA-2,50'),  ('DCA-ctrl', 'DCA-5,00')]

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
    for median in lines[4:len(lines)-lines_per_box:lines_per_box]:
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

    ax.set_title('{}'.format(attribute))

    plt.tight_layout()
    plt.show()


def main():
    # root_folder = '/mnt/6TB_Media/PhD Work/2021_cod/egg_results/egg_results_2188'
    # dates = ['202104{:02d}'.format(i) for i in range(8, 19)]
    # treatments = ['1', '2']

    # 2020 reanalysis
    # root_folder = '/media/dave/DATA/2020_reanalysis/eggs/1088/'
    root_folder = '/media/dave/DATA/2020_reanalysis/eggs/1151/'
    dates = ['202004{:02d}'.format(i) for i in range(4, 13)]
    treatments = ['1', '2', '3', 'DCA-ctrl', 'DCA-ctrl-2', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']

    df: pd.DataFrame = load_csv_files(root_folder, dates, treatments)
    df = prepare_dataframe(df)
    df = additional_calcs(df)

    boxplot_by_treatment_all_dates(df, 'Egg diameter[mm]')
    boxplot_by_treatment_all_dates(df, 'Yolk area[mm2]')
    boxplot_by_treatment_all_dates(df, 'Yolk fraction (area)')
    boxplot_by_treatment_all_dates(df, 'Eye area[mm2]')
    boxplot_by_treatment_all_dates(df, 'Embryo area[mm2]')


    plot_attribute_by_date(df, 'Egg diameter[mm]')
    plot_attribute_by_date(df, 'Yolk area[mm2]')
    plot_attribute_by_date(df, 'Yolk fraction (area)')
    plot_attribute_by_date(df, 'Eye area[mm2]')
    plot_attribute_by_date(df, 'Embryo area[mm2]')

    # plot_attribute_by_date(df, 'Eye area calculated[mm2]')

    treatment = '1'

    # # journal_plot(df, 'Egg min diameter[mm]', treatment)
    # # journal_plot(df, 'Egg max diameter[mm]', treatment)
    # journal_plot(df, 'Yolk area[mm2]', treatment)
    # # journal_plot(df, 'Eye area[mm2]', treatment)
    # journal_plot(df, 'Eye area calculated[mm2]', treatment)
    # # journal_plot(df, 'Eye volume[mm3]', treatment)
    # journal_plot(df, 'Egg area[mm2]', treatment)
    # journal_plot(df, 'Yolk fraction (area)', treatment)

    # plot_both_treatments(df, 'Egg diameter[mm]')
    # plot_both_treatments(df, 'Yolk area[mm2]')
    # plot_both_treatments(df, 'Eye area[mm2]')
    # journal_plot(df, 'Embryo area[mm2]', treatment)


main()
