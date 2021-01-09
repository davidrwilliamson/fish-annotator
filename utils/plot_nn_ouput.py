import os
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import seaborn as sns
import ptitprince as pt


def prepare_dataframe(df):
    df = remove_bad_measurements(df)

    df.loc[df['Treatment'] == 2, ['Treatment']] = 1
    df.loc[df['Treatment'] == 3, ['Treatment']] = 1
    df.loc[df['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl'

    # Drop useless columns
    df = df.drop(columns=['Dev.', 'Myotome height[mm]', 'Yolk fraction'])
    # Drop rows where all measurements are null
    df = df.drop(index=df[df.isnull().sum(axis=1) == 11].index)

    # # Aggregate measurements for each fish into a single row
    # agg_func = {'Date': 'first', 'Treatment': 'first', 'Body area[mm2]': 'sum', 'Myotome length[mm]': 'sum', 'Yolk area[mm2]': 'sum', 'Yolk length[mm]': 'sum', 'Yolk height[mm]': 'sum'}
    # df = df.groupby(by=['Image ID', 'Fish ID']).aggregate(agg_func)
    # df.reset_index(inplace=True)
    #
    # # Replace zero measurements with nan again
    # # cols = ['Body area[mm2]', 'Myotome length[mm]', 'Eye area[mm2]', 'Eye min diameter[mm]', 'Eye max diameter[mm]', 'Yolk area[mm2]', 'Yolk length[mm]', 'Yolk height[mm]']
    # cols = ['Body area[mm2]', 'Myotome length[mm]', 'Yolk area[mm2]', 'Yolk length[mm]', 'Yolk height[mm]']
    # df[cols] = df[cols].replace(0, np.nan)

    df['Total body volume[mm3]'] = ((df['Body area[mm2]'] * df['Body area[mm2]']) / df['Myotome length[mm]'])\
        .apply(lambda x: x * np.pi * 0.25)
    # The yolk calculation is wrong! BUT I already based my manual checking off it, so we'll need to correct after that step
    df['Yolk volume[mm3]'] = df['Yolk length[mm]'] * df['Yolk height[mm]'] * df['Yolk height[mm]'] * np.pi * 0.25
    df['Body volume[mm3]'] = df['Total body volume[mm3]'] - df['Yolk volume[mm3]']

    df = remove_bad_volumes(df)
    # Ellipsoid: 4/3 pi a^2 c
    a = df['Yolk height[mm]'] * 0.5
    c = df['Yolk length[mm]'] * 0.5
    df['Yolk volume[mm3]'] = np.pi * (4./3.) * a * a * c
    df['Body volume[mm3]'] = df['Total body volume[mm3]'] - df['Yolk volume[mm3]']

    df = remove_bad_eyes(df)
    a = df['Eye min diameter[mm]'] * 0.5
    c = df['Eye max diameter[mm]'] * 0.5
    df['Eye volume[mm3]'] = np.pi * (4. / 3.) * a * a * c
    # df = df.drop(index=df[df.isnull().sum(axis=1) == 8].index)

    return df


def load_csv_files(root_folder, dates, treatments) -> pd.DataFrame:
    master_df = None
    csv_file = 'measurements_log.csv'

    for d in dates:
        for t in treatments:
            dir_path = os.path.join(root_folder, d, t)
            # csv_path = os.path.join(dir_path, '{}'.format(csv_file)) # Eggs
            csv_path = os.path.join(dir_path, '{}_{}_{}'.format(d, t, csv_file))
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                df['Date'] = df['Date'].astype(str)
                # df['Date'] = pd.to_datetime(df['Date'])

                if master_df is None:
                    master_df = df
                else:
                    master_df = pd.concat([master_df, df], ignore_index=True)

    return master_df


def plot_attribute(df, dx, dy, title):
    figsize = (9, 9)
    ort = 'v'
    pal = sns.color_palette('Set2')
    # pal = sns.color_palette('Paired')
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
    # ax.set(ylim=(1.5, 5.5))

    plt.show()


def larvae_grid(df: pd.DataFrame):
    dx = 'Myotome length[mm]'
    pal = sns.color_palette('Set2')
    sigma = 0.2

    g = sns.FacetGrid(df, row="Date", col="Treatment", margin_titles=True)
    g.map_dataframe(pt.half_violinplot, x=dx, data=df, palette=pal, bw=sigma, cut=0.,
                    scale="area", width=.6, inner=None)
    g.map(pt.stripplot, x=dx, data=df, palette=pal, edgecolor="white",
          size=3, jitter=1, zorder=0, move=0.25)
    g.map(sns.boxplot, x=dx, data=df, palette=pal, width=.15, zorder=10,
          showcaps=True, boxprops={'linewidth': 1, "zorder": 10},
          showfliers=False, whiskerprops={'linewidth': 1, "zorder": 10},
          saturation=1)
    g.set_axis_labels(dx, '')
    g.fig.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.show()


def plot_means_heatmap(df):
    attributes = ['Body volume[mm3]', 'Total body volume[mm3]', 'Yolk volume[mm3]']
    # attributes = ['Myotome length[mm]', 'Body area[mm2]', 'Eye area[mm2]', 'Eye min diameter[mm]', 'Yolk area[mm2]']
    dates = ['20200413', '20200414', '20200415', '20200416', '20200417']
    treatments = [1, 'DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50']

    for a in attributes:
        means = pd.DataFrame(columns=['Date', 'Treatment', a])
        for d in dates:
            for t in treatments:
                all_means = df.loc[(df['Treatment'] == t) & (df['Date'] == d)].mean()
                means = means.append({'Date': d, 'Treatment': t, a: all_means[a]}, ignore_index=True)
        pivot = means.pivot('Treatment', 'Date', a)
        pivot = pivot.reindex(treatments)
        ax = sns.heatmap(pivot, annot=True, fmt='0.3f', cbar=False)
        ax.set_title(a)
        # plt.sca(ax)
        # plt.yticks()
        plt.show()


def journal_plot(df, attribute, treatment, y_lims):
    figsize = (18, 18)
    ort = 'v'
    linewidth = 2
    label_fontsize = 12
    title_fontsize = 20
    x = 'Date'
    y = attribute
    y_lims = y_lims

    pal = sns.color_palette('Set2')
    colour = pal[1]

    data = df[df['Treatment'] == treatment]
    dates = data[x].unique()
    date_ordinal = data['Date'].copy()
    date_strings = ['20200413', '20200414', '20200415', '20200416', '20200417']
    for i, d in enumerate(date_strings):
        date_ordinal[date_ordinal == d] = float(i)
    means = [data[data['Date'] == date].mean()[attribute] for date in dates]
    std_errs = [data[data['Date'] == date].std()[attribute] for date in dates]

    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(x=x, y=y, data=data, ax=ax, color='gray', size=15, zorder=-5)
    fig.set_size_inches(9, 9)
    # Linear regression
    sns.regplot(x=date_ordinal.values.astype('float'), y=y, data=data, ax=ax, scatter=False, color=colour,
                truncate=False, line_kws={'zorder': 10})

    # Error bars
    for i in range(len(dates)):
        y_lo = means[i] - std_errs[i]
        y_hi = means[i] + std_errs[i]
        plt.plot([i, i], [y_lo, y_hi], linewidth=linewidth, color='black')
        # Mean bars
        plt.plot([i-0.25, i+0.25], [means[i], means[i]], linewidth=linewidth * 1.5, color='black')
        # Whisker caps
        plt.plot([i - 0.125, i + 0.125], [y_lo, y_lo], linewidth=linewidth * 1.5, color='black')
        plt.plot([i - 0.125, i + 0.125], [y_hi, y_hi], linewidth=linewidth * 1.5, color='black')

    ax.set(ylim=y_lims)
    ax.set_title('{} - {}'.format(attribute.split('[')[0], treatment), fontsize=title_fontsize)

    # Tick appearance
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(y_lims[0], y_lims[1], 0.01)))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=True)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=8, labelsize=label_fontsize)
    ax.tick_params(axis='both', which='both', width=linewidth)

    ax.set_xlabel('')
    ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)
    # plt.show()
    plt.savefig('/home/dave/Desktop/tox_plots/{}_{}.png'.format(attribute, treatment))


def journal_plot_daily(df, attribute, date, y_lims):
    figsize = (18, 18)
    linewidth = 2
    label_fontsize = 12
    title_fontsize = 20
    x = 'Treatment'
    y = attribute
    y_lims = y_lims

    pal = sns.color_palette('Set2')
    colour = pal[1]

    data = df[df['Date'] == date]
    data = data.drop(index=data[data['Treatment'] == 1].index)

    treatments = data[x].unique()
    treatment_float = data['Treatment'].copy()
    treatment_strings = ['DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50']
    treatments_conc = [0.00208, 7.89311, 27.1179, 108.1051, 219.5015, 342.9883]
    for i, t in enumerate(treatment_strings):
        treatment_float[treatment_float == t] = treatments_conc[i]
    means = [data[data['Treatment'] == treatment].mean()[attribute] for treatment in treatments]
    std_errs = [data[data['Treatment'] == treatment].std()[attribute] for treatment in treatments]

    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(x=x, y=y, data=data, ax=ax, color='gray', size=15, zorder=-5)
    fig.set_size_inches(9, 9)
    # sns.regplot(x=treatment_float.values.astype('float'), y=y, data=data, ax=ax, scatter=False, color=colour,
    #             truncate=False, line_kws={'zorder': 10})

    # Error bars
    for i in range(len(treatments)):
        y_lo = means[i] - std_errs[i]
        y_hi = means[i] + std_errs[i]
        plt.plot([i, i], [y_lo, y_hi], linewidth=linewidth, color='black')
        # Mean bars
        plt.plot([i-0.25, i+0.25], [means[i], means[i]], linewidth=linewidth * 1.5, color='black')
        # Whisker caps
        plt.plot([i - 0.125, i + 0.125], [y_lo, y_lo], linewidth=linewidth * 1.5, color='black')
        plt.plot([i - 0.125, i + 0.125], [y_hi, y_hi], linewidth=linewidth * 1.5, color='black')

    ax.set(ylim=y_lims)
    ax.set_title('{} - {}'.format(attribute.split('[')[0], date), fontsize=title_fontsize)

    # Tick appearance
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(y_lims[0], y_lims[1], 0.01)))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=True)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=8, labelsize=label_fontsize)
    ax.tick_params(axis='both', which='both', width=linewidth)

    ax.set_xlabel('Concentration (µg/L)', fontsize=label_fontsize)
    ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    xticklabels = ['Ctrl', '8', '27', '108', '220', '342']
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)
    #plt.show()
    plt.savefig('/home/dave/Desktop/tox_plots/{}_{}.png'.format(attribute, date))


def daily_regression_plot_only(df, attribute, date, y_lim):
    data = df[df['Date'] == date]
    data = data.drop(index=data[data['Treatment'] == 1].index)

    treatments = data['Treatment'].unique()
    treatment_float = data['Treatment'].copy()
    treatment_strings = ['DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50']
    treatments_conc = [0.00208, 7.89311, 27.1179, 108.1051, 219.5015, 342.9883]
    for i, t in enumerate(treatment_strings):
        treatment_float[treatment_float == t] = treatments_conc[i]
    means = [data[data['Treatment'] == treatment].mean()[attribute] for treatment in treatments]
    std_errs = [data[data['Treatment'] == treatment].std()[attribute] for treatment in treatments]

    figsize = (9, 9)
    linewidth = 2
    label_fontsize = 12
    title_fontsize = 20

    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(x=treatments_conc, y=means, ax=ax)

    ax.set_title('{} - {}'.format(attribute.split('[')[0], date), fontsize=title_fontsize)

    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=True)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=8, labelsize=label_fontsize)
    ax.tick_params(axis='both', which='both', width=linewidth)

    ax.set_xlabel('Concentration (µg/L)', fontsize=label_fontsize)
    ax.set_ylabel('Yolk volume (mm^3)', fontsize=label_fontsize)

    xticklabels = ['Ctrl', '8', '27', '108', '220', '342']
    ax.set_xticks(treatments_conc)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)
    ax.set(xlim=(-10, 350), ylim=y_lim)

    # plt.show()
    plt.savefig('/home/dave/Desktop/tox_plots/by_date/by_date_regression_only/{}_{}.png'.format(attribute, date))


def main():
    bhh = pd.read_csv('/home/dave/cod_results/BHH/bhh_larvae.csv')

    treatments = ['1', '2', '3', 'DCA-ctrl', 'DCA-ctrl-2', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50',
                  'DCA-5,00']

    root_folder_larvae = '/home/dave/cod_results/2106/'
    # root_folder_larvae = '/home/dave/cod_results/uncropped/1246/'
    dates_larvae = ['20200413', '20200414', '20200415', '20200416', '20200417']

    df = load_csv_files(root_folder_larvae, dates_larvae, treatments)
    df = prepare_dataframe(df)

    # journal_plot(df, 'Myotome length[mm]', 'DCA-ctrl', (0.0, 6.0))
    # daily_regression_plot_only(df, 'Yolk volume[mm3]', '20200413')
    y_lims = [(0.0, 0.7), (0.0, 0.3), (0.0, 0.6)]
    y_lims = [(0.0, 0.5), (0.0, 0.15), (0.0, 0.4)] # For daily regression plots
    y_lims = [(0.0, 6.0)]
    plot_attributes = ['Myotome length[mm]']#['Total body volume[mm3]', 'Yolk volume[mm3]', 'Body volume[mm3]']
    plot_treatments = ['DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50']
    for i, attribute in enumerate(plot_attributes):
        for date in dates_larvae:
            # daily_regression_plot_only(df, attribute, date, y_lims[i])
            journal_plot_daily(df, attribute, date, y_lims[i])
        # for treatment in plot_treatments:
        #     journal_plot(df, attribute, treatment, y_lims[i])

    # df.to_csv('/home/dave/Desktop/larvae_stats_(no_eyes).csv')

    # outliers_to_check = df.loc[(df['Eye min diameter[mm]'] > 0.37)]
    #                            # (df['Date'] != '20200413')]
    #
    # with open('/home/dave/cod_results/2106/outliers', 'w') as file:
    #     for i, row in outliers_to_check.iterrows():
    #         im_path = os.path.join(str(row['Date']), str(row['Treatment']), str(row['Image ID']))
    #         file.write('{}\n'.format(im_path))

    # plot_means_heatmap(df)

    attribute = 'Date'

    # plot_attribute(df, attribute, 'Myotome length[mm]', '')
    # plot_attribute(df, attribute, 'Body area[mm2]', '')
    # plot_attribute(df[df['Date'] == '20200415'], attribute, 'Yolk area[mm2]', '')
    # plot_attribute(df, attribute, 'Yolk area[mm2]', '')
    # plot_attribute(df, attribute, 'Yolk length[mm]', '')
    # plot_attribute(df, attribute, 'Yolk fraction', '')

    plot_attribute(df, attribute, 'Total body volume[mm3]', '')
    plot_attribute(df, attribute, 'Yolk volume[mm3]', '')
    plot_attribute(df, attribute, 'Body volume[mm3]', '')
    plot_attribute(df[df['Treatment'] == 'DCA-ctrl'], 'Date', 'Body volume[mm3]', 'DCA-ctrl')
    plot_attribute(df[df['Treatment'] == 'DCA-ctrl'], 'Date', 'Yolk volume[mm3]', 'DCA-ctrl')

    plot_attribute(df[df['Date'] == '20200416'], 'Treatment', 'Total body volume[mm3]', '20200416')
    plot_attribute(df[df['Date'] == '20200416'], 'Treatment', 'Body volume[mm3]', '20200416')
    plot_attribute(df[df['Date'] == '20200416'], 'Treatment', 'Yolk volume[mm3]', '20200416')

    plot_attribute(df, attribute, 'Yolk area[mm2]', '')
    plot_attribute(df, attribute, 'Yolk length[mm]', '')
    plot_attribute(df, attribute, 'Yolk height[mm]', '')

    plot_attribute(df[df['Treatment'] == 1], attribute, 'Yolk area[mm2]', '')

    plot_attribute(df, attribute, 'Eye area[mm2]', '')
    plot_attribute(df, attribute, 'Eye min diameter[mm]', '')
    plot_attribute(df, attribute, 'Eye max diameter[mm]', '')
    plot_attribute(df, attribute, 'Eye volume[mm3]', '')
    plot_attribute(df, attribute, 'Myotome length[mm]', '')
    plot_attribute(df, attribute, 'Body area[mm2]', '')

    # larvae_grid(df)

    # root_folder_eggs = '/home/dave/cod_results/cod_eggs/0735/'
    # dates_eggs = ['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410']
    #
    # eggs = load_csv_files(root_folder_eggs, dates_eggs, treatments)
    # eggs.loc[eggs['Treatment'] == 2, ['Treatment']] = 1
    # eggs.loc[eggs['Treatment'] == 3, ['Treatment']] = 1
    # eggs.loc[eggs['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl'
    #
    # eggs = eggs[eggs['Egg diameter[mm]'] > 1.25]
    # eggs = eggs[eggs['Egg diameter[mm]'] < 1.45]

    # plot_attribute(eggs[eggs['Treatment'] == '1'], 'Date', 'Egg diameter[mm]', 'Egg diameters with treatment 1')
    # plot_attribute(eggs[eggs['Date'] == '20200407'], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200407')
    # plot_attribute(eggs[eggs['Date'] == '20200408'], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200408')
    # plot_attribute(eggs[eggs['Date'] == '20200409'], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200409')
    # plot_attribute(eggs[eggs['Date'] == '20200410'], 'Treatment', 'Egg diameter[mm]', 'Egg diameters on 20200410')

    plot_attribute(df[df['Treatment'] == 'DCA-ctrl'], 'Date', 'Myotome length[mm]', 'DCA-ctrl')
    plot_attribute(df[df['Treatment'] == 'DCA-1,25'], 'Date', 'Myotome length[mm]', 'DCA-1,25')
    plot_attribute(df[df['Treatment'] == 'DCA-2,50'], 'Date', 'Myotome length[mm]', 'DCA-2,50')


def remove_bad_measurements(df):
    # Checked bodies longer than 5.5
    bad_rows_body = set([])
    bad_rows_yolk = set([])
    bad_rows_eye = set([])

    long_body_bad = df[df['Myotome length[mm]'] > 5.5]
    long_body_good = ['D20200414T152839.724122', 'D20200417T144818.260228']
    bad_rows_body.update([i for i, row in long_body_bad.iterrows() if row['Image ID'] not in long_body_good])
    short_body_bad = df[df['Myotome length[mm]'] < 3.5]
    short_body_good = ['D20200413T142505.217061']
    bad_rows_body.update([i for i, row in short_body_bad.iterrows() if row['Image ID'] not in short_body_good])

    small_body_bad = df[df['Body area[mm2]'] < 1.1]
    # Most of these turned out to be top-down views of larvae
    small_body_good = ['D20200413T160924.212674', 'D20200414T140104.388475', 'D20200417T154557.171992',
                       'D20200413T160653.341132', 'D20200414T144815.328230', 'D20200414T144825.386115',
                       'D20200414T161009.887529', 'D20200414T161145.101991', 'D20200414T163643.830330',
                       'D20200414T163953.800494', 'D20200414T164012.251490', 'D20200414T164015.258490',
                       'D20200415T161034.168505', 'D20200415T163446.742714', 'D20200415T171346.614542',
                       'D20200415T175042.812863', 'D20200416T130340.562986', 'D20200416T130341.942677',
                       'D20200416T145114.898082', 'D20200416T145418.557019', 'D20200416T151246.294328',
                       'D20200417T145139.672536', 'D20200417T154443.307138', 'D20200417T154455.164238',
                       'D20200417T154525.397128']
    bad_rows_body.update([i for i, row in small_body_bad.iterrows()]) # if row['Image ID'] not in small_body_good])

    large_body_bad = df[df['Body area[mm2]'] > 1.7]
    large_body_good = [
        'D20200413T132121.816669', 'D20200413T132124.325863', 'D20200413T132129.086159', 'D20200413T132143.505090',
        'D20200413T132145.010741', 'D20200413T132154.922511', 'D20200413T132212.534823', 'D20200413T132253.805714',
        'D20200413T132310.012605', 'D20200413T132317.660578', 'D20200413T132332.825405', 'D20200413T132332.951644',
        'D20200413T132353.389506', 'D20200413T132423.414061', 'D20200413T134021.085747', 'D20200413T134024.258697',
        'D20200413T134031.665507', 'D20200413T134038.051070', 'D20200413T134051.628632', 'D20200413T134053.007810',
        'D20200413T134056.319119', 'D20200413T134115.521369', 'D20200413T134136.993367', 'D20200413T134141.008157',
        'D20200413T134148.777159', 'D20200413T134204.387632', 'D20200413T134205.014534', 'D20200413T134222.712145',
        'D20200413T134233.675140', 'D20200413T134234.801972', 'D20200413T134239.301778', 'D20200413T134244.059384',
        'D20200413T134245.939653', 'D20200413T134301.260003', 'D20200413T134328.603213', 'D20200413T134412.352835',
        'D20200413T134414.357287', 'D20200413T134452.394012', 'D20200413T140209.893634', 'D20200413T140213.030847',
        'D20200413T140242.745574', 'D20200413T140243.374364', 'D20200413T140335.369001', 'D20200413T140409.655173',
        'D20200413T140428.740520', 'D20200413T140436.268023', 'D20200413T140603.902237', 'D20200413T142422.199090',
        'D20200413T142452.089148', 'D20200413T142453.594034', 'D20200413T142456.608896', 'D20200413T142457.237521',
        'D20200413T142457.988088', 'D20200413T142504.589779', 'D20200413T142512.612858', 'D20200413T142523.267188',
        'D20200413T142524.023115', 'D20200413T142526.280172', 'D20200413T142538.991654', 'D20200413T142606.669416',
        'D20200413T142622.463392', 'D20200413T142654.067608', 'D20200413T142655.948442', 'D20200413T142706.877499',
        'D20200413T142720.258141', 'D20200413T142720.384083', 'D20200413T142726.652636', 'D20200413T142728.784041',
        'D20200413T142747.739170', 'D20200413T142754.880046', 'D20200413T142807.293729', 'D20200413T142809.541599',
        'D20200413T142811.302182', 'D20200413T142812.681772', 'D20200413T142819.781696', 'D20200413T142850.827841',
        'D20200413T151116.748176', 'D20200413T151125.351364', 'D20200413T151140.585497', 'D20200413T151207.639213',
        'D20200413T151209.813316', 'D20200413T151248.937880', 'D20200413T151302.403830', 'D20200413T151341.719138',
        'D20200413T151347.991515', 'D20200413T151408.663607', 'D20200413T151504.850911', 'D20200413T151509.742228',
        'D20200413T151530.165757', 'D20200413T151623.689895', 'D20200413T151623.816016', 'D20200413T154944.002821',
        'D20200413T155017.491685', 'D20200413T155040.856220', 'D20200413T155117.733360', 'D20200413T155117.984603',
        'D20200413T155125.505895', 'D20200413T155136.710633', 'D20200413T155243.068021', 'D20200413T160638.701796',
        'D20200413T160757.402273', 'D20200414T140026.362917', 'D20200414T140206.326564', 'D20200414T140215.607471',
        'D20200414T140225.515273', 'D20200414T140312.521472', 'D20200414T142029.305309', 'D20200414T142120.070539',
        'D20200414T142214.285798', 'D20200414T142247.225297', 'D20200414T142334.838984', 'D20200414T152720.077075',
        'D20200414T152720.329268', 'D20200414T152820.377215', 'D20200414T161012.021152', 'D20200414T161114.270974',
        'D20200415T154516.773670', 'D20200415T154547.449904', 'D20200415T154651.422208', 'D20200415T171342.849738',
        'D20200416T130302.852417', 'D20200416T135910.916914', 'D20200417T144818.260228', 'D20200417T145123.143566']
    bad_rows_body.update([i for i, row in large_body_bad.iterrows() if row['Image ID'] not in large_body_good])
    large_body_check = df.loc[(df['Body area[mm2]'] <= 1.7) &
                              (df['Body area[mm2]'] > 1.6) &
                              (df['Date'] != '20200413')]
    bad_rows_body.update([i for i, row in large_body_check.iterrows() if row['Image ID'] in [
        'D20200414T142051.282137', 'D20200414T142058.430535', 'D20200414T142153.939852',
        'D20200414T142347.273930', 'D20200414T150521.463361', 'D20200414T163927.376564',
        'D20200414T163959.570080', 'D20200415T154605.258721', 'D20200415T160832.023872',
        'D20200415T161124.099010', 'D20200415T171300.225066', 'D20200415T181220.427865',
        'D20200416T130212.553664', 'D20200416T130314.387700', 'D20200416T130331.283813',
        'D20200416T135823.167540', 'D20200417T144822.019195', 'D20200417T152742.931520',
        'D20200417T152807.141005', 'D20200417T154400.539749', 'D20200417T160707.913175',
        'D20200417T160722.338021']])

    large_yolk_20200417_bad = df.loc[(df['Yolk area[mm2]'] > 0.05) & (df['Date'] == '20200417')]
    large_yolk_20200417_good = ['D20200417T144821.896526', 'D20200417T145123.143566', 'D20200417T154407.693884',
                                'D20200417T160853.233420', 'D20200417T144935.457509', 'D20200417T151038.269319',
                                'D20200417T152806.888908', 'D20200417T152807.014393', 'D20200417T152807.141005',
                                'D20200417T152807.393549', 'D20200417T152807.643911']
    bad_rows_yolk.update(
        [i for i, row in large_yolk_20200417_bad.iterrows() if row['Image ID'] not in large_yolk_20200417_good])

    short_yolk_20200417_good = df.loc[(df['Yolk length[mm]'] < 0.3) & (df['Date'] == '20200417')]
    short_yolk_20200417_bad = ['D20200417T144802.139576', 'D20200417T144921.449146', 'D20200417T150731.119034',
                               'D20200417T152445.229393', 'D20200417T152456.688054', 'D20200417T152704.836353',
                               'D20200417T152712.100731', 'D20200417T154418.970961', 'D20200417T160630.152832',
                               'D20200417T160838.169300', 'D20200417T170459.732437']
    bad_rows_yolk.update(
        [i for i, row in short_yolk_20200417_good.iterrows() if row['Image ID'] in short_yolk_20200417_bad])

    large_yolk_20200416_bad = df.loc[(df['Yolk area[mm2]'] > 0.07) & (df['Date'] == '20200416')]
    large_yolk_20200416_good = ['D20200416T130147.412234', 'D20200416T130218.312609', 'D20200416T130219.313372',
                                'D20200416T130323.547182', 'D20200416T132108.399455', 'D20200416T132108.525721',
                                'D20200416T132113.607278', 'D20200416T132223.939095', 'D20200416T132231.469435',
                                'D20200416T133842.037177', 'D20200416T133846.049960', 'D20200416T133849.197380',
                                'D20200416T133918.530452', 'D20200416T133932.718229', 'D20200416T134038.056123',
                                'D20200416T134054.588217', 'D20200416T134117.236012', 'D20200416T134117.361063',
                                'D20200416T134122.266908', 'D20200416T134134.334639', 'D20200416T134134.460182',
                                'D20200416T134154.837623', 'D20200416T134214.715276', 'D20200416T134214.840508',
                                'D20200416T134218.245552', 'D20200416T135823.167540', 'D20200416T135931.916010',
                                'D20200416T140013.901041', 'D20200416T145123.925654', 'D20200416T145254.726303',
                                'D20200416T145441.630818', 'D20200416T145518.201142', 'D20200416T145518.326138',
                                'D20200416T145533.376835', 'D20200416T151235.477671', 'D20200416T151306.995523',
                                'D20200416T151321.237668', 'D20200416T151408.696814', 'D20200416T152709.090827']
    bad_rows_yolk.update(
        [i for i, row in large_yolk_20200416_bad.iterrows() if row['Image ID'] not in large_yolk_20200416_good])

    short_yolk_20200416_good = df.loc[(df['Yolk length[mm]'] < 0.4) & (df['Date'] == '20200416')]
    short_yolk_20200416_bad = ['D20200416T130141.520147', 'D20200416T130245.358796', 'D20200416T130300.973723',
                               'D20200416T130322.918849', 'D20200416T130330.155835', 'D20200416T130331.908536',
                               'D20200416T130337.050703', 'D20200416T130342.695102', 'D20200416T130345.327202',
                               'D20200416T130402.795076', 'D20200416T130415.334285', 'D20200416T130418.266326',
                               'D20200416T132127.903182', 'D20200416T132248.734691', 'D20200416T132250.115078',
                               'D20200416T132259.394986', 'D20200416T133936.480758', 'D20200416T133936.480758',
                               'D20200416T134016.717781', 'D20200416T135812.386547', 'D20200416T135813.389034',
                               'D20200416T140007.885982', 'D20200416T140013.023882', 'D20200416T145053.138864',
                               'D20200416T145056.649646', 'D20200416T145157.256022', 'D20200416T145227.930424',
                               'D20200416T145513.559547', 'D20200416T151106.391046', 'D20200416T151123.300973',
                               'D20200416T151203.799566', 'D20200416T151335.293407', 'D20200416T152629.479348',
                               'D20200416T152649.806079', 'D20200416T152659.057044']
    bad_rows_yolk.update(
        [i for i, row in short_yolk_20200416_good.iterrows() if row['Image ID'] in short_yolk_20200416_bad])

    large_yolk_20200415_bad = df.loc[(df['Yolk area[mm2]'] > 0.125) & (df['Date'] == '20200415')]
    large_yolk_20200415_good = ['D20200415T154439.530103', 'D20200415T160649.940240',
     'D20200415T160708.137030', 'D20200415T160756.147999', 'D20200415T160758.907889', 'D20200415T160759.536174',
     'D20200415T160827.090197', 'D20200415T160827.215830', 'D20200415T160827.595688', 'D20200415T160827.720933',
     'D20200415T160835.537226', 'D20200415T160929.580793', 'D20200415T160937.141644', 'D20200415T160945.666730',
     'D20200415T160956.715470', 'D20200415T161045.428475', 'D20200415T163329.128097', 'D20200415T163330.632527',
     'D20200415T163330.757809', 'D20200415T163344.813225', 'D20200415T163538.135601', 'D20200415T163557.584048',
     'D20200415T163558.465306', 'D20200415T163559.849092', 'D20200415T163626.146481', 'D20200415T163637.708566',
     'D20200415T163657.640783', 'D20200415T163704.883506', 'D20200415T163712.170741', 'D20200415T163714.554317',
     'D20200415T163756.742186', 'D20200415T165053.665997', 'D20200415T165148.534762', 'D20200415T165148.660072',
     'D20200415T165148.784862', 'D20200415T165230.680765', 'D20200415T165237.081454', 'D20200415T165320.900809',
     'D20200415T165358.324389', 'D20200415T165418.746548', 'D20200415T165539.463892', 'D20200415T165539.590512',
     'D20200415T171109.704501', 'D20200415T171113.827510', 'D20200415T171142.191307', 'D20200415T171151.360151',
     'D20200415T171200.400383', 'D20200415T171205.417922', 'D20200415T171249.626783', 'D20200415T171342.849738',
     'D20200415T171504.293506', 'D20200415T171518.688985', 'D20200415T171547.222058', 'D20200415T175100.854619',
     'D20200415T175110.272767', 'D20200415T175157.931458', 'D20200415T175227.540026', 'D20200415T175232.571842',
     'D20200415T175252.369991', 'D20200415T181015.294825', 'D20200415T181015.419784', 'D20200415T181051.907334',
     'D20200415T181057.303104', 'D20200415T181111.331942', 'D20200415T181158.682318']
    bad_rows_yolk.update(
        [i for i, row in large_yolk_20200415_bad.iterrows() if row['Image ID'] not in large_yolk_20200415_good])

    small_yolk_20200415_good = df.loc[(df['Yolk area[mm2]'] < 0.075) & (df['Date'] == '20200415')]
    small_yolk_20200415_bad = ['D20200415T154332.928084', 'D20200415T154358.348908', 'D20200415T154359.729301',
                                'D20200415T154415.686255', 'D20200415T154425.969347', 'D20200415T154436.771938',
                                'D20200415T154441.035407', 'D20200415T154501.104530', 'D20200415T154507.095268',
                                'D20200415T154509.977729', 'D20200415T154516.773670', 'D20200415T154529.565965',
                                'D20200415T154537.718808', 'D20200415T154551.964069', 'D20200415T154605.258721',
                                'D20200415T154616.220195', 'D20200415T154620.861040', 'D20200415T154625.876015',
                                'D20200415T154635.532610', 'D20200415T154637.285982', 'D20200415T154638.041049',
                                'D20200415T154639.420774', 'D20200415T154658.067238', 'D20200415T160633.858486',
                                'D20200415T160648.935388', 'D20200415T160658.724646', 'D20200415T160705.122601',
                                'D20200415T160708.825496', 'D20200415T160712.213481', 'D20200415T160712.587851',
                                'D20200415T160715.848891', 'D20200415T160715.848891', 'D20200415T160719.134751',
                                'D20200415T160735.303406', 'D20200415T160737.185858', 'D20200415T160802.543575',
                                'D20200415T160804.294236', 'D20200415T160827.342644', 'D20200415T160836.790380',
                                'D20200415T160855.554104', 'D20200415T160903.966985', 'D20200415T160910.111049',
                                'D20200415T160927.321455', 'D20200415T160932.245548', 'D20200415T160952.327142',
                                'D20200415T161008.376459', 'D20200415T161009.508204', 'D20200415T161009.633759',
                                'D20200415T161011.896787', 'D20200415T161037.553802', 'D20200415T161041.690122',
                                'D20200415T161112.823545', 'D20200415T161122.342349', 'D20200415T163303.562279',
                                'D20200415T163316.502952', 'D20200415T163318.008928', 'D20200415T163339.795130',
                                'D20200415T163345.566957', 'D20200415T163345.690019', 'D20200415T163359.684081',
                                'D20200415T163406.718309', 'D20200415T163416.501722', 'D20200415T163426.910328',
                                'D20200415T163429.935550', 'D20200415T163446.119035', 'D20200415T163448.877067',
                                'D20200415T163509.516327', 'D20200415T163541.773911', 'D20200415T163554.451937',
                                'D20200415T163555.831953', 'D20200415T163603.686626', 'D20200415T163630.286388',
                                'D20200415T163639.087798', 'D20200415T163649.743784', 'D20200415T163656.639567',
                                'D20200415T165103.328241', 'D20200415T165110.610999', 'D20200415T165111.929987',
                                'D20200415T165114.949287', 'D20200415T165131.893466', 'D20200415T165132.018617',
                                'D20200415T165135.907881', 'D20200415T165136.413148', 'D20200415T165145.394350',
                                'D20200415T165154.195765', 'D20200415T165202.348524', 'D20200415T165203.103565',
                                'D20200415T165226.813634', 'D20200415T165228.551226', 'D20200415T165249.725377',
                                'D20200415T165251.728110', 'D20200415T165251.855268', 'D20200415T165253.737148',
                                'D20200415T165255.243181', 'D20200415T165255.619766', 'D20200415T165258.503114',
                                'D20200415T165309.289551', 'D20200415T165311.930219', 'D20200415T165341.475618',
                                'D20200415T165342.729960', 'D20200415T165345.270327', 'D20200415T165350.548922',
                                'D20200415T165411.137042', 'D20200415T165421.631416', 'D20200415T165452.886683',
                                'D20200415T165503.818152', 'D20200415T165504.195931', 'D20200415T165510.713744',
                                'D20200415T165515.998738', 'D20200415T165546.738186', 'D20200415T165553.639814',
                                'D20200415T171126.539697', 'D20200415T171134.832542', 'D20200415T171137.598587',
                                'D20200415T171212.530650', 'D20200415T171227.225062', 'D20200415T171253.957288',
                                'D20200415T171255.207749', 'D20200415T171257.591592', 'D20200415T171300.225066',
                                'D20200415T171303.986318', 'D20200415T171304.111276', 'D20200415T171317.858583',
                                'D20200415T171317.981329', 'D20200415T171319.235501', 'D20200415T171406.724669',
                                'D20200415T171406.724669', 'D20200415T171425.574345', 'D20200415T171432.098528',
                                'D20200415T171436.764387', 'D20200415T171455.110855', 'D20200415T171508.934227',
                                'D20200415T171517.317831', 'D20200415T171600.134922', 'D20200415T175002.768866',
                                'D20200415T175018.383524', 'D20200415T175045.196503', 'D20200415T175142.753507',
                                'D20200415T175147.771092', 'D20200415T175202.696554', 'D20200415T175212.238780',
                                'D20200415T175220.643688', 'D20200415T175220.768254', 'D20200415T175242.318284',
                                'D20200415T175249.109920', 'D20200415T175309.588081', 'D20200415T175320.976230',
                                'D20200415T175335.631737', 'D20200415T175335.756400', 'D20200415T181056.174533',
                                'D20200415T181117.350899', 'D20200415T181222.060289']
    bad_rows_yolk.update(
        [i for i, row in small_yolk_20200415_good.iterrows() if row['Image ID'] in small_yolk_20200415_bad])

    short_yolk_20200415_good = df.loc[(df['Yolk length[mm]'] < 0.5) & (df['Date'] == '20200415')]
    short_yolk_20200415_bad = ['D20200415T154339.540527', 'D20200415T154357.345816', 'D20200415T154416.061422',
                               'D20200415T154640.422997', 'D20200415T163419.637997', 'D20200415T163800.376326',
                               'D20200415T165154.697783', 'D20200415T171536.704875', 'D20200415T175026.796172',
                               'D20200415T175138.581798']
    bad_rows_yolk.update(
        [i for i, row in short_yolk_20200415_good.iterrows() if row['Image ID'] in short_yolk_20200415_bad])

    large_yolk_20200414_bad = [
        'D20200414T140051.863581', 'D20200414T140125.273944', 'D20200414T140249.094596',
        'D20200414T142120.569982', 'D20200414T142121.950964', 'D20200414T142326.552109',
        'D20200414T142347.273930', 'D20200414T142408.707740', 'D20200414T144644.104900',
        'D20200414T150333.268475', 'D20200414T150340.466250', 'D20200414T150405.800198',
        'D20200414T150406.551999', 'D20200414T150600.753571', 'D20200414T150601.631242',
        'D20200414T150604.514937', 'D20200414T161208.828480', 'D20200414T163523.750748',
        'D20200414T163527.271565', 'D20200414T163619.921995', 'D20200414T163635.402374',
        'D20200414T163700.664168', 'D20200414T163834.475176']
    large_yolk_20200414_good = df.loc[(df['Yolk area[mm2]'] > 0.25) & (df['Date'] == '20200414')]
    bad_rows_yolk.update(
        [i for i, row in large_yolk_20200414_good.iterrows() if row['Image ID'] in large_yolk_20200414_bad])

    small_yolk_20200414_bad = df.loc[(df['Yolk area[mm2]'] < 0.1) & (df['Date'] == '20200414')]
    small_yolk_20200414_good = [
         'D20200414T140053.369722', 'D20200414T140059.629751', 'D20200414T140103.762779', 'D20200414T140108.151342',
         'D20200414T140112.918005', 'D20200414T140114.064208', 'D20200414T140130.688049', 'D20200414T140137.082994',
         'D20200414T140142.459713', 'D20200414T140143.210454', 'D20200414T140221.752129', 'D20200414T140301.427416',
         'D20200414T140318.962564', 'D20200414T140327.486521', 'D20200414T140336.518324', 'D20200414T140340.786161',
         'D20200414T140342.666398', 'D20200414T140344.799814', 'D20200414T140347.411997', 'D20200414T140353.177375',
         'D20200414T140353.303475', 'D20200414T140356.812557', 'D20200414T142032.818062', 'D20200414T142051.660591',
         'D20200414T142115.553003', 'D20200414T142209.619771', 'D20200414T142300.443983', 'D20200414T142311.768871',
         'D20200414T144659.712051', 'D20200414T144725.905157', 'D20200414T144817.719122', 'D20200414T144817.836532',
         'D20200414T144836.557995', 'D20200414T144844.452427', 'D20200414T150357.772870', 'D20200414T150357.898827',
         'D20200414T150439.362003', 'D20200414T150441.635433', 'D20200414T150555.864218', 'D20200414T150602.254889',
         'D20200414T150642.102115', 'D20200414T150705.200508', 'D20200414T152704.648857', 'D20200414T152710.920519',
         'D20200414T152716.436985', 'D20200414T152820.000449', 'D20200414T152850.531347', 'D20200414T161022.346833',
         'D20200414T163519.990616', 'D20200414T163656.525784', 'D20200414T163659.662528', 'D20200414T163916.257258']
    bad_rows_yolk.update(
        [i for i, row in small_yolk_20200414_bad.iterrows() if row['Image ID'] not in small_yolk_20200414_good])

    short_yolk_20200414_good = df.loc[(df['Yolk length[mm]'] < 0.6) & (df['Date'] == '20200414')]
    short_yolk_20200414_bad = ['D20200414T140019.953309', 'D20200414T140039.623829', 'D20200414T140125.273944',
                               'D20200414T140340.283565', 'D20200414T140347.411997', 'D20200414T142153.814419',
                               'D20200414T142209.619771', 'D20200414T142258.438129', 'D20200414T142327.554487',
                               'D20200414T142349.401101', 'D20200414T144556.392077', 'D20200414T144844.452427',
                               'D20200414T144850.154746', 'D20200414T144904.507240', 'D20200414T150340.592345',
                               'D20200414T150433.970873', 'D20200414T150441.635433', 'D20200414T150450.169268',
                               'D20200414T152649.172025', 'D20200414T152846.139541', 'D20200414T152850.531347',
                               'D20200414T161022.346833', 'D20200414T161151.904288', 'D20200414T161252.817093',
                               'D20200414T163537.934566', 'D20200414T163547.657111', 'D20200414T163802.749182',
                               'D20200414T163927.376564', 'D20200414T163943.822029']
    bad_rows_yolk.update(
        [i for i, row in short_yolk_20200414_good.iterrows() if row['Image ID'] in short_yolk_20200414_bad])

    large_yolk_20200413_bad = df.loc[(df['Yolk area[mm2]'] > 0.5) & (df['Date'] == '20200413')]
    large_yolk_20200413_good = ['D20200413T134010.321304', 'D20200413T134034.044982', 'D20200413T134037.799584', 'D20200413T134053.007810',
     'D20200413T134202.882911', 'D20200413T134233.549795', 'D20200413T134233.675140', 'D20200413T140402.381515',
     'D20200413T140428.740520', 'D20200413T140450.196813', 'D20200413T140523.172530', 'D20200413T142504.589779',
     'D20200413T142812.681772', 'D20200413T142843.082694', 'D20200413T151207.639213', 'D20200413T151408.663607',
     'D20200413T151509.742228', 'D20200413T154850.221720', 'D20200413T155008.969072', 'D20200413T155026.141386',
     'D20200413T160638.991692', 'D20200413T160715.914916', 'D20200413T160725.493154', 'D20200413T160930.106638']
    bad_rows_yolk.update(
        [i for i, row in large_yolk_20200413_bad.iterrows() if row['Image ID'] not in large_yolk_20200413_good])

    small_yolk_20200413_bad = df.loc[(df['Yolk area[mm2]'] < 0.2) & (df['Date'] == '20200413')]
    small_yolk_20200413_good = ['D20200413T132202.886573', 'D20200413T132209.654588', 'D20200413T132226.500910',
                                'D20200413T132229.370250', 'D20200413T132249.290738', 'D20200413T132249.415244',
                                'D20200413T132318.787858', 'D20200413T132318.914347', 'D20200413T142746.987717',
                                'D20200413T151520.276294', 'D20200413T160942.772303']
    bad_rows_yolk.update(
        [i for i, row in small_yolk_20200413_bad.iterrows() if row['Image ID'] not in small_yolk_20200413_good])

    short_yolk_20200413_good = df.loc[(df['Yolk length[mm]'] < 0.7) & (df['Date'] == '20200413')]
    short_yolk_20200413_bad = ['D20200413T134033.815812', 'D20200413T142724.145692', 'D20200413T132124.948704',
                               'D20200413T132241.908137', 'D20200413T134028.810482', 'D20200413T134034.421266',
                               'D20200413T134054.757834', 'D20200413T134100.967695', 'D20200413T134323.580165',
                               'D20200413T134400.448176', 'D20200413T140217.668884', 'D20200413T140455.832925',
                               'D20200413T140456.101157', 'D20200413T140457.005322', 'D20200413T140508.179046',
                               'D20200413T142514.117625', 'D20200413T142546.266587', 'D20200413T142652.938491',
                               'D20200413T142816.077529', 'D20200413T151247.166510', 'D20200413T151325.029118',
                               'D20200413T151427.735532', 'D20200413T154943.501916', 'D20200413T155115.972438',
                               'D20200413T160617.069519', 'D20200413T160705.733239', 'D20200413T160709.905116',
                               'D20200413T160754.152143', 'D20200413T160900.856298']
    bad_rows_yolk.update(
        [i for i, row in short_yolk_20200413_good.iterrows() if row['Image ID'] in short_yolk_20200413_bad])

    df = set_body_nan(df, bad_rows_body)
    df = set_yolk_nan(df, bad_rows_yolk)

    return df


def remove_bad_volumes(df):
    bad_rows_body = set([])
    bad_rows_yolk = set([])

    # In most cases where the total body volume - yolk volume is very small, we have a fish viewed from above
    # at an early stage where the yolk sac is a large bulge on both sides. We should exclude body AND yolk measures
    small_body_volume = df.loc[(df['Body volume[mm3]'] < 0.1)]
    bad_rows_body.update([i for i, row in small_body_volume.iterrows()])
    # In a very few cases we have a fish on its side but curled, with an incorrect body mask. Here we keep the yolk.
    small_body_volume_body_bad = ['D20200413T134121.418199', 'D20200413T142505.217061', 'D20200413T160958.652863']
    bad_rows_yolk.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] not in small_body_volume_body_bad])

    small_body_volume = df.loc[(df['Body volume[mm3]'] >= 0.1) & (df['Body volume[mm3]'] < 0.2)]
    # More fish viewed from above
    topdown = ['D20200413T132124.823283', 'D20200413T132224.699399', 'D20200413T132242.660511',
               'D20200413T132308.130920', 'D20200413T132308.257461', 'D20200413T132310.389335',
               'D20200413T132347.118920', 'D20200413T132426.171722', 'D20200413T132426.294573',
               'D20200413T134031.539882', 'D20200413T134057.077764', 'D20200413T134104.851085',
               'D20200413T134200.251079', 'D20200413T134327.475752', 'D20200413T134338.508854',
               'D20200413T134414.986219', 'D20200413T134417.993497', 'D20200413T134422.004329',
               'D20200413T140432.002526', 'D20200413T142441.057773', 'D20200413T142443.564928',
               'D20200413T142523.772199', 'D20200413T142550.778290', 'D20200413T142745.597527',
               'D20200413T142759.524237', 'D20200413T142834.820506', 'D20200413T151120.921260',
               'D20200413T151154.611330', 'D20200413T151156.628482', 'D20200413T151243.919929',
               'D20200413T151300.627322', 'D20200413T151400.884169', 'D20200413T154821.568456',
               'D20200413T154949.885642', 'D20200413T154950.388498', 'D20200413T155114.466677',
               'D20200413T155124.629977', 'D20200413T155124.755371', 'D20200413T155135.714589',
               'D20200413T155229.704085', 'D20200413T160637.699244', 'D20200413T160715.624156',
               'D20200413T160724.532350', 'D20200413T160828.335577', 'D20200413T160829.594667',
               'D20200414T140245.330043', 'D20200414T140430.638383', 'D20200414T142057.052199',
               'D20200414T150632.313062', 'D20200414T152828.518868', 'D20200413T142808.305903']
    bad_rows_yolk.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] in topdown])
    bad_rows_body.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] in topdown])
    bad_yolk_only = ['D20200413T132300.866919', 'D20200413T134023.632835', 'D20200413T134111.626858',
                     'D20200413T134128.850128', 'D20200413T134232.546243', 'D20200413T134233.046833',
                     'D20200413T134235.318217', 'D20200413T134352.667911', 'D20200413T140446.047422',
                     'D20200413T140449.446395', 'D20200413T142428.586644', 'D20200413T142504.126342',
                     'D20200413T142555.158948', 'D20200413T151155.048354', 'D20200413T151403.640405',
                     'D20200413T154913.479044', 'D20200413T154920.253748', 'D20200413T155008.969072',
                     'D20200413T155009.093467', 'D20200413T155020.497914', 'D20200413T155038.722313',
                     'D20200413T160656.126395', 'D20200413T160702.288083', 'D20200413T160737.696824',
                     'D20200413T161006.815082']
    bad_rows_yolk.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] in bad_yolk_only])
    bad_body_only = ['D20200413T134233.046833', 'D20200413T140405.393257', 'D20200413T140437.397129',
                     'D20200413T142440.931928', 'D20200413T142520.267140', 'D20200413T142555.158948',
                     'D20200413T155110.479016', 'D20200413T160656.126395', 'D20200413T160737.696824']
    bad_rows_body.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] in bad_body_only])

    small_body_volume = df.loc[(df['Body volume[mm3]'] < 0.25) & (df['Date'] != '20200413')]
    bad_yolk_only = ['D20200414T140405.092040', 'D20200414T142344.389441', 'D20200414T152740.932273',
                     'D20200414T152830.146812', 'D20200414T161143.472289', 'D20200414T163611.967446']
    bad_rows_yolk.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] in bad_yolk_only])
    bad_body_only = ['D20200414T140405.092040', 'D20200414T142344.389441', 'D20200414T144824.886660',
                     'D20200414T150634.213385', 'D20200414T152740.932273', 'D20200414T152830.146812',
                     'D20200414T161143.472289', 'D20200414T161308.997782', 'D20200414T163611.967446',
                     'D20200415T171222.959188', 'D20200415T171528.029494']
    bad_rows_body.update(
        [i for i, row in small_body_volume.iterrows() if row['Image ID'] in bad_body_only])

    large_body_volume = df.loc[(df['Body volume[mm3]'] > 0.43)]
    bad_yolk_only = ['D20200413T134346.653962', 'D20200414T161114.270974', 'D20200414T163501.523774',
                     'D20200416T130212.678990', 'D20200417T144938.465908']
    bad_rows_yolk.update(
        [i for i, row in large_body_volume.iterrows() if row['Image ID'] in bad_yolk_only])
    bad_body_only = ['D20200414T150708.964664', 'D20200416T130212.678990', 'D20200416T133951.966673',
                     'D20200416T135751.612941']
    bad_rows_body.update(
        [i for i, row in large_body_volume.iterrows() if row['Image ID'] in bad_body_only])

    large_yolk_volume = df.loc[(df['Yolk volume[mm3]'] > 0.3)]
    bad_yolk_only = ['D20200413T140454.577056', 'D20200413T142528.178707', 'D20200413T142604.946197',
                     'D20200413T142607.669712']
    bad_rows_yolk.update(
        [i for i, row in large_yolk_volume.iterrows() if row['Image ID'] in bad_yolk_only])
    bad_body_only = ['D20200413T134010.321304', 'D20200413T134037.799584', 'D20200413T142528.178707',
                     'D20200413T142604.946197', 'D20200413T142607.669712', 'D20200413T155026.141386',
                     'D20200413T160620.522643', 'D20200413T160924.590704']
    bad_rows_body.update(
        [i for i, row in large_yolk_volume.iterrows() if row['Image ID'] in bad_body_only])

    large_yolk_volume = df.loc[(df['Yolk volume[mm3]'] > 0.1) & (df['Date'] == '20200414')]
    bad_yolk_only = ['D20200414T140151.657766', 'D20200414T142045.344055', 'D20200414T142159.709690', 'D20200414T152740.806802', 'D20200414T161249.932200', 'D20200414T163840.626403']
    bad_rows_yolk.update(
        [i for i, row in large_yolk_volume.iterrows() if row['Image ID'] in bad_yolk_only])

    df = set_body_nan(df, bad_rows_body)
    df = set_yolk_nan(df, bad_rows_yolk)

    return df


def remove_bad_eyes(df):
    bad_rows_eye = set([])

    small_eyes = df.loc[(df['Eye min diameter[mm]'] < 0.2)]
    small_eyes_good = ['D20200413T132241.908137', 'D20200413T134054.631847', 'D20200413T134054.757834',
                       'D20200413T134054.757834', 'D20200413T134257.220115', 'D20200413T134257.220115',
                       'D20200413T134343.642767', 'D20200413T140510.812547', 'D20200413T140510.812547',
                       'D20200413T142638.739164', 'D20200413T142836.951783', 'D20200413T160736.533668',
                       'D20200413T160822.567087', 'D20200413T160829.464362', 'D20200413T160925.467889',
                       'D20200414T140029.474823', 'D20200414T142009.244003', 'D20200414T142009.244003',
                       'D20200414T142011.993450', 'D20200414T142203.599432', 'D20200414T142203.599432',
                       'D20200414T150330.385760', 'D20200414T150406.551999',
                       'D20200414T150406.551999', 'D20200414T152813.603940', 'D20200414T152813.603940',
                       'D20200414T163618.792020', 'D20200414T163815.419437', 'D20200414T163953.800494',
                       'D20200414T164002.213490', 'D20200414T164002.339113', 'D20200415T160753.005830',
                       'D20200415T171127.038699', 'D20200415T171127.038699', 'D20200415T171436.764387',
                       'D20200415T181028.090842', 'D20200416T145524.346384', 'D20200416T145524.346384',
                       'D20200416T151046.944428', 'D20200416T151134.962178', 'D20200416T151339.429175',
                       'D20200417T170421.169757', 'D20200417T170437.148486']
    bad_rows_eye.update([i for i, row in small_eyes.iterrows() if row['Image ID'] not in small_eyes_good])

    large_eyes = df.loc[(df['Eye min diameter[mm]'] > 0.36)]
    large_eyes_bad = ['D20200413T134011.940861', 'D20200413T134318.187137', 'D20200413T134352.415848',
                      'D20200413T142421.070478', 'D20200413T142639.489921', 'D20200413T142645.770074',
                      'D20200413T142645.895466', 'D20200413T151122.819588', 'D20200413T154821.193247',
                      'D20200413T155015.234397', 'D20200413T155049.156400', 'D20200413T160624.754741',
                      'D20200414T142031.689692', 'D20200414T142136.229363', 'D20200414T142254.498900',
                      'D20200414T142333.710836', 'D20200414T142338.475406', 'D20200414T144925.908141',
                      'D20200414T150355.766992', 'D20200414T150406.676222', 'D20200414T150416.795072',
                      'D20200414T150449.034899', 'D20200414T150544.431187', 'D20200414T150612.246176',
                      'D20200414T150612.371143', 'D20200414T150650.273804', 'D20200414T152703.146053',
                      'D20200414T152732.775435', 'D20200414T152744.947111', 'D20200414T152809.465051',
                      'D20200414T152850.531347', 'D20200414T161153.785398', 'D20200414T161220.843937',
                      'D20200414T164000.448592', 'D20200415T160633.858486', 'D20200415T160650.442588',
                      'D20200415T160815.433045', 'D20200415T160819.312976', 'D20200415T160915.903704',
                      'D20200415T160953.203205', 'D20200415T163434.447766', 'D20200415T163718.957144',
                      'D20200415T165154.195765', 'D20200415T165421.506244', 'D20200415T165421.631416',
                      'D20200415T165516.940916', 'D20200415T171253.030670', 'D20200415T171521.947214',
                      'D20200415T175320.976230', 'D20200415T181123.743499', 'D20200415T181132.143425',
                      'D20200415T181136.283614', 'D20200415T181136.409762', 'D20200415T181216.298019',
                      'D20200415T181216.423005', 'D20200416T130228.544057', 'D20200416T130239.847776',
                      'D20200416T130314.387700', 'D20200416T132042.699809', 'D20200416T132223.816485',
                      'D20200416T133846.049960', 'D20200416T133909.502304', 'D20200416T133952.971049',
                      'D20200416T134001.244593', 'D20200416T134120.626171', 'D20200416T135826.575867',
                      'D20200416T135833.100135', 'D20200416T140014.151281', 'D20200416T145330.179651',
                      'D20200416T152656.673737', 'D20200417T144833.022830', 'D20200417T150803.610761',
                      'D20200417T150827.327102', 'D20200417T150837.616157', 'D20200417T150845.538996',
                      'D20200417T150942.840116', 'D20200417T151001.525353', 'D20200417T152546.655509',
                      'D20200417T152549.916925', 'D20200417T152556.350448', 'D20200417T154526.400177',
                      'D20200417T154552.363662', 'D20200417T154607.078064', 'D20200417T154607.202530',
                      'D20200417T154648.408919', 'D20200417T160643.087343', 'D20200417T160655.000219',
                      'D20200417T160904.267258', 'D20200417T160919.168231', 'D20200417T160919.295767']
    bad_rows_eye.update([i for i, row in large_eyes.iterrows() if row['Image ID'] in large_eyes_bad])

    df = set_eye_nan(df, bad_rows_eye)

    return df


def remove_bad_measurements_1246(df):
    # Body areas larger than 2 are all spurious
    bad_rows_body = set(df[df['Body area[mm2]'] > 2].index.tolist())

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
    bad_rows_body.update([i for i, row in small_areas.iterrows() if row['Image ID'] not in small_area_good])

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
    bad_rows_body.update([i for i, row in large_areas.iterrows() if row['Image ID'] not in large_area_good])

    # Body lengths larger than 7 are all spurious
    bad_rows_body.update(df[df['Myotome length[mm]'] > 7].index.tolist())

    long_bodies_good = ['D20200414T150328.756238',
                        'D20200415T154649.792826',
                        'D20200415T160856.307877',
                        'D20200416T134046.817531',
                        'D20200417T152522.705000',
                        'D20200417T154418.597554'
                        ]
    long_bodies = df[df['Myotome length[mm]'] > 5.5]
    bad_rows_body.update([i for i, row in long_bodies.iterrows() if row['Image ID'] not in long_bodies_good])

    df = set_body_nan(df, bad_rows_body)

    large_eye_diams_good = ['D20200413T151122.819588',
                            'D20200414T144922.581310',
                            'D20200417T151106.196059',
                            ]
    large_eye_diams = df[df['Eye min diameter[mm]'] > 0.4]
    bad_rows_eyes = set([i for i, row in large_eye_diams.iterrows() if row['Image ID'] not in large_eye_diams_good])

    # Min eye diameters smaller than 0.05 are all spurious
    bad_rows_eyes.update(df[df['Eye min diameter[mm]'] < 0.05].index.tolist())

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
    bad_rows_eyes.update([i for i, row in small_eye_diams.iterrows() if row['Image ID'] not in small_eye_diams_good])

    df = set_eye_nan(df, bad_rows_eyes)

    short_yolks_good = ['D20200417T150858.091196', 'D20200417T164854.342576']
    short_yolks = df[df['Yolk length[mm]'] < 0.2]
    bad_rows_yolks = set([i for i, row in short_yolks.iterrows() if row['Image ID'] not in short_yolks_good])

    short_yolks_early_good = ['D20200414T140104.388475',
                              'D20200414T142119.819073',
                              'D20200414T144734.929584',
                              'D20200414T150422.810736']
    short_yolks_early = df.loc[((df['Date'] == '20200413') |
                                (df['Date'] == '20200414')) &
                               (df['Yolk length[mm]'] < 0.4)]
    bad_rows_yolks.update(
        [i for i, row in short_yolks_early.iterrows() if row['Image ID'] not in short_yolks_early_good])

    long_yolks_good = ['D20200413T134010.069907',
                       'D20200413T134010.321304',
                       'D20200413T134033.546619',
                       'D20200413T134034.044982',
                       'D20200413T134228.154395',
                       'D20200413T134232.919814',
                       'D20200413T134244.059384',
                       'D20200413T134245.939653',
                       'D20200413T134257.220115',
                       'D20200413T134430.506310',
                       'D20200413T140211.024382',
                       'D20200413T140402.381515',
                       'D20200413T140428.740520',
                       'D20200413T140523.172530',
                       'D20200413T140626.459179',
                       'D20200413T142511.484410',
                       'D20200413T142512.612858',
                       'D20200413T142528.530846',
                       'D20200413T142547.520443',
                       'D20200413T142627.858712',
                       'D20200413T142706.877499',
                       'D20200413T142728.660367',
                       'D20200413T142754.880046',
                       'D20200413T142811.302182',
                       'D20200413T142812.681772',
                       'D20200413T142843.082694',
                       'D20200413T142850.827841',
                       'D20200413T142852.458463',
                       'D20200413T142852.582381',
                       'D20200413T142922.817177',
                       'D20200413T151353.634823',
                       'D20200413T154820.813391',
                       'D20200413T154838.563275'
                       ]
    long_yolks = df[df['Yolk length[mm]'] > 1.2]
    bad_rows_yolks.update([i for i, row in long_yolks.iterrows() if row['Image ID'] not in long_yolks_good])

    long_yolks_20200417_good = ['D20200417T145009.193963',
                                'D20200417T145038.403729',
                                'D20200417T145123.143566',
                                'D20200417T150908.920277',
                                'D20200417T152435.675921']
    long_yolks_20200417 = df.loc[(df['Date'] == '20200417') & (df['Yolk length[mm]'] > 0.7)]
    bad_rows_yolks.update(
        [i for i, row in long_yolks_20200417.iterrows() if row['Image ID'] not in long_yolks_20200417_good])

    big_yolks_late_good = ['D20200415T154408.380776',
                           'D20200415T154456.491491',
                           'D20200415T154646.531343',
                           'D20200415T160756.147999',
                           'D20200415T160854.041938',
                           'D20200415T160929.580793',
                           'D20200415T163344.813225',
                           'D20200415T163637.708566',
                           'D20200415T165539.463892',
                           'D20200415T165539.590512',
                           'D20200415T171200.400383',
                           'D20200415T171205.417922',
                           'D20200415T171342.849738',
                           'D20200416T130157.389719',
                           'D20200416T130322.292713',
                           'D20200416T132302.034973',
                           'D20200417T144821.896526',
                           'D20200417T145123.143566']
    big_yolks_late = df.loc[((df['Date'] == '20200415') |
                             (df['Date'] == '20200416') |
                             (df['Date'] == '20200417')) & (df['Yolk area[mm2]'] > 0.16)]
    bad_rows_yolks.update([i for i, row in big_yolks_late.iterrows() if row['Image ID'] not in big_yolks_late_good])

    df = set_yolk_nan(df, bad_rows_yolks)

    return df


def set_body_nan(df, idx):
    df.loc[idx, 'Body area[mm2]'] = np.nan
    df.loc[idx, 'Myotome length[mm]'] = np.nan
    df.loc[idx, 'Total body volume[mm3]'] = np.nan
    df.loc[idx, 'Body volume[mm3]'] = np.nan

    return df


def set_eye_nan(df, idx):
    df.loc[idx, 'Eye min diameter[mm]'] = np.nan
    df.loc[idx, 'Eye max diameter[mm]'] = np.nan
    df.loc[idx, 'Eye area[mm2]'] = np.nan

    return df


def set_yolk_nan(df, idx):
    df.loc[idx, 'Yolk area[mm2]'] = np.nan
    df.loc[idx, 'Yolk length[mm]'] = np.nan
    df.loc[idx, 'Yolk height[mm]'] = np.nan
    df.loc[idx, 'Yolk volume[mm3]'] = np.nan
    df.loc[idx, 'Body volume[mm3]'] = np.nan

    return df


main()
