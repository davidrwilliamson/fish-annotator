import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ptitprince as pt

# This and the other utils files are extremely dirty.
# They can be expected to break and/or cause irreparable data loss without warning.
# Use with caution or, better yet, not at all.


def prepare_dataframe():
    stats_file = '/home/dave/Desktop/circle_stats.csv'
    sep = ';'
    col_names = ['folder', 'file', 'circles']

    df = pd.read_csv(stats_file, sep=sep, names=col_names)
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
    df['circles'] = df['circles'].apply(lambda x: [x[i:i+3] for i in range(0, len(x), 3)])
    df = df.explode('circles')
    # Separate out into circle centre position and radius
    df['pos'] = df['circles'].apply(lambda x: x[0:2])
    df['radius'] = df['circles'].apply(lambda x: x[2])

    # Create Date and Treatment columns based on folder
    df['Date'] = df['folder'].apply(lambda x: x.split('/')[0])
    df['Treatment'] = df['folder'].apply(lambda x: x.split('/')[1])
    # Collapse identical treatments with different names into one
    df.loc[df['Treatment'] == '2', ['Treatment']] = '1'
    df.loc[df['Treatment'] == '3', ['Treatment']] = '1'
    df.loc[df['Treatment'] == 'DCA-ctrl-2', ['Treatment']] = 'DCA-ctrl'

    # Convert radius to millimetres
    df['Radius[mm]'] = df['radius'].apply(lambda x: float(x) / 287.0)
    df['Diameter[mm]'] = df['Radius[mm]'].apply(lambda x: x * 2)
    df['Volume[mm3]'] = df['Radius[mm]'].apply(lambda x: (4.0 / 3.0) * np.pi * (x * x * x))

    # Get rid of bad measurements
    df = drop_bad_measurements(df)
    # Delete unnecessary columns
    df = df.drop(columns=['circles', 'folder'])

    return df


def plot_swarmplot(x, y, df, title):
    figsize = (9, 9)
    ort = 'v'
    pal = sns.color_palette('Set2')
    # pal = pal[0]

    f, ax = plt.subplots(figsize=figsize)
    ax = pt.stripplot(x=x, y=y, data=df, palette=pal, edgecolor="white",
                      size=3, jitter=1, zorder=0, orient=ort, move=0.25)
    # ax = sns.swarmplot(x=dx, y=dy, data=df, color='gray', zorder=15, orient=ort)
    ax = sns.boxplot(x=x, y=y, data=df, palette=pal, width=.15, zorder=10,
                     showcaps=True, boxprops={'linewidth': 1, "zorder": 10},
                     showfliers=False, whiskerprops={'linewidth': 1, "zorder": 10},
                     saturation=1, orient=ort)#, order=['20200408', '20200409', '20200410', '20200411', '20200412'])
                     # order=['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410', '20200411', '20200412'])

    ax.set_title(title)
    # ax.set(ylim=(1.0, 1.6))

    plt.show()


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


def plot_means_heatmap(df):
    attributes = ['Volume[mm3]']
    # dates = ['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410', '20200411', '20200412']
    # treatments = ['1', 'DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
    dates = ['20200408', '20200409', '20200410', '20200411', '20200412']
    treatments = ['DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']

    for a in attributes:
        means = pd.DataFrame(columns=['Date', 'Treatment', a])
        for d in dates:
            for t in treatments:
                all_means = df.loc[(df['Treatment'] == t) & (df['Date'] == d)].mean()
                means = means.append({'Date': d, 'Treatment': t, a: all_means[a]}, ignore_index=True)
        pivot = means.pivot('Treatment', 'Date', a)
        pivot = pivot.reindex(treatments)
        figsize = (9, 9)
        f, ax = plt.subplots(figsize=figsize)
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
    data = data.sort_values(by=['Date'])
    treatment_names = {'DCA-ctrl': 'Control',
                       'DCA-0,15': '8 μg/L',
                       'DCA-0,31': '27 μg/L',
                       'DCA-0,62': '108 μg/L',
                       'DCA-1,25': '220 μg/L',
                       'DCA-2,50': '343 μg/L',
                       'DCA-5,00': '747 μg/L'}
    dates = data[x].unique()
    date_ordinal = data['Date'].copy()
    date_strings = ['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410', '20200411', '20200412']
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
        plt.plot([i - 0.25, i + 0.25], [means[i], means[i]], linewidth=linewidth * 1.5, color='black')
        # Whisker caps
        plt.plot([i - 0.125, i + 0.125], [y_lo, y_lo], linewidth=linewidth * 1.5, color='black')
        plt.plot([i - 0.125, i + 0.125], [y_hi, y_hi], linewidth=linewidth * 1.5, color='black')

    ax.set(ylim=y_lims)
    # ax.set_title('{} - {}'.format(attribute.split('[')[0], treatment_names[treatment]), fontsize=title_fontsize)
    ax.set_title('Egg volume', fontsize=title_fontsize)

    # Tick appearance
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(y_lims[0], y_lims[1], 0.01)))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', left=True)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=8, labelsize=label_fontsize)
    ax.tick_params(axis='both', which='both', width=linewidth)

    ax.set_xlabel('Days post fertilisation', fontsize=label_fontsize)
    ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    new_labels = ['3', '4', '5', '6', '7', '8', '9', '10', '11']
    ax.set_xticklabels(new_labels)
    # ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)

    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/dave/Desktop/tox_plots/{}_{}.png'.format(attribute, treatment))


def main():
    df: pd.DataFrame = prepare_dataframe()

    # df.to_csv('/home/dave/Desktop/egg_stats.csv')

    # plot_means_heatmap(df)

    dates = ['20200404', '20200405', '20200406', '20200407', '20200408', '20200409', '20200410', '20200411', '20200412']
    # treatments = ['1', 'DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
    treatments = ['1']#, 'DCA-ctrl', 'DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']

    ylims = (0.0, 1.6)
    for treatment in treatments:
        journal_plot(df, 'Volume[mm3]', treatment, ylims)

    treatment = 'DCA-ctrl'
    date = '20200410'
    # plot_swarmplot('Date', 'radius', df, 'Egg volume by date')
    # plot_swarmplot('Date', 'Volume[mm^3]', df[df['Treatment'] == treatment], 'Egg volume by date, treatment = {}'.format(treatment))
    # plot_swarmplot('Treatment', 'Volume[mm^3]', df[df['Date'] == date], 'Egg volume by treatment, Date={}'.format(date))


main()
