import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

# This and the other utils files are extremely dirty.
# They can be expected to break and/or cause irreparable data loss without warning.
# Use with caution or, better yet, not at all.

def prepare_dataframe():
    stats_file = '/home/davidw/Desktop/circle_stats.csv'
    sep = ';'
    col_names = ['folder', 'file', 'circles']

    df = pd.read_csv(stats_file, sep=sep, names=col_names)
    # Remove brackets & whitespace
    df['circles'] = df['circles'].str.replace('[', '')
    df['circles'] = df['circles'].str.replace(']', '')
    df['circles'] = df['circles'].str.replace(' ', '')
    # Convert to list of floats
    df['circles'] = df['circles'].str.split(',')
    df['circles'] = df['circles'].apply(lambda x: list(map(float, x)))
    # Now every third float is a radius
    df['radius'] = df['circles'].apply(lambda x: x[2::3])
    # Put each radius on its own line
    df = df.explode('radius')

    return df


def main():
    df: pd.DataFrame = prepare_dataframe()

    days = [['20200404/3'],
            ['20200405/1'],
            ['20200406/1'],
            ['20200407/1'],
            ['20200408/1', '20200408/DCA-ctrl', '20200408/DCA-0,15', '20200408/DCA-0,31', '20200408/DCA-0,62', '20200408/DCA-1,25', '20200408/DCA-2,50', '20200408/DCA-5,00'],
            ['20200409/1', '20200409/DCA-ctrl', '20200409/DCA-0,15', '20200409/DCA-0,31', '20200409/DCA-0,62', '20200409/DCA-1,25', '20200409/DCA-2,50', '20200409/DCA-5,00'],
            ['20200410/1', '20200410/DCA-ctrl', '20200410/DCA-0,15', '20200410/DCA-0,31', '20200410/DCA-0,62', '20200410/DCA-1,25', '20200410/DCA-2,50', '20200410/DCA-5,00'],
            ['20200411/1', '20200411/DCA-ctrl', '20200411/DCA-0,15', '20200411/DCA-0,31', '20200411/DCA-0,62', '20200411/DCA-1,25', '20200411/DCA-2,50', '20200411/DCA-5,00']]

    treatments = [['20200404/3', '20200405/1', '20200406/1', '20200407/1','20200408/1', '20200409/1', '20200410/1', '20200411/1'],
                  ['20200408/DCA-ctrl', '20200409/DCA-ctrl', '20200410/DCA-ctrl', '20200411/DCA-ctrl'],
                  ['20200408/DCA-0,15', '20200409/DCA-0,15', '20200410/DCA-0,15', '20200411/DCA-0,15'],
                  ['20200408/DCA-0,31', '20200409/DCA-0,31', '20200410/DCA-0,31', '20200411/DCA-0,31'],
                  ['20200408/DCA-0,62', '20200409/DCA-0,62', '20200410/DCA-0,62', '20200411/DCA-0,62'],
                  ['20200408/DCA-1,25', '20200409/DCA-1,25', '20200410/DCA-1,25', '20200411/DCA-1,25'],
                  ['20200408/DCA-2,50', '20200409/DCA-2,50', '20200410/DCA-2,50', '20200411/DCA-2,50'],
                  ['20200408/DCA-5,00', '20200409/DCA-5,00', '20200410/DCA-5,00', '20200411/DCA-5,00']]

    df['day'] = None
    df['treatment'] = None

    i = 1
    for day in days:
        temp_df = df[df['folder'].isin(day)]
        print('Day {0}: mean = {1:.2f}, std. dev = {2:.2}'.format(i, temp_df['radius'].mean(), temp_df['radius'].std()))
        df['day'] = np.where(df['folder'].isin(days[i - 1]), i, df['day'])
        i += 1

    t_labels = ['No treatment', 'DCA-ctrl', 'DCA-0,15', 'DCA-0.31', 'DCA-0,62', 'DCA-1,25', 'DCA-2,50', 'DCA-5,00']
    i = 0
    for t in treatments:
        temp_df = df[df['folder'].isin(t)]
        print('Treatment {0}: mean = {1:.2f}, std. dev = {2:.2}'.format(t_labels[i], temp_df['radius'].mean(), temp_df['radius'].std()))
        df['treatment'] = np.where(df['folder'].isin(t), t_labels[i], df['treatment'])
        i += 1

    day_plot = sb.boxplot(x='day', y='radius', data=df)
    # treat_plot = sb.boxplot(x='treatment', y='radius', data=df)


main()
