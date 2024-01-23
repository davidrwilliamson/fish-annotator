import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from bioinfokit.analys import stat as bstat


class OilDepletePlots:

    @staticmethod
    def get_colors_marker(t: str):
        t_type = t.split(' ')[0]
        if t_type == 'SW':
            color = 'blue'
            marker = 'o'
        elif t_type == 'Statfjord':
            color = 'red'
            marker = 'P'
        elif t_type == 'ULSFO':
            color = 'black'
            marker = 'X'
        else:
            color = 'green'
            marker = 's'

        return color, marker

    @staticmethod
    def add_hue_group(df: pd.DataFrame) -> pd.DataFrame:
        df['Group'] = df['Treatment'].apply(lambda x: x.split(' ')[0])

        return df

    @staticmethod
    def plot_measurements_both_methods(df: pd.DataFrame, attribute):
        figsize = (20, 10)
        linewidth = 2
        label_fontsize = 24
        title_fontsize = 24
        tick_fontsize = 24
        x = 'Treatment'
        y = attribute

        data = OilDepletePlots.add_hue_group(df)

        order = ['SW 4d (ctrl)', 'SW 60d-1', 'SW 60d-2', 'SW 60d-3',
                 'Statfjord 4d-1', 'Statfjord 4d-2', 'Statfjord 14d', 'Statfjord 21d', 'Statfjord 40d',
                 'Statfjord 60d',
                 'ULSFO 28d-1', 'ULSFO 28d-2', 'ULSFO 28d-3', 'ULSFO 60d-1', 'ULSFO 60d-2']

        fig, ax = plt.subplots(figsize=figsize)
        groups = ['SW', 'Statfjord', 'ULSFO']
        palettes = ['Blues', 'Reds', 'Greys']

        for i, g in enumerate(groups):
            # sns.violinplot(x=x, y=y, data=data[data['Group'] == g]), ax=ax, hue='Method', palette=palettes[i],
            #                order=order, inner='stick', split=True, dodge=True)
            sns.violinplot(x=x, y=y, data=data[data['Group'] == g], ax=ax, hue='Method',
                           palette=palettes[i], order=order, inner='stick', split=True, dodge=True)
        # sns.swarmplot(x=x, y=y, data=data, ax=ax, hue='Method', dodge=True, color='black', order=order, size=5)

        # ax.set_title('{}'.format(attribute.split('[')[0]), fontsize=title_fontsize)

        # Tick appearance
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(y_lims[0], y_lims[1], 0.01)))
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', left=True)
        ax.tick_params(axis='both', which='minor', length=4)
        ax.tick_params(axis='both', which='major', length=8, labelsize=tick_fontsize)
        ax.tick_params(axis='both', which='both', width=linewidth)

        ylabel = ax.get_ylabel()
        # ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_ylabel('', fontsize=label_fontsize)
        xlabel = ax.get_xlabel()
        # ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_xlabel('', fontsize=label_fontsize)

        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', rotation_mode='anchor')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(linewidth)

        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)

        ax.get_legend().remove()

        plt.tight_layout()
        plt.savefig('/mnt/6TB_Media/PhD Work/oil_paper_june_23/measurements/measurements_both_{}'.format(attribute))
        # plt.show()

    @staticmethod
    def plot_cross_compare(df: pd.DataFrame, df_bhh: pd.DataFrame, attribute: str) -> None:
        treatments = df['Treatment'].unique()

        figsize = (12, 12)
        fig, ax = plt.subplots(figsize=figsize)
        title_fontsize = 24
        label_fontsize = 24
        tick_fontsize = 24

        means_aut = np.array([df[df['Treatment'] == treatment][attribute].mean() for treatment in treatments])
        means_man = np.array([df_bhh[df_bhh['Treatment'] == treatment][attribute].mean() for treatment in treatments])
        std_aut = np.array([df[df['Treatment'] == treatment][attribute].std() for treatment in treatments])
        std_man = np.array([df_bhh[df_bhh['Treatment'] == treatment][attribute].std() for treatment in treatments])

        for i, t in enumerate(treatments):
            a_m = means_aut[i]
            m_m = means_man[i]
            a_s = std_aut[i]
            m_s = std_man[i]

            color, marker = OilDepletePlots.get_colors_marker(t)

            plt.plot(m_m, a_m, color=color, marker=marker, markersize=20)

            y_lo, y_hi = a_m - a_s, a_m + a_s
            x_lo, x_hi = m_m - m_s, m_m + m_s

            plt.plot([x_lo, x_hi], [a_m, a_m], color=color)
            plt.plot([m_m, m_m], [y_lo, y_hi], color=color)

            plt.plot([-10, 10], [-10, 10], linestyle='--')

        # ax.set_title('{}'.format(attribute.split('[')[0]), fontsize=title_fontsize)
        # ax.set_xlabel('Manual method', fontsize=label_fontsize)
        # ax.set_ylabel('Automated method', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        margin = max(means_aut.mean(), means_man.mean()) * 0.0
        min_val = min(means_aut.min() - std_aut.max(), means_man.min() - std_man.max())
        min_val = max(min_val, 0.0)
        max_val = max(means_aut.max() + std_aut.max(), means_man.max() + std_man.max())

        ax.set(xlim=(min_val - margin, max_val + margin))
        ax.set(ylim=(min_val - margin, max_val + margin))

        plt.tight_layout()
        plt.savefig('/mnt/6TB_Media/PhD Work/oil_paper_june_23/comparison/cross_compare_{}'.format(attribute))
        # plt.show()

    @staticmethod
    def prepare_latex_table(df: pd.DataFrame) -> str:
        header = '\\begin{table}\n' \
                 '\\centering\n' \
                 '{\\tiny\n' \
                 '\\begin{tabular}{lllllll}\n' \
                 '\\toprule\n' \
                 'Treatment & Eye area (mm$^2$) & Standard length (mm) & Structural body area (mm$^2$) & ' \
                 'Total body area (mm$^2$) & Yolk area (mm$^2$) & Yolk fraction \\\\\n' \
                 '\\midrule\n'
        footer = '\\bottomrule\n' \
                 '\\end{tabular}\n' \
                 '}\n' \
                 '\\caption{}\n' \
                 '\\label{tab:}\n' \
                 '\\end{table}'
        content = ''

        treatments = df['Treatment'].unique()
        attributes = ['Eye area[mm2]', 'Total body area[mm2]', 'Structural body area[mm2]', 'Standard length[mm]',
                      'Yolk area[mm2]', 'Yolk fraction']

        ## Get significance symbols
        def sig_symbol(p: float) -> str:
            if p < 0.0001:
                return '\\textasteriskcentered\\textasteriskcentered\\textasteriskcentered\\textasteriskcentered'
            elif p < 0.001:
                return '\\textasteriskcentered\\textasteriskcentered\\textasteriskcentered'
            elif p < 0.01:
                return '\\textasteriskcentered\\textasteriskcentered'
            elif p < 0.05:
                return '\\textasteriskcentered'
            else:
                return ''

        symbols = {}
        for attribute in attributes:
            df_melt = pd.melt(df.reset_index(), id_vars=['Treatment'], value_vars=[attribute], value_name='Value')
            df_melt = df_melt.drop(columns='variable')
            df_melt = df_melt[df_melt['Value'].notna()]
            control = 'SW 4d (ctrl)'
            control_array = df_melt[df_melt['Treatment'] == control]['Value']
            t = [df_melt[df_melt['Treatment'] == t]['Value'] for t in treatments]
            res = stats.dunnett(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13],
                                t[14], control=control_array)
            sym_list = []
            for p in res.pvalue:
                sym_list.append(sig_symbol(p))

            symbols[attribute] = sym_list

        ## Calculate means and stds, put together the row string
        for i, treatment in enumerate(treatments):
            name = treatment
            means = []
            stds = []
            syms = []
            for attribute in attributes:
                means.append(df[df['Treatment'] == treatment][attribute].mean())
                stds.append(df[df['Treatment'] == treatment][attribute].std())
                syms.append(symbols[attribute][i])
            table_row = '{} & {:.3f} $\\pm$ {:.3f} {} & {:.2f} $\\pm$ {:.2f} {} & {:.2f} $\\pm$ {:.2f} {} & ' \
                        '{:.2f} $\\pm$ {:.2f} {} & {:.3f} $\\pm$ {:.3f} {} & {:.3f} $\\pm$ {:.3f} {}'.\
                format(name, means[0], stds[0], syms[0], means[1], stds[1], syms[1], means[2], stds[2], syms[2],
                       means[3], stds[3], syms[3], means[4], stds[4], syms[4], means[5], stds[5], syms[5])
            content += table_row
            content += '\\\\\n'

        return '{}{}{}'.format(header, content, footer)


    @staticmethod
    def plot_compare_difference_from_mean(df: pd.DataFrame, df_bhh: pd.DataFrame, attribute: str, normalize: bool) -> None:
        treatments = df['Treatment'].unique()

        figsize = (18, 12)
        fig, ax = plt.subplots(figsize=figsize)
        title_fontsize = 24
        label_fontsize = 32
        tick_fontsize = 24

        means_aut = np.array([df[df['Treatment'] == treatment][attribute].mean() for treatment in treatments])
        means_man = np.array([df_bhh[df_bhh['Treatment'] == treatment][attribute].mean() for treatment in treatments])
        av_of_means = (means_aut + means_man) / 2.0
        if normalize:
            means_aut /= av_of_means
            means_man /= av_of_means

        diff_of_means = means_aut - means_man
        std_of_means = diff_of_means.std()

        colors_markers = [OilDepletePlots.get_colors_marker(t) for t in treatments]
        colors = [c[0] for c in colors_markers]
        markers = [c[1] for c in colors_markers]

        for i in range(len(means_aut)):
            plt.plot(av_of_means[i], diff_of_means[i], color=colors[i], marker=markers[i], markersize=20)

        # Linear regression
        x = av_of_means.reshape((-1, 1))
        y = diff_of_means
        model = LinearRegression().fit(x, y)
        plt.plot(x, model.predict(x), color='black', linestyle='-')
        r_sq = model.score(x, y)
        # This weird positioning argument is in order to put the R2 just above the left of each line
        ax.annotate('R2 = {:.3f}'.format(r_sq), xy=(ax.dataLim.minposx, model.predict(sorted(x))[0]), xytext=(0, 20),
                    textcoords='offset pixels', fontsize=tick_fontsize)
        # ax.annotate('R2 = {:.3f}'.format(r_sq), xy=(x[-1], model.predict(x)[-1]), xytext=(x[-1], model.predict(x)[-1]),
        #             fontsize=tick_fontsize)

        # plt.axhline(y=0.0, color='black', linestyle='-')
        plt.axhline(y=diff_of_means.mean(), color='gray', linestyle='--')
        ax.annotate('Mean', xy=(av_of_means.max(), diff_of_means.mean()), xytext=(av_of_means.max(),
                                                                                  diff_of_means.mean() + 0.0005),
                    fontsize=tick_fontsize, horizontalalignment='right')
        plt.axhline(y=diff_of_means.mean() + 2 * std_of_means, color='gray', linestyle='--')
        ax.annotate('Mean + 2 * std. dev.', xy=(av_of_means.max(), diff_of_means.mean() + 2 * std_of_means),
                    xytext=(av_of_means.max(), diff_of_means.mean() + 2 * std_of_means * 1.02),
                    fontsize=tick_fontsize, horizontalalignment='right')
        plt.axhline(y=diff_of_means.mean() + 2 * -std_of_means, color='gray', linestyle='--')
        ax.annotate('Mean - 2 * std. dev.', xy=(av_of_means.max(), diff_of_means.mean() + 2 * -std_of_means),
                    xytext=(av_of_means.max(), diff_of_means.mean() + 2 * -std_of_means * 0.98),
                    fontsize=tick_fontsize, horizontalalignment='right')

        # ax.set_title('{}'.format(attribute.split('[')[0]), fontsize=title_fontsize)
        # ax.set_xlabel('Average of automated and manual meaurements', fontsize=label_fontsize)
        # ax.set_ylabel('Difference in automated and manual measurements', fontsize=label_fontsize)
        # ax.set_xlabel('', fontsize=label_fontsize)
        # ax.set_ylabel('', fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)

        plt.tight_layout()
        # plt.savefig('/mnt/6TB_Media/PhD Work/oil_paper_june_23/comparison/difference_means_normalized_{}'.format(attribute))
        plt.savefig('/mnt/6TB_Media/PhD Work/oil_paper_june_23/comparison/difference_means_{}'.format(attribute))
        # plt.show()
    @staticmethod
    def test_replicates(df: pd.DataFrame, attribute: str):
        groups = [['Statfjord 4d-1', 'Statfjord 4d-3'],
                 ['SW 4d-3', 'SW 60d-2', 'SW 60d-3', 'SW 60d-4'],
                 ['ULSFO 28d-1', 'ULSFO 28d-2', 'ULSFO 28d-4'],
                 ['ULSFO 60d-1', 'ULSFO 60d-2']]
        print(attribute)
        for group in groups:
            curr_df = df[df['Treatment'].apply(lambda x: x in group)]

            df_melt = pd.melt(curr_df.reset_index(), id_vars=['Treatment'], value_vars=[attribute], value_name='Value')
            df_melt = df_melt.drop(columns='variable')
            df_melt = df_melt[df_melt['Value'].notna()]

            res = bstat()
            res.anova_stat(df=df_melt, res_var='Value', anova_model='Value ~ C(Treatment)')
            print(group, res.anova_model_out.f_pvalue)

    @staticmethod
    def anova(df: pd.DataFrame, attribute: str):
        df_melt = pd.melt(df.reset_index(), id_vars=['Treatment'], value_vars=[attribute], value_name='Value')
        df_melt = df_melt.drop(columns='variable')
        df_melt = df_melt[df_melt['Value'].notna()]

        # Bioinfokit approach
        res = bstat()
        res.anova_stat(df=df_melt, res_var='Value', anova_model='Value ~ C(Treatment)')

        fig, (ax1, ax2) = plt.subplots(1, 2)
        sm.qqplot(res.anova_std_residuals, line='45', ax=ax1)
        ax2.hist(res.anova_std_residuals, bins='auto', histtype='bar', ec='k')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        fig.suptitle('{}'.format(attribute))
        plt.show()

        model = sm.formula.ols('Value ~ C(Treatment)'.format(attribute), data=df_melt).fit()
        W, p_shapiro = stats.shapiro(model.resid)
        print('Shapiro-Wilks, {}: W={:.2f}, p={:.2f}'.format(attribute, W, p_shapiro))

        control = 'SW 4d (ctrl)'
        treatments = df['Treatment'].unique()
        control_array = df_melt[df_melt['Treatment'] == control]['Value']
        t = [df_melt[df_melt['Treatment'] == t]['Value'] for t in treatments]
        res = stats.dunnett(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], control=control_array)

        # dunnett_stats = zip(treatments, res.pvalue, res.statistic)
        # for i in dunnett_stats:
        #     print(i[2])

        foo = -1

        # res.levene(df=df_melt, res_var='Value')

        # model = sm.formula.ols('Value ~ C(Treatment)'.format(attribute), data=df_melt).fit()
        # anova_table = sm.stats.anova_lm(model, typ=2)

        # sm.qqplot(model.resid, line='45')

    @staticmethod
    def prepare_method_difference_table(df: pd.DataFrame, df_bhh: pd.DataFrame, attributes: list, normalize: bool) -> str:
        treatments = df['Treatment'].unique()

        content = ''
        if not normalize:
            for treatment in treatments:
                content += '{} '.format(treatment)
                for attribute in attributes:
                    mean_aut = df[df['Treatment'] == treatment][attribute].mean()
                    mean_man = df_bhh[df_bhh['Treatment'] == treatment][attribute].mean()

                    diff_of_means = mean_aut - mean_man
                    # std_of_means = diff_of_means.std()

                    content += '& {:.3f} '.format(diff_of_means)
                content += ' \\\\\n'

        overall = []
        if normalize:
            columns = []
        for attribute in attributes:
            means_aut = np.array([df[df['Treatment'] == treatment][attribute].mean() for treatment in treatments])
            means_man = np.array([df_bhh[df_bhh['Treatment'] == treatment][attribute].mean() for treatment in treatments])

            diff_of_means = means_aut - means_man

            if normalize:
                av_of_means = (means_aut + means_man) / 2.0
                diff_of_means /= av_of_means
                columns.append(diff_of_means)

            overall.append(np.abs(diff_of_means).mean())

        if normalize:
            for j, treatment in enumerate(treatments):
                content += '{} '.format(treatment)
                for i in range(len(columns)):
                    content += '& {:.3f} '.format(columns[i][j])
                content += ' \\\\\n'

        content += '\\textbf{Overall mean}'
        for o in overall:
            content += '& {:.3f} '.format(o)
        content += ' \\\\\n'

        header = '\\begin{table}\n' \
                 '\\centering\n' \
                 '{\\tiny\n' \
                 '\\begin{tabular}{lcccccc}\n' \
                 '\\toprule\n' \
                 'Treatment & Eye area (mm$^2$) & Standard length (mm) & Structural body area (mm$^2$) & ' \
                 'Total body area (mm$^2$) & Yolk area (mm$^2$) & Yolk fraction \\\\\n' \
                 '\\midrule\n'
        footer = '\\bottomrule\n' \
                 '\\end{tabular}\n' \
                 '}\n' \
                 '\\caption{}\n' \
                 '\\label{tab:}\n' \
                 '\\end{table}'

        return '{}{}{}'.format(header, content, footer)

    @staticmethod
    def survival_hatching_plots(df: pd.DataFrame, column: str):
        old_names = ['Statfjord 4d-3', 'Statfjord 21d-2', 'Statfjord 40d-4', 'Statfjord 60d-2', 'ULSFO 28d-4']
        new_names = ['Statfjord 4d-2', 'Statfjord 21d',  'Statfjord 40d', 'Statfjord 60d', 'ULSFO 28d-3']
        for i in range(len(old_names)):
            df.loc[df['Treatment'] == old_names[i], ['Treatment']] = new_names[i]
        # order = ['SW 4d (ctrl)', 'SW 60d-1', 'SW 60d-2', 'SW 60d-3',
        #          'Statfjord 4d-1', 'Statfjord 4d-2', 'Statfjord 14d', 'Statfjord 21d', 'Statfjord 40d',
        #          'Statfjord 60d',
        #          'ULSFO 28d-1', 'ULSFO 28d-2', 'ULSFO 28d-3', 'ULSFO 60d-1', 'ULSFO 60d-2']
        data = OilDepletePlots.add_hue_group(df)
        palette = {'SW': 'tab:blue', 'Statfjord': 'tab:red', 'ULSFO': 'tab:gray'}
        hue_order = ['SW', 'Statfjord', 'ULSFO']

        figsize = (16, 6)
        fig, ax = plt.subplots(figsize=figsize)
        title_fontsize = 24
        label_fontsize = 24
        tick_fontsize = 16

        ax = sns.barplot(data=data, x='Treatment', y=column, hue='Group', palette=palette, hue_order=hue_order, dodge=False)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel('{} Rate (%)'.format(column), fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        ax.set_ylim([0, 100])
        ax.get_legend().remove()
        plt.tight_layout()
        # plt.show()
        plt.savefig('/mnt/6TB_Media/PhD Work/oil_paper_june_23/{}_rate'.format(column))


    @staticmethod
    def count_fish(df: pd.DataFrame):
        df = df.drop(df[df.isnull().sum(axis=1) == 10].index)  # No valid measurements
        unique_fish = df.groupby(by=['Image ID', 'Treatment']).apply(lambda x: len(x['Fish ID'].unique()))

        count_df = pd.DataFrame()
        count_df['Fish'] = unique_fish.groupby('Treatment').sum()
        attributes = ['Eye area[mm2]', 'Total body area[mm2]', 'Structural body area[mm2]', 'Standard length[mm]',
                      'Yolk area[mm2]', 'Yolk fraction']

        for attribute in attributes:
            count_df[attribute] = df.groupby('Treatment', dropna=True)[attribute].count()

        count_df.loc['Mean'] = count_df.mean()
        table = count_df.to_latex(float_format='{:0.0f}'.format, column_format='lccccccc')
        foo = -1