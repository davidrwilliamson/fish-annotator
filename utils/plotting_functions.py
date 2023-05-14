import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FixedLocator, LinearLocator)
import statsmodels.api as sm
from statsmodels.othermod import betareg
import scipy.stats as stats


# from statannotations.Annotator import Annotator


class PlottingFunctions:
    @staticmethod
    def plot_control_attributes(df: pd.DataFrame, attributes: list, dpf: list, lifestage: str):
        df = df[df['Treatment'] == 'Control']
        attributes.append('DateTime')  # Add datetime for the next step, then remove it again
        df = df[attributes]
        attributes.pop()
        df = df.melt(id_vars=['DateTime'], var_name='Endpoint', value_name='Value')
        df = df[df['Value'].notna()]
        basedate = pd.Timestamp('2020-04-01')
        df['dpf'] = (df['DateTime'] - basedate).dt.days
        df = df[df['dpf'].isin(dpf)]

        x = 'dpf'
        y = 'Value'

        if lifestage == 'Eggs':
            figsize = (8, 12)
        elif lifestage == 'Larvae':
            figsize = (8, 14)
        else:
                raise NotImplementedError('lifestage should be "Eggs" or "Larvae.')
        fontsize = 14
        plt.rcParams.update({'text.usetex': True, 'font.size': fontsize})

        f, axs = plt.subplots(figsize=figsize, nrows=len(attributes), ncols=1, sharex='col')
        table_stats = []
        for i, attribute in enumerate(attributes):
            curr_df = df[df['Endpoint'] == attribute]
            sns.lineplot(ax=axs[i], data=curr_df, x=x, y=y,
                         markers='x', errorbar='ci', err_style='bars', err_kws={'capsize': 5},
                         color='black')

            # GLM regression
            if attribute in ['Yolk fraction (area)']:
                glm_formula = 'Value ~ dpf'
            elif attribute in ['Eye area[mm2]'] and lifestage == 'Larvae':
                glm_formula = 'Value ~ dpf'
            else:
                glm_formula = 'Value ~ dpf + I(dpf**2)'

            if attribute == 'Yolk fraction (area)':
                binomial_model = sm.formula.glm(data=curr_df, formula=glm_formula, family=sm.families.Binomial())
                r = binomial_model.fit()
            else:
                gamma_model = sm.formula.glm(data=curr_df, formula=glm_formula,
                                             family=sm.families.Gamma(sm.genmod.families.links.Log()))
                r = gamma_model.fit()

            table_stats.append([attribute, r])
            pred = r.get_prediction()

            ci_lower = pred.summary_frame()['mean_ci_lower']
            ci_upper = pred.summary_frame()['mean_ci_upper']
            sns.lineplot(ax=axs[i], data=curr_df, x=x, y=r.fittedvalues, color='red')

            axs[i].fill_between(data=curr_df, x=x, y1=ci_lower, y2=ci_upper, alpha=0.3, color='red')

            axs[i].margins(x=0.02, tight=True)
            axs[i].set_ylabel(attribute)

        axs[0].xaxis.set_major_locator(FixedLocator(dpf))
        axs[0].set_xticklabels([str(d) for d in dpf], minor=False)
        axs[0].xaxis.set_minor_locator(MultipleLocator(2))
        # axs[0].set_title(r'{} --- Control Group'.format(lifestage))
        axs[-1].set_xlabel(r'Days post fertilization')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.03, right=0.99, top=0.99)
        f.align_ylabels()
        # plt.savefig('/media/dave/Seagate Hub/PhD Work/Writing/dca-paper/control-plots/larvae_{}.png'.format(lifestage, attribute))
        plt.show()
        return table_stats

    @staticmethod
    def lineplot_by_treatment(df: pd.DataFrame, attributes: list, dpf: list, lifestage: 'str') -> None:
        sns.color_palette('colorblind')
        attributes.append('DateTime')
        attributes.append('Treatment')
        df = df[attributes]
        attributes.pop()
        attributes.pop()
        df = df.melt(id_vars=['DateTime', 'Treatment'], var_name='Endpoint', value_name='Value')
        df = df[df['Value'].notna()]
        basedate = pd.Timestamp('2020-04-01')
        df['dpf'] = (df['DateTime'] - basedate).dt.days
        df = df[df['dpf'].isin(dpf)]

        treatments = df['Treatment'].unique()

        figsize = (11, 12)
        fontsize = 14
        plt.rcParams.update({'text.usetex': True, 'font.size': fontsize})
        x = 'dpf'
        y = 'Value'
        rows = 3
        cols = 2

        fig = plt.figure(figsize=figsize)
        # Assign None here, but we'll shortly replace them with Axes
        if lifestage == 'Eggs':
            axs = [None, None, None, None, None]
            idx = [1, 3, 5, 2, 4]
            ylims = [(0.028, 0.0485), (0.31, 0.72), (1.315, 1.356), (0.91, 1.18), (0.48, 0.87)]
            yticks = [[0.03, 0.035, 0.04, 0.045], [0.4, 0.5, 0.6, 0.7], [1.32, 1.33, 1.34, 1.35], [0.95, 1.0, 1.05, 1.1, 1.15],
                      [0.5, 0.6, 0.7, 0.8]]
        else:
            axs = [None, None, None, None, None, None]
            idx = [1, 3, 5, 2, 4, 6]
            ylims = [(1.1, 1.53), (0.95, 1.5), (3.7, 4.87), (0.053, 0.076), (0.0, 0.42), (0.0, 0.38)]
            yticks = [[1.2, 1.3, 1.4, 1.5], [1.0, 1.1, 1.2, 1.3, 1.4], [3.8, 4.0, 4.2, 4.4, 4.6, 4.8],
                      [0.055, 0.06, 0.065, 0.07, 0.075], [0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]]

        # This will break if we change the rows/cols!
        # We're doing it like this instead of f, axs = plt.subplots() because I can't figure out how to have an xlabel
        # at the bottom otherwise if the bottom plot in the second column is removed

        for i, attribute in enumerate(attributes):
            # plt.subplot Index starts at 1 in the upper left corner and increases to the right.
            # This is different to normal plt.subplots behaviour where axs[rows][cols] starts at 0 and increases down
            axs[i] = plt.subplot(rows, cols, idx[i])
            attr_df = df[df['Endpoint'] == attribute]
            sns.lineplot(ax=axs[i], data=attr_df, x=x, y=y, style='Treatment', hue='Treatment', markers=True,
                         legend=False)
            axs[i].yaxis.set_major_locator(FixedLocator(yticks[i]))
            axs[i].set_ylim(ylims[i])
            axs[i].margins(x=0.02, tight=True)
            axs[i].set_ylabel(attribute)

        if lifestage == 'Eggs':
            bottom_idx = [2, 4]
        elif lifestage == 'Larvae':
            bottom_idx = [2, 5]

        for i in bottom_idx:
            axs[i].xaxis.set_major_locator(FixedLocator(dpf))
            axs[i].set_xticklabels([str(d) for d in dpf], minor=False)
            axs[i].xaxis.set_minor_locator(MultipleLocator(2))
            axs[i].set_xlabel(r'Days post fertilization')

        # fig.suptitle(r'{}: Measured results --- All treatment groups'.format(lifestage))
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, right=0.99, top=0.99)
        fig.align_ylabels()
        if lifestage == 'Eggs':
            plt.figlegend(labels=[t for t in treatments], handles=[l for l in axs[0].lines], title='Treatment',
                          loc='lower right', bbox_to_anchor=[0.8, 0.09], borderaxespad=0, frameon=False)
        else:
            plt.figlegend(labels=[t for t in treatments], handles=[l for l in axs[0].lines], title='Treatment',
                          loc='lower right', bbox_to_anchor=[0.95, 0.18], borderaxespad=0, frameon=False)
        plt.show()

    @staticmethod
    def calculate_anova_control(df, attributes, dpf):
        df = df[df['Treatment'] == 'Control']
        attributes.append('DateTime')  # Add datetime for the next step, then remove it again
        df = df[attributes]
        attributes.pop()
        df = df.melt(id_vars=['DateTime'], var_name='Endpoint', value_name='Value')
        df = df[df['Value'].notna()]
        basedate = pd.Timestamp('2020-04-01')
        df['dpf'] = (df['DateTime'] - basedate).dt.days
        df = df[df['dpf'].isin(dpf)]

        """
        The function below was created specifically for the one-way ANOVA table results returned for Type II sum of squares
        https://www.pythonfordatascience.org/anova-python/
        """

        def anova_table(aov):
            aov['mean_sq'] = aov[:]['sum_sq'] / aov[:]['df']
            aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
            aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / (
                    sum(aov['sum_sq']) + aov['mean_sq'][-1])
            cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
            aov = aov[cols]
            return aov

        for i, attribute in enumerate(attributes):
            curr_df = df[df['Endpoint'] == attribute]
            model = sm.formula.ols(data=curr_df, formula='Value ~ C(dpf)').fit()
            aov_table = sm.stats.anova_lm(model, typ=2)
            anova_interpretation = anova_table(aov_table)
            W, p_shapiro = stats.shapiro(model.resid)
            print(attribute)
            # print(aov_table)
            # print('\n')
            # print(anova_interpretation)
            # print('\n')
            # print('{}\n'.format(attribute))
            # print('One-way ANOVA\nF: {}, p: {}, omega_sq: {}'.format(anova_interpretation['F'][-1], anova_interpretation['PR(>F)'][-1], anova_interpretation['omega_sq'][-1]))
            print('Shapiro-Wilk\nW: {}, p: {}'.format(W, p_shapiro))

    @staticmethod
    def compare_treatment_group_models(df: pd.DataFrame, attributes: list, dpf: list, lifestage: str):
        palette = sns.color_palette('colorblind')
        attributes.append('DateTime')
        attributes.append('Treatment')
        df = df[attributes]
        attributes.pop()
        attributes.pop()
        df = df.melt(id_vars=['DateTime', 'Treatment'], var_name='Endpoint', value_name='Value')
        df = df[df['Value'].notna()]
        basedate = pd.Timestamp('2020-04-01')
        df['dpf'] = (df['DateTime'] - basedate).dt.days
        df = df[df['dpf'].isin(dpf)]

        treatments = df['Treatment'].unique()
        df = df.rename(columns={'Treatment': 'treat'})

        figsize = (11, 12)
        fontsize = 14
        plt.rcParams.update({'text.usetex': True, 'font.size': fontsize})
        x = 'dpf'
        y = 'Value'
        rows = 3
        cols = 2

        fig = plt.figure(figsize=figsize)
        # Assign None to axes here but we'll shortly replace them with Axes
        if lifestage == 'Eggs':
            axs = [None, None, None, None, None]
            idx = [1, 3, 5, 2, 4]
            ylims = [(0.028, 0.0485), (0.31, 0.72), (1.315, 1.356), (0.91, 1.18), (0.48, 0.87)]
            yticks = [[0.03, 0.035, 0.04, 0.045], [0.4, 0.5, 0.6, 0.7], [1.32, 1.33, 1.34, 1.35], [0.95, 1.0, 1.05, 1.1, 1.15],
                      [0.5, 0.6, 0.7, 0.8]]
        elif lifestage == 'Larvae':
            axs = [None, None, None, None, None, None]
            idx = [1, 3, 5, 2, 4, 6]
            # Hand tuned plot limits and axis ticks to maintain consistency with other line plots
            ylims = [(1.1, 1.53), (0.95, 1.5), (3.7, 4.87), (0.053, 0.076), (0.0, 0.42), (0.0, 0.38)]
            yticks = [[1.2, 1.3, 1.4, 1.5], [1.0, 1.1, 1.2, 1.3, 1.4], [3.8, 4.0, 4.2, 4.4, 4.6, 4.8],
                      [0.055, 0.06, 0.065, 0.07, 0.075], [0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3]]

        # This will break if we change the rows/cols!
        # We're doing it like this instead of f, axs = plt.subplots() because I can't figure out how to have an xlabel
        # at the bottom otherwise if the bottom plot in the second column is removed
        table_stats = []

        for i, attribute in enumerate(attributes):
            # plt.subplot Index starts at 1 in the upper left corner and increases to the right.
            # This is different to normal plt.subplots behaviour where axs[rows][cols] starts at 0 and increases down
            axs[i] = plt.subplot(rows, cols, idx[i])
            attr_df = df[df['Endpoint'] == attribute]
            axs[i].yaxis.set_major_locator(FixedLocator(yticks[i]))
            axs[i].set_ylim(ylims[i])
            axs[i].margins(x=0.02, tight=True)
            axs[i].set_ylabel(attribute)

            if attribute == 'Yolk fraction (area)':
                glm_formula = 'Value ~ dpf * C(treat, Treatment(reference="Control"))'
            elif attribute in ['Standard length[mm]', 'Eye area[mm2]'] and lifestage == 'Larvae':
                glm_formula = 'Value ~ dpf * C(treat, Treatment(reference="Control"))'
            else:
                glm_formula = 'Value ~ dpf * C(treat, Treatment(reference="Control")) + I(dpf ** 2) * C(treat, Treatment(reference="Control"))'

            if attribute == 'Yolk fraction (area)':
                binomial_model = sm.formula.glm(formula=glm_formula, data=attr_df, family=sm.families.Binomial())
                r = binomial_model.fit()
            else:
                gamma_model = sm.formula.glm(data=attr_df, formula=glm_formula,
                                             family=sm.families.Gamma(sm.genmod.families.links.Log()))
                r = gamma_model.fit()
            pred = r.get_prediction()

            table_stats.append([attribute, r])
            # print('{}\n'.format(attribute))
            # print(r.summary())
            sns.lineplot(ax=axs[i], data=attr_df, x=x, y=r.fittedvalues, hue='treat', legend=False)

            for count, treatment in enumerate(treatments):
                treatment_df = attr_df.loc[attr_df['treat'] == treatment]
                indices = treatment_df.index
                ci_lower = pred.summary_frame()['mean_ci_lower'][indices]
                ci_upper = pred.summary_frame()['mean_ci_upper'][indices]
                axs[i].fill_between(data=treatment_df, x=x, y1=ci_lower, y2=ci_upper, alpha=0.3, color=palette[count])


        if lifestage == 'Eggs':
            bottom_idx = [2, 4]
        elif lifestage == 'Larvae':
            bottom_idx = [2, 5]

        for i in bottom_idx:
            axs[i].xaxis.set_major_locator(FixedLocator(dpf))
            axs[i].set_xticklabels([str(d) for d in dpf], minor=False)
            axs[i].xaxis.set_minor_locator(MultipleLocator(2))
            axs[i].set_xlabel(r'Days post fertilization')

        # fig.suptitle(r'{}: GLM regression --- All treatment groups'.format(lifestage))
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, right=0.99, top=0.99)
        fig.align_ylabels()
        if lifestage == 'Eggs':
            plt.figlegend(labels=[t for t in treatments], handles=[l for l in axs[0].lines], title='Treatment',
                          loc='lower right', bbox_to_anchor=[0.8, 0.09], borderaxespad=0, frameon=False)
        else:
            plt.figlegend(labels=[t for t in treatments], handles=[l for l in axs[0].lines], title='Treatment',
                          loc='lower right', bbox_to_anchor=[0.95, 0.18], borderaxespad=0, frameon=False)

        plt.show()
        return table_stats

    @staticmethod
    def prepare_latex_table(table_stats: list, control: bool):
        header = '\\begin{table}\n' \
                 '\\centering\n' \
                 '{\\tiny\n' \
                 '\\begin{tabular}{lccccccc}\n' \
                 '\\toprule\n' \
                 'Endpoint & Term & Estimate & Standard error & df & $t$ or $z$ & $p$ & \\\\\n' \
                 '\\midrule\n'
        footer = '\\bottomrule\n' \
                 '\\end{tabular}\n' \
                 '}\n' \
                 '\\caption{}\n' \
                 '\\label{tab:}\n' \
                 '\\end{table}'
        content = ''

        def p_marker(p: float):
            if p < 0.001:
                return '\\textasteriskcentered\\textasteriskcentered\\textasteriskcentered'
            elif p < 0.01:
                return '\\textasteriskcentered\\textasteriskcentered'
            elif p < 0.05:
                return '\\textasteriskcentered'
            else:
                return 'ns'

        def short_name(n: str):
            parts = n.split('C(treat, Treatment(reference="Control"))[T.')
            new_parts = []
            for part in parts:
                if len(part) > 0:
                    if part == 'I(dpf ** 2):':
                        new_parts.append('dpf$^2$:')
                    elif part == 'I(dpf ** 2)':
                        new_parts.append('dpf$^2$')
                    elif part[-1] == ']':
                        new_parts.append(part[:-1])
                    else:
                        new_parts.append(part)
            new_str = ''
            for np in new_parts:
                new_str += np
            return new_str

        def sort_order(item):
            item = item[1][0]
            if item == '8 $\\mu$g/L':
                return 0
            elif item == '27 $\\mu$g/L':
                return 1
            elif item == '108 $\\mu$g/L':
                return 2
            elif item == '220 $\\mu$g/L':
                return 3
            elif item == '343 $\\mu$g/L':
                return 4
            elif item == '747 $\\mu$g/L':
                return 5
            elif item == 'dpf:8 $\\mu$g/L':
                return 6
            elif item == 'dpf:27 $\\mu$g/L':
                return 7
            elif item == 'dpf:108 $\\mu$g/L':
                return 8
            elif item == 'dpf:220 $\\mu$g/L':
                return 9
            elif item == 'dpf:343 $\\mu$g/L':
                return 10
            elif item == 'dpf:747 $\\mu$g/L':
                return 11
            elif item == 'dpf$^2$:8 $\\mu$g/L':
                return 12
            elif item == 'dpf$^2$:27 $\\mu$g/L':
                return 13
            elif item == 'dpf$^2$:108 $\\mu$g/L':
                return 14
            elif item == 'dpf$^2$:220 $\\mu$g/L':
                return 15
            elif item == 'dpf$^2$:343 $\\mu$g/L':
                return 16
            elif item == 'dpf$^2$:747 $\\mu$g/L':
                return 17
            else:
                return 99

        def latex_e(n):
            '''Convert scientific notation to format suitable for L&O:M'''
            e_n = '{: .3e}'.format(n)
            m, e = e_n.split('e')
            if int(e) == 0 or int(e) == 1:
                return m
            else:
                x_e = ' $\\times 10^{{{}}}$'.format(int(e))

            return m + x_e

        for [attribute, r] in table_stats:
            names = r.model.exog_names
            data = {}
            for n in names:
                if control:
                    if n in ['Intercept']:
                        continue
                else:
                    if n in ['Intercept', 'dpf', 'I(dpf ** 2)']:
                        continue
                data[n] = [short_name(n), r.params[n], r.bse[n], r.tvalues[n], r.pvalues[n], p_marker(r.pvalues[n])]

            data = dict(sorted(data.items(), key=sort_order))
            for i, n in enumerate(data):
                if i == 0:
                    content += '{} & {} & {} & {} & {} & {: .2f} & {} & {} \\\\\n'.format(attribute, data[n][0], latex_e(data[n][1]), latex_e(data[n][2]),  r.df_resid, data[n][3], latex_e(data[n][4]), data[n][5])
                else:
                    content += '& {} & {} & {} & & {: .2f} & {} & {} \\\\\n'.format(data[n][0], latex_e(data[n][1]), latex_e(data[n][2]), data[n][3], latex_e(data[n][4]), data[n][5])

        return '{}{}{}'.format(header, content, footer)

    @staticmethod
    def calculate_precision(df: pd.DataFrame):
        df = df[df['Treatment'] == 'Control']
        basedate = pd.Timestamp('2020-04-01')
        df['dpf'] = (df['DateTime'] - basedate).dt.days

        df['Egg mean diameter[mm]'].describe()
        df['Egg area[mm2]'].describe()

        eye_area = []
        eye_diameter = []
        for dpf in range(7, 12):
            curr_df = df[df['dpf'] == dpf]
            eye_area.append(curr_df['Eye area[mm2]'].mean())
            eye_diameter.append(curr_df['Eye mean diameter[mm]'].mean())

        foo = -1
