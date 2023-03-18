import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FixedLocator)
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

        figsize = (8, 12)
        plt.rcParams.update({'text.usetex': True})

        f, axs = plt.subplots(figsize=figsize, nrows=len(attributes), ncols=1, sharex='col')
        for i, attribute in enumerate(attributes):
            curr_df = df[df['Endpoint'] == attribute]
            sns.lineplot(ax=axs[i], data=curr_df, x=x, y=y,
                         markers='x', errorbar='ci', err_style='bars', err_kws={'capsize': 5},
                         # markers='x', errorbar='ci', err_style='band',
                         color='black')

            # GLM regression
            if attribute == 'Yolk fraction (area)':
                # This is in the range (0, 1) so uses a Beta distribution
                # model = sm.formula.glm(data=curr_df, formula='curr_df[y] ~ curr_df[x]',
                #                              family=sm.families.Binomial(sm.genmod.families.links.Logit()))
                # r = model.fit()
                beta_model = betareg.BetaModel.from_formula(data=curr_df, formula='Value ~ dpf')
                r = beta_model.fit()
            else:
                gamma_model = sm.formula.glm(data=curr_df, formula='curr_df[y] ~ curr_df[x] + curr_df[x] ^ 2',
                               family=sm.families.Gamma(sm.genmod.families.links.Log()))
                r = gamma_model.fit()
                pred = r.get_prediction()

                ci_lower = pred.summary_frame()['mean_ci_lower']
                ci_upper = pred.summary_frame()['mean_ci_upper']
                axs[i].fill_between(data=curr_df, x=x, y1=ci_lower, y2=ci_upper, alpha=0.3, color='red')

            sns.lineplot(ax=axs[i], data=curr_df, x=x, y=r.fittedvalues, color='red')

            # Requires Latex installation
            # Good values eggs 0.75, 0.1' larvae 0.9, 0.5
            if lifestage == 'Eggs':
                left, top = 0.75, 0.1
            elif lifestage == 'Larvae':
                left, top = 0.9, 0.5
            else:
                raise NotImplementedError('lifestage should be "Eggs" or "Larvae.')
            # axs[i].text(left, top, r'\raggedright $y = {:.4f}x + {:.4f}$, \\ $R^2 = {:.4f}$'.format(r.params[1], r.params[0], r.rsquared),
            #             horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)

            axs[i].set_ylabel(attribute)

        axs[0].xaxis.set_major_locator(FixedLocator(dpf))
        axs[0].set_xticklabels([str(d) for d in dpf], minor=False)
        axs[0].xaxis.set_minor_locator(MultipleLocator(2))
        axs[0].set_title(r'{} --- Control Group'.format(lifestage))
        axs[-1].set_xlabel(r'Days post fertilization')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        # plt.savefig('/media/dave/Seagate Hub/PhD Work/Writing/dca-paper/control-plots/larvae_{}.png'.format(lifestage, attribute))
        plt.show()

    @staticmethod
    def lineplot_by_treatment(df: pd.DataFrame, attributes: list, dpf: list) -> None:
        sns.color_palette('colorblind')
        attributes.append('DateTime')
        attributes.append('Treatment')
        df = df[attributes]
        attributes.pop()
        attributes.pop()
        df = df.melt(id_vars=['DateTime', 'Treatment'], var_name='Endpoint', value_name='Value')
        df = df[df['Value'].notna()]

        plt.rcParams.update({'text.usetex': True, 'font.size': 14})

        basedate = pd.Timestamp('2020-04-01')
        df['dpf'] = (df['DateTime'] - basedate).dt.days
        df = df[df['dpf'].isin(dpf)]

        x = 'dpf'
        y = 'Value'
        treatment = df['Treatment']

        g = sns.relplot(kind='line',
                        data=df,
                        x=x, y=y,
                        hue=treatment,
                        col='Endpoint',
                        col_wrap=2,
                        col_order=attributes,
                        style=treatment,
                        markers=True,
                        height=6,
                        aspect=1,
                        facet_kws={'sharey': False, 'sharex': True})
        leg = g._legend
        leg.set_bbox_to_anchor([0.65, 0.15])
        leg._loc = 4
        leg.borderaxespad = 0

        for ax in g.axes:
            ax.xaxis.set_major_locator(FixedLocator(dpf))
            ax.xaxis.set_minor_locator(MultipleLocator(2))
        g.set_xticklabels([str(d) for d in dpf], minor=False)
        g.set_axis_labels('Days post fertilization', 'Value')
        g.tight_layout(w_pad=0)

        plt.show()

    @staticmethod
    def new_lineplot_by_treatment(df: pd.DataFrame, attributes: list, dpf: list, lifestage: 'str') -> None:
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

        figsize = (10, 12)
        plt.rcParams.update({'text.usetex': True})
        x = 'dpf'
        y = 'Value'
        rows = 3
        cols = 2

        fig = plt.figure(figsize=figsize)
        # Assign None here, but we'll shortly replace them with Axes
        axs = [None, None, None, None, None]
        idx = [1, 3, 5, 2, 4]

        # This will break if we change the rows/cols!
        # We're doing it like this instead of f, axs = plt.subplots() because I can't figure out how to have an xlabel
        # at the bottom otherwise if the bottom plot in the second column is removed

        for i, attribute in enumerate(attributes):
            # plt.subplot Index starts at 1 in the upper left corner and increases to the right.
            # This is different to normal plt.subplots behaviour where axs[rows][cols] starts at 0 and increases down
            axs[i] = plt.subplot(rows, cols, idx[i])
            attr_df = df[df['Endpoint'] == attribute]
            sns.lineplot(ax=axs[i], data=attr_df, x=x, y=y, style='Treatment', hue='Treatment', markers=True, legend=False)
            axs[i].set_ylabel(attribute)

        for i in [2, 4]:
            axs[i].xaxis.set_major_locator(FixedLocator(dpf))
            axs[i].set_xticklabels([str(d) for d in dpf], minor=False)
            axs[i].xaxis.set_minor_locator(MultipleLocator(2))
            axs[i].set_xlabel(r'Days post fertilization')

        fig.suptitle(r'{}: Measured results --- All treatment groups'.format(lifestage))
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.figlegend(labels=[t for t in treatments], handles=[l for l in axs[0].lines], title='Treatment',
                      loc='lower right', bbox_to_anchor=[0.7, 0.15], borderaxespad=0, frameon=False)
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

        figsize = (10, 12)
        plt.rcParams.update({'text.usetex': True})
        x = 'dpf'
        y = 'Value'
        rows = 3
        cols = 2

        fig = plt.figure(figsize=figsize)
        # Assign None here but we'll shortly replace them with Axes
        axs = [None, None, None, None, None]
        idx = [1, 3, 5, 2, 4]

        # This will break if we change the rows/cols!
        # We're doing it like this instead of f, axs = plt.subplots() because I can't figure out how to have an xlabel
        # at the bottom otherwise if the bottom plot in the second column is removed

        for i, attribute in enumerate(attributes):
            # plt.subplot Index starts at 1 in the upper left corner and increases to the right.
            # This is different to normal plt.subplots behaviour where axs[rows][cols] starts at 0 and increases down
            axs[i] = plt.subplot(rows, cols, idx[i])
            attr_df = df[df['Endpoint'] == attribute]
            for treatment in treatments:
                curr_df = attr_df[attr_df['Treatment'] == treatment]
                gamma_model = sm.formula.glm(data=curr_df, formula='Value ~ dpf + dpf ^ 2',
                                             family=sm.families.Gamma(sm.genmod.families.links.Log()))
                r = gamma_model.fit()
                pred = r.get_prediction()
                ci_lower = pred.summary_frame()['mean_ci_lower']
                ci_upper = pred.summary_frame()['mean_ci_upper']

                sns.lineplot(ax=axs[i], data=curr_df, x=x, y=r.fittedvalues)
                axs[i].fill_between(data=curr_df, x=x, y1=ci_lower, y2=ci_upper, alpha=0.3)

            axs[i].set_ylabel(attribute)

        for i in [2, 4]:
            axs[i].xaxis.set_major_locator(FixedLocator(dpf))
            axs[i].set_xticklabels([str(d) for d in dpf], minor=False)
            axs[i].xaxis.set_minor_locator(MultipleLocator(2))
            axs[i].set_xlabel(r'Days post fertilization')

        fig.suptitle(r'{}: GLM regression --- All treatment groups'.format(lifestage))
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.figlegend(labels=[t for t in treatments], handles=[l for l in axs[0].lines], title='Treatment',
                      loc='lower right', bbox_to_anchor=[0.7, 0.15], borderaxespad=0, frameon=False)

        plt.show()

    @staticmethod
    def calculate_ancova(df: pd.DataFrame, attributes: list, dpf: list, lifestage: str):
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

        for i, attribute in enumerate(attributes):
            curr_df = df[df['Endpoint'] == attribute]

            model = sm.formula.glm(data=curr_df, formula='Value ~ Treatment + dpf', family=sm.families.Gamma(sm.genmod.families.links.Log()))
            r = model.fit()