import pandas as pd
import statsmodels.api as sm


def calculate_ecx(df: pd.DataFrame, attributes: list, dpf: list, lifestage: str, x: int):
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

    df.loc[df['treat'] == 'Control', 'treat'] = 0
    df.loc[df['treat'] == '8 $\\mu$g/L', 'treat'] = 8
    df.loc[df['treat'] == '27 $\\mu$g/L', 'treat'] = 27
    df.loc[df['treat'] == '108 $\\mu$g/L', 'treat'] = 108
    df.loc[df['treat'] == '220 $\\mu$g/L', 'treat'] = 220
    df.loc[df['treat'] == '343 $\\mu$g/L', 'treat'] = 343
    df.loc[df['treat'] == '747 $\\mu$g/L', 'treat'] = 747

    for i, attribute in enumerate(attributes):
        attr_df = df[df['Endpoint'] == attribute]

        if attribute == 'Yolk fraction (area)':
            glm_formula = 'Value ~ dpf * treat'
        elif attribute in ['Standard length[mm]', 'Eye area[mm2]'] and lifestage == 'Larvae':
            glm_formula = 'Value ~ dpf * treat'
        else:
            glm_formula = 'Value ~ dpf * treat + I(dpf ** 2) * treat'

        if attribute == 'Yolk fraction (area)':
            binomial_model = sm.formula.glm(formula=glm_formula, data=attr_df, family=sm.families.Binomial())
            r = binomial_model.fit()
        else:
            gamma_model = sm.formula.glm(data=attr_df, formula=glm_formula,
                                         family=sm.families.Gamma(sm.genmod.families.links.Log()))
            r = gamma_model.fit()
        pred = r.get_prediction()

        foo = -1