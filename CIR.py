import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fractions import Fraction
import statsmodels.api as sm




def CIR(data, term):
 '''Complete CIR Function - https://www.soa.org/48e9a7/globalassets/assets/files/resources/research-report/2023/interest-rate-model-calibration-study.pdf'''

    def clean(df):
        '''Cleaning the initial data'''

        df = df.iloc[::-1].reset_index(drop = True)

        # Only pulling securities with a 1 year maturity
        new_df = df[(df['Maturity'] == '1 Year') | (df['Maturity'] == '1 year')]
        rates = new_df['Rate Indicated for Current Month Year']

        cleaned_rates = []
        for i in rates:
            if len(i) > 5:
                split = i.split('-')
                cleaned_rate = float(Fraction(split[0]) / 100) + float(Fraction(split[1][:-1]) / 100)
            else:
                cleaned_rate = float(Fraction(i[:-1]) / 100)

            cleaned_rates.append(cleaned_rate)

        return cleaned_rates


    def parameter_est(rates):
        '''Takes in historical interest rates and estimates the three main parameters for the CIR function.'''

        diff_ray = np.diff(rates)

        r_t = rates[:-1]
        new_ray = sm.add_constant(r_t)

        # Performing a linear regression on historical interest rates and historical interest rate changes
        model = sm.OLS(diff_ray, new_ray)
        results = model.fit()

        a, b = results.params

        gamma = -b / (1 / 12)
        rbar = a / b

        residuals = results.resid
        alpha = np.mean((residuals ** 2) / r_t)

        # If rbar is negative then it will be reset to the mean
        if rbar < 0:
            rbar = np.mean(rates)

        return gamma, rbar, alpha

    def find_CIR(rates, gamma, rbar, alpha):
        # rt + delta - rt = gamma * (rbar - rt)*delta + sqrt(rt) * et+delta

        # Initializing vars for CIR model
        delta = 1 / 12
        r0 = rates[-1]
        alpha1 = gamma * rbar * delta
        beta1 = 1 - gamma * delta
        sigma = np.sqrt(alpha * delta)
        new_rates = [r0]

        for i in range(1, 120 * 12):
            e = np.random.normal(0, 1)
            rate = alpha1 + beta1 * new_rates[i - 1] + np.sqrt(new_rates[i - 1]) * e * sigma

            # Flooring future interest rates at 0
            new_rates.append(max(rate, 0))

        return new_rates

    def adjust_rates(rates):
        # Setting interest rates to be expressed annually
        annual_rates = []
        monthly_rates = [((1 + i) ** (1 / 12) - 1) for i in rates]

        for i in range(0, len(monthly_rates), 12):
            total_rates = monthly_rates[i:i + 12]
            annual_rate = np.prod([1 + rate for rate in total_rates]) - 1
            annual_rates.append(annual_rate)


        return annual_rates




    ir_df = pd.read_csv(data)
    rates = clean(ir_df)
    gamma, rbar, alpha = parameter_est(rates)

    # Simulating 1000 interest rate paths and averaging them out
    simulations = 1000
    rates_list = []
    for i in range(simulations):
        rates = find_CIR(rates, gamma, rbar, alpha)
        final_rates = adjust_rates(rates)
        rates_list.append(final_rates)

    avg_rates = [sum(i) / len(i) for i in zip(*rates_list)]

    return avg_rates
