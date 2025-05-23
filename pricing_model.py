import pandas as pd
import numpy as np
from scipy.optimize import brentq
from tqdm import tqdm
from CIR import CIR
import matplotlib.pyplot as plt


# FILES
dataset = pd.read_csv('Insurance_Dataset.csv')
mortality_file = 'Final_Mortality_Projections.xlsx'
ir_data = 'TCIR_Monthly_SpecificMaturities_20150502_20250501.csv'
mortality_excel = pd.ExcelFile(mortality_file)


# Mortality Tables
male_smoker = pd.read_excel(mortality_excel, sheet_name='male_smoker')
male_nonsmoker = pd.read_excel(mortality_excel, sheet_name='male_nonsmoker')
female_smoker = pd.read_excel(mortality_excel, sheet_name='female_smoker')
female_nonsmoker = pd.read_excel(mortality_excel, sheet_name='female_nonsmoker')

tables = {
    ('male', True): male_smoker,
    ('male', False): male_nonsmoker,
    ('female', True): female_smoker,
    ('female', False): female_nonsmoker
}

# Expense Load
exp_load = 0.1

# Profit Margin
margin = 0.07




# Interest Rates

# Pull monthly rates from CIR Model
rates = CIR(ir_data, term = 'Permanent')

# Including a year list for PowerBI visualization
years = [(2015 + i) for i in range(len(rates))]


rates_df = pd.DataFrame()
rates_df['Rates'] = rates
rates_df['Year'] = years

# Exporting rates to an Excel File
with pd.ExcelWriter('Final_Annual_Rates.xlsx', engine = 'openpyxl') as writer:
    rates_df.to_excel(writer, sheet_name='Rates', index=False)


# Lapse Rates
lapse_dict = {'1': 0.07, '2': 0.0575, '3': 0.06, '4': 0.05, '5': 0.0525, '6': 0.0425}


# Creating a class for each policyholder in the dataset
class Row:
    def __init__(self, sex, age, smoker, term, F, payback_type):
        self.sex = sex
        self.age = age
        self.smoker = True if smoker == 'yes' else False
        self.term = term
        self.F = F
        self.payback_type = payback_type

    def get_lapse_rate(self, duration):
        if self.age > 72:
            lapse_rate = 0
        elif str(duration) in lapse_dict.keys():
            lapse_rate = lapse_dict[str(duration)]
        elif 6 < duration <= 15:
            lapse_rate = 0.02
        else:
            lapse_rate = 0.005

        return lapse_rate


    def calculate_mortality(self):
        ''' Generates a mortality list for an individual, where each item in the list is a persons chance of dying in that year.'''

        # Pulling the correct mortality table given a policyholders demographics
        table = tables[(self.sex, self.smoker)]

        mortality_list = []

        # The mortality list will stop when a policyholder attains an age of 119 years
        for i, j in enumerate(range(self.age, 120)):

            # Differentiating between Select and Ultimate mortalities. Ultimate mortalities are used after a duration of 25 years.
            if int(i) < 24:

                # Mortality table goes to age of 90 years. Once a policyholder is older than 90 their mortality is graded up 10% every year.
                if j <= 90:
                    mortality = table[table['Age'] == j][i + 1].values[0]
                else:
                    mortality = min(table[table['Age'] == 90][i + 1].values[0] * (1.1 ** (j - 90)), 1)
            else:
                if j <= 90:
                    mortality = table[table['Age'] == j]['Ult.'].values[0]
                else:
                    mortality = min(table[table['Age'] == 90]['Ult.'].values[0] * (1.1 ** (j - 90)), 1)

            mortality_list.append(mortality)

        return mortality_list



    def calculate_epv(self, mortality):
        '''Calculates the expected present value of benefit payments under the assumption that benefit is paid at the end of the policy year'''

        # Setting initial vars
        discount = 1
        EPV = 0
        immortality = [(1 - i) for i in mortality]

        for i in range(len(mortality)):
            discount *= 1 / (1 + rates[i])
            lapse_rate = self.get_lapse_rate(i)

            # Calculating the probability that a policyholder will die in a year given that they have not died previously
            if i == 0:
                death_prob = 1 * mortality[i]
            else:
                death_prob = np.prod(immortality[:i]) * mortality[i]

            EPV += discount * death_prob * self.F * (1 - lapse_rate)

        return EPV

    def calculate_pv_premium(self, premium, mortality):
        '''Takes in a level premium, and calculates the present value of that level premium, taking into account discount, lapse, and mortality'''

        # Setting initial vars
        PV = 0
        survival = 1
        discount = 1

        for i in range(120 - self.age):
            if 'Forever' not in self.payback_type and i >= int(self.payback_type):
                break
            PV += premium * survival * discount
            lapse = self.get_lapse_rate(i)
            survival *= (1 - lapse) * (1 - mortality[i])
            discount *= 1 / (1 + rates[i])

        return PV

    def premium_solve(self, mortality):
        '''Implements a premium solve using brents method to find roots.'''
        EPV = self.calculate_epv(mortality)

        def root_solve(gross_premium):
            '''Defining function that we will root solve for'''
            pv_premium = self.calculate_pv_premium(gross_premium, mortality)
            expenses = pv_premium * exp_load
            profit = EPV * margin
            return pv_premium - (EPV + expenses + profit)

        return brentq(root_solve, 0.01, 1200000)



    def calculate_cfs(self, mortality, calculated_premium):
        '''Takes in calculated premium and mortality to generate policy level cfs.'''

        # Setting initial vars
        year_stats = []
        discount = 1
        survival = 1
        immortality = [(1 - i) for i in mortality]
        pv_benefit = 0
        pv_premium = 0
        lapse = 1

        for i in range(120 - self.age):
            if 'Forever' not in self.payback_type:
                if i < int(self.payback_type):
                    premium_payment = survival * calculated_premium * lapse

                else:
                    premium_payment = 0
            else:
                premium_payment = survival * calculated_premium * lapse

            # Calculating PV of premium and benefit for PowerBI and premium solve function
            pv_premium += premium_payment * discount

            # Applying EOY adjustments to lapse and discount for yearly exp. benefit calculations
            lapse *= (1 - self.get_lapse_rate(i))
            discount *= 1 / (1 + rates[i])
            survival *= immortality[i]

            if i == 0:
                death_prob = 1 * mortality[i]
            else:
                death_prob = np.prod(immortality[:i]) * mortality[i]

            benefit =  death_prob * self.F * lapse
            pv_benefit += benefit * discount

            year_stats.append({'Year': (2015 + i),
                               'Survival Probability': survival,
                               'Discount Rates': discount,
                               'Benefit Payment': benefit,
                               'Premium Payment': premium_payment,
                               'Discounted Premium': pv_premium,
                               'Discounted Benefit': pv_benefit
                               })

        return year_stats, pv_premium


def main():
    np.random.seed(10)
    premiums = []
    EPVs = []
    final_cfs = []

    for row in tqdm(dataset.itertuples(index = True), total = len(dataset)):
        X = Row(sex = row[2], age = row[1], smoker = row[3], term = 'Permanent', F = row[5], payback_type = row[4])

        mortality = X.calculate_mortality()
        EPV = X.calculate_epv(mortality)
        EPVs.append(EPV)
        gross_premium = X.premium_solve(mortality)
        premiums.append(gross_premium)

        cfs, pv_premium = X.calculate_cfs(mortality, gross_premium)
        for cf in cfs:
            cf['Age'] = X.age
            cf['Sex'] = X.sex
            cf['Smoker'] = X.smoker
            cf['Face Value'] = X.F
            cf['Payback Type'] = X.payback_type
            final_cfs.append(cf)

    dataset['Premium'] = premiums
    dataset['EPV'] = EPVs

    dataset.to_excel('Final_Premiums.xlsx')

    cf_df = pd.DataFrame(final_cfs)
    cf_df.to_excel('Yearly_Cashflows.xlsx')



if __name__ == '__main__':
    main()

