import pandas as pd
import numpy as np
import statsmodels.api as sm


# Grabbing appropriate VBTs, reading them in, and cleaning them
VBT_2015_file = '2015_VBT.xlsx'
VBT_2008_file = '2008_VBT.xls'
VBT_2001_male_file = '2001_VBT_Male.xls'
VBT_2001_female_file = '2001_VBT_Female.xls'

xl_2015 = pd.ExcelFile(VBT_2015_file)
xl_2008 = pd.ExcelFile(VBT_2008_file)
xl_2001_male = pd.ExcelFile(VBT_2001_male_file)
xl_2001_female = pd.ExcelFile(VBT_2001_female_file)

male_smoker_15 = pd.read_excel(xl_2015, sheet_name='2015 MSM ANB', skiprows=2)
male_nonsmoker_15 = pd.read_excel(xl_2015, sheet_name='2015 MNS ANB', skiprows=2)
female_nonsmoker_15 = pd.read_excel(xl_2015, sheet_name='2015 FNS ANB', skiprows=2)
female_smoker_15 = pd.read_excel(xl_2015, sheet_name='2015 FSM ANB', skiprows=2)

columns = male_nonsmoker_15.columns

male_smoker_08 = pd.read_excel(xl_2008, sheet_name='Male NS ANB', skiprows=5, header = None, names = columns)
male_nonsmoker_08 = pd.read_excel(xl_2008, sheet_name='Male SM ANB', skiprows=5, header = None, names = columns)
female_smoker_08 = pd.read_excel(xl_2008, sheet_name='Female NS ANB', skiprows=5, header = None, names = columns)
female_nonsmoker_08 = pd.read_excel(xl_2008, sheet_name='Female SM ANB', skiprows=5, header = None, names = columns)

male_smoker_01 = pd.read_excel(xl_2001_male, sheet_name='MaleNS2001VBT', skiprows=8, header = None, names = columns)
male_nonsmoker_01 = pd.read_excel(xl_2001_male, sheet_name='MaleSM2001VBT', skiprows=8, header = None, names = columns)
female_smoker_01 = pd.read_excel(xl_2001_female, sheet_name='FemaleNS2001VBT', skiprows=8, header = None, names = columns)
female_nonsmoker_01 = pd.read_excel(xl_2001_female, sheet_name='FemaleSM2001VBT', skiprows=8, header = None, names = columns)

# Initializing a mortality table dictionary
tables = {
    'male_smoker': [male_smoker_01, male_smoker_08, male_smoker_15],
    'male_nonsmoker': [male_nonsmoker_01, male_nonsmoker_08, male_nonsmoker_15],
    'female_smoker': [female_smoker_01, female_smoker_08, female_smoker_15],
    'female_nonsmoker': [female_nonsmoker_01, female_nonsmoker_08, female_nonsmoker_15]
}


def generate_matrix(tables, duration):
    ''' Creating the matrix that we will perform Singular Value Decomposition on. Making a matrix all select and ult. durations.'''
    matrix = pd.DataFrame()
    ages = list(range(18, 91))

    for i, df in enumerate(tables):
        duration_series = df[df['Iss. Age'].isin(ages)][duration].reset_index(drop = True)

        # Additional data cleaning
        if i == 1 or i == 2:
            duration_series = duration_series / 1000

        # Taking the log of all mortality rates as dictated by the Lee-Carter method
        matrix[str(i)] = np.log(duration_series)

    return matrix

def svd(matrix):
    '''Singular value decomposition function. Returns average log mortality rate, sensitivity of log mortality, and k_t which is how mortality changes over time'''

    # Prepping matrix for svd function
    matrix['Average Mortality'] = (matrix['0'] + matrix['1'] + matrix['2']) / 3
    matrix['2001'] = matrix['0'] - matrix['Average Mortality']
    matrix['2008'] = matrix['1'] - matrix['Average Mortality']
    matrix['2015'] = matrix['2'] - matrix['Average Mortality']
    svd_matrix = matrix[['2001', '2008', '2015']]


    U, S, V_t = np.linalg.svd(svd_matrix)

    b = U[:, 0]
    k = S[0] * V_t[0, :]

    # Year 0 is the base year (2001), year 7 is 7 years from base year (2008), and year 14 is 14 years from base year (2015)
    model = sm.OLS(k, sm.add_constant([0, 7, 14]))
    results = model.fit()

    alpha, beta = results.params

    future_years = np.arange(0, 100)
    k_t = alpha + beta * future_years

    # Returning the three parameters necessary for Lee-Carter
    return matrix['Average Mortality'], b, k_t

def final_tables(matrix_dict):
    '''Generates a final mortality table with inputs a, b, and k_t'''
    mortality_df = pd.DataFrame()
    mortality_df['Age'] = np.arange(18, 91)

    for k, v in matrix_dict.items():
        yearly_mort = []
        for j in range(len(v[1])):

            #Generating different mortality depending on select vs ultimate rates
            if k != 'Ult.':
                qx = np.exp(v[0].iloc[j] + v[1][j] * v[2][int(k)])
            else:
                qx = np.exp(v[0].iloc[j] + v[1][j] * v[2][25])

            yearly_mort.append(qx)

        mortality_df[k] = yearly_mort

    return mortality_df

def main():
    final_dict = {}
    # Iterating through each table type
    for k, v in tables.items():
        matrix_dict = {}

        # Iterating through every duration in every table type
        for i in v[0].columns[1:-1]:
            matrix = generate_matrix(v, duration = i)
            a, b, k_t = svd(matrix)
            matrix_dict[i] = [a, b, k_t]

        final_dict[k] = final_tables(matrix_dict)

    with pd.ExcelWriter('Final_Mortality_Projections.xlsx', engine = 'openpyxl') as writer:
        for k, v in final_dict.items():
            v.to_excel(writer, sheet_name = k, index = False)

if __name__ == '__main__':
    main()




