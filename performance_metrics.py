import numpy as np
import numpy_financial as npf
import pandas as pd


base_file = 'Yearly_Cashflows.xlsx'
cf_excel = pd.ExcelFile(base_file)
cf_df = pd.read_excel(cf_excel)

def profit_metrics():
    cfs = cf_df['Premium Payment'] - cf_df['Benefit Payment']
    rates = cf_df['Discount Rates']

    irr = npf.irr(cfs)

    npv = sum(cf / ((1 + 0.05) ** i) for i, cf in enumerate(cfs))

    cumulative_sum = np.cumsum(cfs)
    breakeven = next((i for i, cf in enumerate(cumulative_sum) if cf > 0), None)

    return {
        'IRR': irr,
        'NPV': npv,
        'Breakeven Year': breakeven
    }


print(profit_metrics())