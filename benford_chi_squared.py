import pandas as pd
import numpy as np

# example of grouping by leading digits
data = pd.DataFrame({'item':['wax','wick','wax','spare_parts','repair','misc'],
                   'quantity':[152,300,148,15,1, 4],
                   'price':[3,0.5,3.5,1000,150,0.2],
                   'total_cost':[152*3,300*0.5,148*3.5,15*1000,1*150,4*0.2]})

def count_leading(df,target_column):
  pivot_df = (df.assign(first_digit=lambda x: np.ceil(x[target_column]
                                            ).astype(str).str[:1].astype(int))
              .groupby('first_digit').count()
              )[target_column]
  return pivot_df

pivot_df = count_leading(data, 'total_cost')

# example of chi-squared test. Are leading digits of the dataset in line with Benford's law?
pivoted_data = pd.DataFrame(
    {'first_digit' : [1, 2, 3, 4, 5, 6, 7, 8, 9],
     'observed_count' : [253, 166, 130, 88, 84, 60, 55, 70 ,94],
     'observed_mean' : [253/1000, 166/1000, 130/1000, 88/1000, 
                        84/1000, 60/1000, 55/1000, 70/1000, 94/1000]})

def add_benford(df,observed_count_name):
  import numpy as np
  import pandas as pd
  
  benford_df = (df.assign(
      expected_count = lambda x: x[observed_count_name].sum() * \
      np.array([.301, .176, .125, .097, .079, .067, .058, .051, .046], dtype=np.float64),
      expected_mean = [.301, .176, .125, .097, .079, .067, .058, .051, .046]))
  return benford_df

final_table = add_benford(pivoted_data, 'observed_count')

def chi2_test(df,obs,exp,alpha,dof,ddof):
  import scipy.stats

  statistic = round(scipy.stats.chisquare(df[obs],
                      df[exp])[0], 2)
  critical_value = round(scipy.stats.chi2.ppf(1-alpha,dof-ddof), 2)
  if statistic > critical_value:
    print(f'The statistic ({statistic}) is more than a critical value for {alpha} significance level \n and {dof-ddof} degrees of freedom ({critical_value}) \n then H0 should be rejected')
  else:
    print(f'The statistic ({statistic}) is less than a critical value for {alpha} significance level \n and {dof-ddof} degrees of freedom ({critical_value}) \n then H0 should not be rejected')
    
 chi2_test(final_table,'observed_count','expected_count',.05,len(final_table),1)   
