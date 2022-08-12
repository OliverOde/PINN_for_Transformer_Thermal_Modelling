# Function for Simulating Fake Data (Ambient temperature, Top-oil temperature and Load factor)

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

def simulate_data(): 

    # Moving average time series
    # Load factor
    sim_data_df = pd.DataFrame(columns=['t', 'K', 'Tamb', 'Ttop'])
    ar = np.array([1])
    ma = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    MA_object = ArmaProcess(ar, ma)
    simulated_data_1 = MA_object.generate_sample(nsample=600) + 0.35
    
    # Ambient and Top-oil temperature
    ar2 = np.array([1])
    ma2 = np.array([1, 1.7, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    MA_object2 = ArmaProcess(ar2, ma2)
    simulated_data_2 = MA_object2.generate_sample(nsample=600) + 2
    simulated_data_3 = simulated_data_2 + 17 + np.random.rand(600)*5

    # Insert average in between values 
    sim_1 = np.empty(2 * simulated_data_1.size -1)
    sim_1[::2] = simulated_data_1
    sim_1[1::2] = simulated_data_1[:-1] + np.diff(simulated_data_1)/2
    sim_data_df['K'] = np.clip(sim_1, 0, 0.7)

    sim_2 = np.empty(2 * simulated_data_2.size -1)
    sim_2[::2] = simulated_data_2
    sim_2[1::2] = simulated_data_2[:-1] + np.diff(simulated_data_2)/2
    sim_data_df['Tamb'] = sim_2

    sim_3 = np.empty(2 * simulated_data_3.size -1)
    sim_3[::2] = simulated_data_3
    sim_3[1::2] = simulated_data_3[:-1] + np.diff(simulated_data_3)/2
    sim_data_df['Ttop'] = sim_3

    sim_data_df['t'] = np.arange(len(sim_data_df))
    
    return sim_data_df
    