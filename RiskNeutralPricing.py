import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('TkAgg')  # Usa el backend TkAgg
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy import stats, interpolate

# Cargar solo las columnas "Fecha" y "Valor" del archivo "temp_max.csv"
max_temp = pd.read_csv("temp_max.csv", usecols=["Fecha", "Valor"], parse_dates=["Fecha"], date_parser=lambda x: pd.to_datetime(x).date())

# Cargar solo las columnas "Fecha" y "Valor" del archivo "temp_min.csv"
min_temp = pd.read_csv("temp_min.csv", usecols=["Fecha", "Valor"], parse_dates=["Fecha"], date_parser=lambda x: pd.to_datetime(x).date())

# Cambiar el nombre de la columna "Valor" a "Tmax" para max_temp
max_temp.rename(columns={'Valor': 'Tmax'}, inplace=True)

# Cambiar el nombre de la columna "Valor" a "Tmin" para min_temp
min_temp.rename(columns={'Valor': 'Tmin'}, inplace=True)

# Cambiar el nombre de la columna "Valor" a "Tmax" para max_temp
max_temp.rename(columns={'Fecha': 'Date'}, inplace=True)

# Cambiar el nombre de la columna "Valor" a "Tmin" para min_temp
min_temp.rename(columns={'Fecha': 'Date'}, inplace=True)


# Establecer la columna "Fecha" como índice para max_temp y min_temp
max_temp.set_index('Date', inplace=True)
min_temp.set_index('Date', inplace=True)


temps = max_temp.merge(min_temp,how='inner',left_on=['Date'],right_on=['Date'])

def avg_temp(row):
    return (row.Tmax+row.Tmin)/2

temps['T'] = temps.apply(avg_temp,axis=1)

# drop na values here
temps = temps.dropna()

temp_t = temps['T'].copy(deep=True)
temp_t = temp_t.to_frame()
first_ord = temp_t.index.map(dt.datetime.toordinal)[0]

temp_vol = temps['T'].copy(deep=True).to_frame()
temp_vol['day'] = temp_vol.index.dayofyear
temp_vol['month'] = temp_vol.index.month

vol = temp_vol.groupby(['day'])['T'].agg(['mean','std'])
days = np.array(vol['std'].index)
T_std = np.array(vol['std'].values)

temp_t.head()



def T_model(x, a, b, alpha, theta):
    omega = 2*np.pi/365.25
    T = a + b*x + alpha*np.sin(omega*x + theta)
    return T

def dT_model(x, a, b, alpha, theta):
    omega=2*np.pi/365.25
    dT =  b + alpha*omega*np.cos(omega*x + theta)
    return dT

def spline(knots, x, y):
    x_new = np.linspace(0, 1, knots+2)[1:-1]
    t, c, k = interpolate.splrep(x, y, t=np.quantile(x, x_new), s=3)
    yfit = interpolate.BSpline(t,c, k)(x)
    return yfit

Tbar_params = [16.8, 3.32e-05, 5.05, 1.27]

def euler_step(row, kappa, M, lamda):
    """Function for euler scheme approximation step in
    modified OH dynamics for temperature simulations
    Inputs:
    - dataframe row with columns: T, Tbar, dTbar and vol
    - kappa: rate of mean reversion
    Output:
    - temp: simulated next day temperatures
    """
    if row['T'] != np.nan:
        T_i = row['Tbar']
    else:
        T_i = row['T']
    T_det = T_i + row['dTbar']
    T_mrev =  kappa*(row['Tbar'] - T_i)
    sigma = row['vol']*np.random.randn(M)
    riskn = lamda*row['vol']
    return T_det + T_mrev + sigma - riskn

def monte_carlo_temp(trading_dates, Tbar_params, vol_model, first_ord, M=1, lamda=0):
    """Monte Carlo simulation of temperature
    Inputs:
    - trading_dates: pandas DatetimeIndex from start to end dates
    - M: number of simulations
    - Tbar_params: parameters used for Tbar model
    - vol_model: fitted volatility model with days in year index
    - first_ord: first ordinal of fitted Tbar model
    Outputs:
    - mc_temps: DataFrame of all components and simulated temperatures
    """
    kappa=0.438
    if isinstance(trading_dates, pd.DatetimeIndex):
        trading_date=trading_dates.map(dt.datetime.toordinal)

    Tbars = T_model(trading_date-first_ord, *Tbar_params)
    dTbars = dT_model(trading_date-first_ord, *Tbar_params)
    mc_temps = pd.DataFrame(data=np.array([Tbars, dTbars]).T,
                            index=trading_dates, columns=['Tbar','dTbar'])
    mc_temps['day'] = mc_temps.index.dayofyear
    mc_temps['vol'] = vol_model[mc_temps['day']-1]

    mc_temps['T'] = mc_temps['Tbar'].shift(1)
    data = mc_temps.apply(euler_step, args=[kappa, M, lamda], axis=1)
    mc_sims = pd.DataFrame(data=[x for x in [y for y in data.values]],
                 index=trading_dates,columns=range(1,M+1))
    return mc_temps, mc_sims

# define trading date range
trading_dates = pd.date_range(start='2015-09-01', end='2016-08-31', freq='D')
volatility = spline(5, days, T_std)
mc_temps, mc_sims = monte_carlo_temp(trading_dates, Tbar_params, volatility, first_ord)

plt.figure(figsize=(10,6))
mc_sims[1].plot(alpha=0.5,linewidth=1, marker='*')
mc_temps["Tbar"].plot(linewidth=3)
plt.legend(loc='lower right')
plt.show()



trading_dates = pd.date_range(start='2016-01-01', end='2017-07-31', freq='D')
volatility = spline(5, days, T_std)
mc_temps, mc_sims = monte_carlo_temp(trading_dates, Tbar_params, volatility, first_ord)

print('Probability P(max(18-Tn, 0) = 0): {0:1.1f}%'.format(len(mc_sims[mc_sims[1] >= 18])/len(mc_sims)*100))


def rn_mean(time_arr, vol_arr, Tbars, lamda, kappa):
    """Evaluate the risk neutral integral above for each segment of constant volatility
    Rectangular integration with step size of days
    """
    dt = 1/365.25
    N = len(time_arr)
    mean_intervals = -vol_arr*(1 - np.exp(-kappa*dt))/kappa
    return 18*N - (np.sum(Tbars) - lamda*np.sum(mean_intervals))


def rn_var(time_arr, vol_arr, kappa):
    """Evaluate the risk neutral integral above for each segment of constant volatility
    Rectangular integration with step size of days
    """
    dt = 1/365.25
    var_arr = np.power(vol_arr,2)
    var_intervals = var_arr/(2*kappa)*(1-np.exp(-2*kappa*dt))
    cov_sum = 0
    for i, ti in enumerate(time_arr):
        for j, tj in enumerate(time_arr):
            if j > i:
                cov_sum += np.exp(-kappa*(tj-ti)) * var_intervals[i]
    return np.sum(var_intervals) + 2*cov_sum



def risk_neutral(trading_dates, Tbar_params, vol_model, first_ord, lamda, kappa=0.438):
    if isinstance(trading_dates, pd.DatetimeIndex):
        trading_date=trading_dates.map(dt.datetime.toordinal)

    Tbars = T_model(trading_date-first_ord, *Tbar_params)
    dTbars = dT_model(trading_date-first_ord, *Tbar_params)
    mc_temps = pd.DataFrame(data=np.array([Tbars, dTbars]).T,
                            index=trading_dates, columns=['Tbar','dTbar'])
    mc_temps['day'] = mc_temps.index.dayofyear
    mc_temps['vol'] = vol_model[mc_temps['day']-1]
    time_arr = np.array([i/365.25 for i in range(1,len(trading_dates)+1)])
    vol_arr = mc_temps['vol'].values
    mu_rn = rn_mean(time_arr, vol_arr, Tbars, lamda, kappa)
    var_rn = rn_var(time_arr, vol_arr, kappa)
    return mu_rn, var_rn

    
def winter_option(trading_dates, r, alpha, K, tau, opt='c', lamda=0.0):
    """Evaluate the fair value of temperature option in winter
    Based on heating degree days only if the physical probability that
    the average temperature exceeds the Tref (18 degC) is close to 0
    """
    mu_rn, var_rn = risk_neutral(trading_dates, Tbar_params, volatility, first_ord, lamda)
    disc = np.exp(-r*tau)
    vol_rn = np.sqrt(var_rn)
    zt = (K - mu_rn)/vol_rn
    exp = np.exp(-zt**2/2)
    if opt == 'c':
        return alpha*disc*((mu_rn - K)*stats.norm.cdf(-zt) + vol_rn*exp/np.sqrt(2*np.pi))
    else:
        exp2 = np.exp(-mu_rn**2/(2*vol_rn**2))
        return alpha*disc*((K - mu_rn)*(stats.norm.cdf(zt) - stats.norm.cdf(-mu_rn/vol_rn)) +
                           vol_rn/np.sqrt(2*np.pi)*(exp-exp2))
        

trading_dates = pd.date_range(start='2023-06-01', end='2023-08-31', freq='D')
r=0.05
K=300
alpha=2500

def years_between(d1, d2):
    d1 = dt.datetime.strptime(d1, "%Y-%m-%d")
    d2 = dt.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)/365.25

start = dt.datetime.today().strftime('%Y-%m-%d')
end = '2023-08-31'

tau = years_between(start, end)

print('Start Valuation Date:', start,
      '\nEnd of Contract Date:', end,
      '\nYears between Dates :', round(tau,3))


print('Call Price: ', round(winter_option(trading_dates, r, alpha, K, tau, 'c'),2))
print('Put Price : ', round(winter_option(trading_dates, r, alpha, K, tau, 'p'),2))


# define trading date range
trading_dates = pd.date_range(start='2023-06-01', end='2023-08-31', freq='D')
no_sims = 10000
vol_model = spline(5, days, T_std)


def temperature_option(trading_dates, no_sims, Tbar_params, vol_model, r, alpha, K, tau, first_ord, opt='c'):
    "Evaluates the price of a temperature call option"
    mc_temps, mc_sims = monte_carlo_temp(trading_dates, Tbar_params, volatility, first_ord, no_sims)
    N, M = np.shape(mc_sims)
    mc_arr = mc_sims.values
    DD = np.sum(np.maximum(18-mc_arr,0), axis=0)
    if opt == 'c':
        CT = alpha*np.maximum(DD-K,0)
    else:
        CT = alpha*np.maximum(K-DD,0)
    C0 = np.exp(-r*tau)*np.sum(CT)/M
    sigma = np.sqrt( np.sum( (np.exp(-r*tau)*CT - C0)**2) / (M-1) )
    SE = sigma/np.sqrt(M)
    return C0, SE


call = np.round(temperature_option(trading_dates, no_sims, Tbar_params, vol_model, r, alpha, K, tau, first_ord, 'c'),2)
put = np.round(temperature_option(trading_dates, no_sims, Tbar_params, vol_model, r, alpha, K, tau, first_ord, 'p'),2)
print('Call Price: {0} +/- {1} (2se)'.format(call[0], call[1]*2))
print('Put Price : {0} +/- {1} (2se)'.format(put[0], put[1]*2))


strikes = np.arange(180,520,20)
strikes

data = np.zeros(shape=(len(strikes),4))
for i, strike in enumerate(strikes):
    data[i,0] = temperature_option(trading_dates, no_sims, Tbar_params, vol_model, r, alpha, strike, tau, first_ord, 'c')[0]
    data[i,1] = winter_option(trading_dates, r, alpha, strike, tau, 'c')
    data[i,2] = temperature_option(trading_dates, no_sims, Tbar_params, vol_model, r, alpha, strike, tau, first_ord, 'p')[0]
    data[i,3] = winter_option(trading_dates, r, alpha, strike, tau, 'p')
    
df = pd.DataFrame({'MC Call': data[:, 0], 'BSA Call': data[:, 1], 'MC Put': data[:, 2], 'BSA Put': data[:, 3]})
df.index = strikes

plt.plot(df)
plt.title('Winter Temperature Options')
plt.ylabel('Option Premium $USD')
plt.xlabel('Heating Degree Days')
plt.legend(df.columns, loc=4)
plt.show()
