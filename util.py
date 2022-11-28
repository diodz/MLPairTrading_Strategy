# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:08:31 2021

@author: Diego Díaz
"""

import quandl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
import pickle
import sklearn.cluster as cluster
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

warnings.filterwarnings('ignore')

quandl.ApiConfig.api_key = "b62EV6q2buHEPzUv4zpa"

### Next we add some constant that are used in the strategy
start_date="2018-12-02"
end_date="2020-12-31"

####
#### Section 2.1 from Trading Strategy - Technical paper
#### Helper function source code for getting Zachs data
####

def get_data_zachs_fc(ticker, year):
    date1 = str(year-1) + '-12-31'
    date2 = str(year) + '-03-31'
    date3 = str(year) + '-06-30'
    date4 = str(year) + '-09-30'
    dates = [date1, date2, date3, date4]

    FC = quandl.get_table('ZACKS/FC',ticker = ticker, per_end_date= dates,\
                      qopts={"columns":['zacks_sector_code', 'emp_cnt',\
                                        'basic_net_eps', 'ebitda', 'compr_income', 'net_income_loss']}, paginate=True)
    FC = pd.DataFrame(FC.mean()).rename(columns={0:ticker})
    if sum(FC[ticker].isnull().values) >= 1:
        return pd.DataFrame()
    return FC

def read_tickers():
    url = 'https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv'
    df = pd.read_csv(url)
    lst = []
    df = df[(df['Exchange'] == 'NASDAQ') | (df['Exchange'] == 'NYSE')]
    for ticker in df['Ticker']:
        if '.' in ticker:
            continue
        lst.append(ticker)
    return lst

def add_yearly_returns(zachs, prices):
    #we get the yearly returns (we add 253 as there are 253 trading days in the year)
    yearly_returns = prices.iloc[[0,253],:].pct_change().iloc[1,:].dropna()
    zachs2 = zachs[list(yearly_returns.T.index)]
    df = zachs2.append(yearly_returns)
    return df

def download_zachs_data(year):
    year = year
    tickers = read_tickers()
    data_zachs = get_data_zachs_fc(tickers[3], year)
    for ticker in tqdm(tickers[4::]):
        ticker_zachs = get_data_zachs_fc(ticker, year)
        if ticker_zachs.empty:
            continue
        data_zachs = data_zachs.join(ticker_zachs)
    
    return data_zachs

def execute_zachs_download():
    year = 2018
    tickers = read_tickers()
    data_zachs = get_data_zachs_fc(tickers[1], year)
    for ticker in tqdm(tickers):
        ticker_zachs = get_data_zachs_fc(ticker, year)
        if ticker_zachs.empty:
            continue
        data_zachs = data_zachs.join(ticker_zachs)
    
    return data_zachs

def save_to_pickle(data_zachs):
    #we save the data to a pickle
    with open('data_zachs_final.pkl', 'wb') as f:
        pickle.dump(data_zachs, f)
        
def load_zachs():
    #we load the pickle of zachs data in case we are working on a different session
    #this allows for faster access to data 
    with open('data_zachs_final.pkl', 'rb') as f:
        data_zachs = pickle.load(f)
    return data_zachs
    
###
### Section 2.2: Download daily close prices to get returns
###

def get_prices(tickers):
    start_date = "2018-12-01"
    end_date = "2020-12-31"
    tickers = list(tickers)
    df_prices = quandl.get("EOD/" + tickers[0], start_date=\
                           start_date, end_date=end_date)['Adj_Close']\
        .rename(tickers[0])
    
    for ticker in tqdm(tickers[1:]):
        if '.' in ticker:
            continue
        try:
            ticker_prices = quandl.get("EOD/" + ticker, start_date=start_date,\
                                       end_date=end_date)['Adj_Close']
            ticker_prices = ticker_prices.rename(ticker)
            df_prices = pd.concat([df_prices, ticker_prices], axis=1)
                
        except:
            print("Error with ticker: " + ticker)
            
    return df_prices

def load_prices():
    with open('prices_eod_final.pkl', 'rb') as f:
        prices = pickle.load(f)
    return prices

def combine_zachs_prices(data_zachs, prices):
    #We combine the data sources and rename the index or return data to year_returns.
    final = add_yearly_returns(data_zachs, prices)
    lst = list(final.index[:-1])
    lst.append('year_returns')
    final.index = lst
    return final

###
### Section 2.3 K-means clustering
###

def kmeans_clustering(final):
    #Performs Kmeans clustering and returns clusters, prints the first 
    #10 clusters as well
    kmeans = cluster.KMeans(n_clusters=1000 ,init="k-means++")
    df = final.T[['emp_cnt', 'basic_net_eps', 'ebitda', 'compr_income', 'year_returns', 'net_income_loss']]
    kmeans = kmeans.fit(df)
    df['Clusters'] = kmeans.labels_
    print(df['Clusters'].value_counts()[:10])
    cluster_counts = df['Clusters'].value_counts()
    cluster_counts = cluster_counts[cluster_counts >= 2]
    return df, cluster_counts
    
def get_combinations_of_pairs(cluster_counts, df):
    target_clusters = list(cluster_counts.index)
    
    dct = {}
    for n_cluster in target_clusters:
        dct[n_cluster] = list(df[df['Clusters'] == n_cluster].index)
        
    pairs = []
    for e in tqdm(dct):
        pairs += list(itertools.combinations(dct[e], 2))
    return pairs

def save_pairs(pairs):
    with open('pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)
        
def load_pairs():
    with open('pairs.pkl', 'rb') as f:
        pair_tickers = pickle.load(f)
        return pair_tickers
    
###
### Section 3.1: Calculating Pearson correlation coefficient
###

def calculating_pearson(prices, pairs):
    final_pairs = []
    pearson_pairs = []
    returns = prices.pct_change()
    for pair in tqdm(pairs):
        pair_returns = returns[list(pair)]
        pearson = pair_returns.astype(float).corr().iloc[0,1]
        final_pairs.append(pair)
        pearson_pairs.append(pearson)
        
    df = pd.DataFrame(list(zip(final_pairs, pearson_pairs)),
                      columns =['pair', 'pearson_corr'])
    return df

def sort_and_select_pairs(df):
    df = df.sort_values('pearson_corr', ascending=False)
    selected_pairs = df[df['pearson_corr'] <= 0.95][:50]
    return selected_pairs

def save_selected_pairs(df):
    with open('selected_pairs.pkl', 'wb') as f:
        pickle.dump(df, f)
        
def load_selected_pairs():
    with open('selected_pairs.pkl', 'rb') as f:
        selected_pairs = pickle.load(f)
        return selected_pairs
    
###
### Section 4.1: Implementing the spread trading strategy
### Helper functions

def get_prices_ticker(ticker, start_date, end_date):
    df = quandl.get("EOD/"+ticker, start_date=start_date, end_date=end_date)
    return df

def process_data(data, m_days=1):
    df = data.copy(deep=True)
    df['avg_dollar_vol'] = df['Close'] * df['Adj_Volume']
    df['N'] = df['avg_dollar_vol'].rolling(15).median()
    lst = ['Adj_Close', 'N', 'returns', 'first_day', 'last_day']
    df = estimate_m_day_returns(df, m_days)
    df = first_last_day_of_month(df)
    return df[lst]

def first_last_day_of_month(df):
    df['day'] = df.index.day
    df['day_diff_yesterday'] = (df['day'] - df['day'].shift(1))
    df['day_diff_tomorrow'] = (df['day'] - df['day'].shift(-1))
    df['first_day'] = df['day_diff_yesterday'] < 0
    df['last_day'] = df['day_diff_tomorrow'] > 0
    df['last_day'][-1] = True
    return df

def estimate_m_day_returns(df, m_days):
    df['returns'] = np.log(df['Adj_Close']/df['Adj_Close'].shift(m_days))
    return df

def get_max_capital(df1):
    return 2 * df1['N'].max() / 50

def make_spread_df(df_1, df_2, tickers):
    #change to higher liquidity equity 
    df_1['N2'] = df_2['N']
    spread = df_1['returns'] - df_2['returns']
    minimum_N = df_1[['N','N2']].min(axis=1)
    df_spread = pd.concat([spread, minimum_N], axis=1)
    df_spread[tickers[0]] = df_1['Adj_Close']
    df_spread[tickers[1]] = df_2['Adj_Close']
    df_spread['first_day'] = df_2['first_day']
    df_spread['last_day'] = df_2['last_day']
    df_spread.rename(columns={0: "N", "returns": "spread"}, inplace=True)
    return df_spread


###
### Section 4.1: Implementing the spread trading strategy
### SpreadTrading class to handle the mechanics of the strategy
### Each object will be a pair and will keep track of all its metrics (such 
### as profits, positions, etc.)
###


class SpreadTrading:
    "A quantitative trading strategy"
    #
    def __init__(self, stop_loss, tickers):
        self.position_1 = [0]
        self.position_2 = [0]
        self.pnl = [0]
        self.e1_last = 0
        self.e2_last = 0
        self.position_profit_1 = 0
        self.position_profit_2 = 0
        self.can_trade = True
        self.stop_loss = stop_loss
        self.dates_traded = []
        self.dates_close_positions = []
        self.ticker_1 = tickers[0]
        self.ticker_2 = tickers[1]
        self.pair = tickers
        self.have_open_trades = [0]
        
    def open_trade(self, row, date):
        if row[self.ticker_1] > row[self.ticker_2]:
            self.position_1.append(-row['N']/1000)
            self.position_2.append(row['N']/1000)
        else:
            self.position_1.append(row['N']/1000)
            self.position_2.append(-row['N']/1000)
        
        self.can_trade = True
        self.update_pnl(row, date)
        self.add_date(date)
        self.have_open_trades.append(1)
    
    def do_not_trade(self, row, date, open_trades):
        self.position_1.append(self.position_1[-1])
        self.position_2.append(self.position_2[-1])
        self.update_pnl(row, date)
        self.check_stop_loss(row, date, open_trades)
        currently_trading = self.have_open_trades[-1]
        self.have_open_trades.append(currently_trading)
        
    def close_trade(self, row, date):
        self.update_pnl(row, date)
        self.position_1.append(0)
        self.position_2.append(0)
        self.position_profit_1 = 0
        self.position_profit_2 = 0
        self.add_close_date(date)
        self.have_open_trades.append(0)
        
    def update_pnl(self, row, date='None'):
        daily_profit_1 = (row[self.ticker_1] - self.e1_last) * self.position_1[-1]
        daily_profit_2 = ((row[self.ticker_2] - self.e2_last)) * self.position_2[-1]
        self.position_profit_1 += daily_profit_1
        self.position_profit_2 += daily_profit_2
        
        self.pnl.append(daily_profit_1+daily_profit_2)
        self.e1_last = row[self.ticker_1]
        self.e2_last = row[self.ticker_2]
        
    def check_stop_loss(self, row, date, open_trades):
        if not open_trades:
            return None
        if (self.position_profit_1 + self.position_profit_2) < - self.stop_loss * 2 * abs(self.position_1[-1]):
            self.position_1.append(0)
            self.position_2.append(0)
            self.position_profit_1 = 0
            self.position_profit_2 = 0
            self.can_trade = False
            self.add_close_date(date)
            
    def add_date(self, date):
        self.dates_traded.append(str(date))
        
    def add_close_date(self, date):
        self.dates_close_positions.append(str(date))
        
###
### Section 4.1:
### Plot cumulative profit and loss from spread trading object


def plot_cumulative_pnl(spread_trading, pair):
    "Helper function to plot cumulative profits from trading strategy"
    new_list=[]
    j = 0
    n = len(spread_trading.pnl)
    for i in range(0, n):
        j+=spread_trading.pnl[i]
        new_list.append(j)

    plt.plot(new_list)
    plt.xlabel('Days since beggining of strategy')
    plt.ylabel('Accumulated profit & loss [$]')
    plt.title('Pair trading strategy for '+pair[0]+' - '+pair[1])
    plt.grid()
    
def example_spread_calculation(selected_pairs):
    start_date="2018-12-02"
    end_date="2020-12-31"
    
    m_days = 1
    
    list_of_pairs = list(selected_pairs['pair'])
    
    df_1 = get_prices_ticker(list_of_pairs[0][0], start_date, end_date)
    df_2 = get_prices_ticker(list_of_pairs[0][1], start_date, end_date)
    
    df_1 = process_data(df_1, m_days)
    df_2 = process_data(df_2, m_days)
    
    df_spread = make_spread_df(df_1, df_2, list_of_pairs[0])
    return df_spread

###
### Section 4.1 
### The following function executes the full strategy for a pair

def execute_strategy(df_spread, stop_loss, g, j, pair):
    "Executes trading strategy for a given set of parameters"
    have_open_trades = False
    can_start_trading = False
    can_trade = True
    n = df_spread.shape[0]
    spread_trading = SpreadTrading(stop_loss, pair)

    for i in range(0,n):
        row = df_spread.iloc[i]
        if row['first_day']:
            can_start_trading = True
            can_trade = True
        
        if not can_start_trading:
            continue
            
        if have_open_trades:
            if row['last_day'] or row['spread'] < j:
                spread_trading.close_trade(row, df_spread.index[i])
                have_open_trades = False
                continue
          
        elif (row['spread'] > g) and can_trade:
            spread_trading.open_trade(row, df_spread.index[i])
            have_open_trades = True
            continue

        spread_trading.do_not_trade(row, df_spread.index[i], have_open_trades)
        can_trade = spread_trading.can_trade
        if not can_trade:
            have_open_trades = can_trade
    return (sum(spread_trading.pnl), spread_trading)
       
 
def trade_example_pair(pair):
    m_days = 1
    s_loss = 0.25
    g = 0.02
    j = 0.001

    start_date="2018-12-02"
    end_date="2020-12-31"
    
    df_1 = get_prices_ticker(pair[0], start_date, end_date)
    df_2 = get_prices_ticker(pair[1], start_date, end_date)
    
    df_1 = process_data(df_1, m_days)
    df_2 = process_data(df_2, m_days)
    
    df_spread = make_spread_df(df_1, df_2, pair)
    total_k = get_max_capital(df_spread)
    pnl = execute_strategy(df_spread, s_loss, g, j, pair)
    roi = ((pnl[0] + total_k) / total_k - 1)
    plot_cumulative_pnl(pnl[1], pair)
    return pnl, roi


def trade_pair(pair, start_date, end_date, m_day, s_loss, g, j):

    df_1 = get_prices_ticker(pair[0], start_date, end_date)
    df_2 = get_prices_ticker(pair[1], start_date, end_date)

    df_1 = process_data(df_1, m_day)
    df_2 = process_data(df_2, m_day)
    
    df_spread = make_spread_df(df_1, df_2, pair)
    total_k = get_max_capital(df_spread)

    pnl = execute_strategy(df_spread, s_loss, g, j, pair)
    roi = ((pnl[0] + total_k) / total_k - 1) * 100
    
    return pnl, roi, total_k


###
### Section 4.2: Implementing the spread trading strategy for all pairs
### 

def trade_all_pairs(list_of_pairs):
    results = []
    rois = []
    capital = []
    
    m_days = 1
    s_loss = 0.25
    g = 0.02
    j = 0.001
    
    for pair in tqdm(list_of_pairs):
        pnl, roi, k = trade_pair(pair, start_date, end_date, m_days, s_loss, g, j)
        results.append(pnl)
        rois.append(roi)
        capital.append(k)
        
    return results, rois, capital


def plot_all_pairs_pnl(results, pairs):
    number_of_trades = []
    for i, res in enumerate(results):
        plot_cumulative_pnl(res[1], pairs[i])
        plt.title('Pair trading all pairs')
        number_of_trades.append(len(res[1].dates_traded))
    return number_of_trades


###
### Section 5.1: Trading patterns and profit and loss
###

        
def count_and_plot_open_trades(results, list_of_pairs):
    dct = {}
    for i, res in enumerate(results):
        dct[res[1].pair] = res[1].have_open_trades
        
    open_trades = pd.DataFrame(dct, columns=list_of_pairs)
    plt.plot(open_trades.sum(axis=1))
    plt.grid()
    plt.title('Trading pattern during backtesting')
    plt.xlabel('Days since beginning of strategy')
    plt.ylabel('Active number of spread positions')
    return open_trades

def calculate_total_daily_pnl(results, list_of_pairs):
    dct = {}
    for res in results:
        dct[res[1].pair] = res[1].pnl
        
    total_pnl = pd.DataFrame(dct, columns=list_of_pairs)
    plt.plot(total_pnl.sum(axis=1))
    plt.grid()
    plt.title('Daily PnL during backtesting period')
    plt.xlabel('Days since beginning of strategy')
    plt.ylabel('Daily profit and loss [$]')  
    return total_pnl

def get_aggregated_pnl(total_pnl):
    aggregated_pnl = total_pnl.sum(axis=1)
    n = len(aggregated_pnl)
    j = 0
    new_list = []
    for i in range(0, n):
        j+=aggregated_pnl[i]
        new_list.append(j)
    
    plt.plot(new_list)
    plt.xlabel('Days since beggining of strategy')
    plt.ylabel('Accumulated profit & loss')
    plt.title('Pair trading strategy for all pairs')
    plt.grid()
    
def scatter_pnl(results):
    total_pnl = []
    pairs = []
    for res in results:
        total_pnl.append(res[0])
        pairs.append(res[1].pair)
    plt.scatter(list(range(50)), total_pnl)
    plt.xlabel('Pair number')
    plt.ylabel('Total profit [$]')
    plt.title('Profit from each pair during the whole period')
    plt.grid()
    return total_pnl

def scatter_roi(rois):
    total_roi = []
    for roi in rois:
        total_roi.append(roi)
    plt.scatter(list(range(50)), total_roi)
    plt.xlabel('Pair number')
    plt.ylabel('Total ROI as a fraction')
    plt.title('Return on investment per pair')
    plt.grid()
    return total_roi
   
def count_profitable_pairs(total_pnl):
    profitable_pairs = []
    for pnl in total_pnl:
        if pnl > 0:
            profitable_pairs.append(pnl)
    return len(profitable_pairs)
    
###
### Section 5.2: Total daily returns of strategy
### Helper functions to study returns - modified from solution to assignment 2
### from from FINM 31150 - Canvas


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())

                         
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r.dropna(), level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns 
    in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=252)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=252)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, \
                         periods_per_year=252)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

###
### Section 5.2: Total daily returns of strategy and comparison to 
### Fama-French factor loadings

def get_strategy_returns(new_list, capital):
    df_returns = pd.DataFrame(new_list+max(capital)).\
        rename(columns={0:'Strategy Returns'}).pct_change()
    return df_returns

def load_ff3():
    """
    Daily annualized returns from Fama-French website. 
    """
    df_ff3 = pd.read_csv('ff3.csv', index_col=None)
    df_ff3['Date'] = pd.to_datetime(df_ff3.Date, format='%Y%m%d')
    df_ff3.set_index('Date', inplace=True)
    return df_ff3[start_date:end_date]

def ols_regression_ff(df_returns):
    df_ff3 = load_ff3()
    df_returns.index = df_ff3[18:].index
    
    df_factor_loading = df_returns.merge(df_ff3, how='left', left_index=True,\
                                         right_index=True).dropna()
    
    # annualize the log returns by multiplying with 252 and then subtract.
    # FF3 returns are already annualized
    
    res = sm.OLS((df_factor_loading['Strategy Returns'] * 252 - df_factor_loading['RF']),
                 sm.add_constant(df_factor_loading[['Mkt-RF', 'SMB', 'HML']])).fit()
    
    print(res.summary())
    return res

