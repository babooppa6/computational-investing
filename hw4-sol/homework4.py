'''

@author: Xavier MENAGE

'''

from math import *
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep


def orders_from_events(cash, ls_symbols, df_close, df_actual_close):
    print "Finding Events"

    # Detect events
    df_test = pd.DataFrame(False,index=df_actual_close.index,columns=df_actual_close.columns)
    np_yesterday = df_actual_close.values[:-1,:] >= 5
    np_today = df_actual_close.values[1:,:] < 5
    df_test.values[0,:] = False
    df_test.values[1:,:] = np.logical_and(np_today, np_yesterday)
    # Identify buy orders and calculate quantity variation
    df_bool_buy = df_test.copy()
    df_bool_buy.values[-5:,:]=False
    df_variation = pd.DataFrame(0,index=df_actual_close.index,columns=df_actual_close.columns)
    df_variation[df_bool_buy] = df_variation[df_bool_buy] + 100
    # Generate sell orders and calculate quantity variation
    df_bool_sell = pd.DataFrame(False,index=df_actual_close.index,columns=df_actual_close.columns)
    df_bool_sell.values[5:,:]=df_bool_buy.values[:-5,:]
    df_variation[df_bool_sell] = df_variation[df_bool_sell] - 100
    df_select = df_variation.loc[df_variation.index[(df_variation<>0).any(1)],df_variation.columns[(df_variation<>0).any(0)]]
    # find first buy order date and last sell order date
    dt_first = df_select.index[0]
    dt_last = df_select.index[-1]
    # print df_select.to_string()
    # Calculate portfolio value across all S&P 500 symbols
    df_quantity = df_variation.cumsum(axis=0)
    df_value = df_quantity.mul(df_close)
    s_stock = df_value.sum(axis=1)
    # Calculate cash variation during period
    df_cash_flow = - df_variation.mul(df_close)
    s_cash = df_cash_flow.sum(axis=1)
    s_cash = s_cash.cumsum()
    s_cash = s_cash + cash
    # Calculate overall portfolio value
    s_portf = s_cash + s_stock
    # Return data necessary to measure portfolio performance
    return s_portf, s_cash, dt_first, dt_last
 
# Measure portfolio performance
def portf_performance(bench_sym, s_portf, s_cash, s_bench):    
    s_rets = s_portf.copy()
    tsu.returnize0(s_rets.values)
    PortfStdev = s_rets.std(ddof=0)
    PortfMean = s_rets.mean()
    PortfSharpe = sqrt(252)*PortfMean/PortfStdev
    PortfAnnualRet = s_portf.values[-1]/s_portf.values[0]
    s_rets = s_bench.copy()
    BenchAnnualRet = s_rets.values[-1]/s_rets.values[0]
    tsu.returnize0(s_rets.values)
    BenchStdev = s_rets.std(ddof=0)
    BenchMean = s_rets.mean()
    BenchSharpe = sqrt(252)*BenchMean/BenchStdev
    print "The final value of the portfolio using the sample file is -- %s %s" % (s_cash.index[-1].strftime('%Y-%m-%d'), s_portf[-1])
    print "Details of the Performance of the portfolio :\n"
    print "Data Range : %s to %s" % (s_portf.index[0].strftime('%Y-%m-%d'), s_portf.index[-1].strftime('%Y-%m-%d'))
    print
    print "Sharpe Ratio of Fund : %f" % PortfSharpe
    print "Sharpe Ratio of %s : %f" % (bench_sym, BenchSharpe)
    print
    print "Total Return of Fund : %f" % PortfAnnualRet
    print "Total Return of %s : %f" % (bench_sym, BenchAnnualRet)
    print
    print "Standard Deviation of Fund : %f" % PortfStdev
    print "Standard Deviation of %s : %f" % (bench_sym, BenchStdev)
    print
    print "Average Daily Return of Fund : %f" % PortfMean
    print "Average Daily Return of %s : %f" % (bench_sym, BenchMean)

    plt.clf()
    s_portf.plot()
    s_bench.plot()
    plt.ylabel('Value')
    plt.xlabel('Date')
    plt.legend(['Fund', bench_sym], loc='best')
#    plt.show()
    plt.savefig('homework4.png', format='png')
    print "chart saved"    
    
def main():
    cash = 50000.0
    bench_sym = '$SPX'

    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    print "working with S&P 500 2012"
    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list('sp5002012')
#    ls_symbols = ls_symbols[:100]
    ls_symbols.append(bench_sym)
#    ls_symbols.append('GLD')
    #print "ls_symbols ", ls_symbols

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    print "reading history data"
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    print "building d_data"
    d_data = dict(zip(ls_keys, ldf_data))
    #print "d_data ", d_data

    print "fill the holes"
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    print "call orders_from_events"
    df_actual_close = d_data['actual_close']
    df_close = d_data['close']
    s_portf, s_cash, dt_first, dt_last = orders_from_events(cash, ls_symbols, df_close, df_actual_close)
    s_bench = df_close[bench_sym].copy()
    s_bench = s_bench/s_bench[0]*cash
    portf_performance(bench_sym, s_portf[dt_first:dt_last], s_cash[dt_first:dt_last], s_bench[dt_first:dt_last])
#    print "Creating Study with S&P 2012"
#    ep.eventprofiler(df_events, d_data, i_lookback=20, i_lookforward=20,
#                s_filename='hw2_sp5002012.pdf', b_market_neutral=True, b_errorbars=True,
#                s_market_sym='SPY')

if __name__ == '__main__':
    main()