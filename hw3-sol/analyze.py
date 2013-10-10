"""
Author     : Tolunay Orkun <tolunay(at)orkun(dot)us>
Date       : Oct 7, 2013
Description: Fund Analyzer for Computational Investing HW3
"""
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import sys


def read_values(valuesfile):
    """Read values to a Panda DataFrame"""

    li_cols = [0, 1, 2, 3]
    ls_names = ['YEAR', 'MONTH', 'DAY', 'TOTAL']
    d_date_columns = { 'DATE': ['YEAR', 'MONTH', 'DAY']}
    s_index_column = 'DATE'
    df_values = pd.read_csv(valuesfile, \
                            dtype={'TOTAL': np.float64}, \
                            sep=',', \
                            comment='#', \
                            skipinitialspace=True, \
                            header=None, \
                            usecols=li_cols, \
                            names=ls_names, \
                            parse_dates=d_date_columns, \
                            index_col=s_index_column)
    if not df_values.index.is_monotonic:
        df_values.sort_index(inplace=True)

    return df_values


def main():
    """Main Function"""

    print "\nProcessing %s...\n" % s_values_filename

    # Reading the values
    df_values = read_values(s_values_filename)

    # Extract just the data (as floating point values)
    na_values = df_values.values * 1.0

    # Normalize
    na_norm_values = na_values / na_values[0, :]

    # Returnize the values
    na_returns = na_norm_values.copy()
    tsu.returnize0(na_returns)

    # Calculate statistical values
    f_avg_return = np.mean(na_returns)
    f_tot_return = np.prod(na_returns + 1.0)
    f_stdev = np.std(na_returns)
    f_sharpe = math.sqrt(252.0) * f_avg_return / f_stdev

    # Reading the historical data.
    dt_start = df_values.index[0]
    dt_end = df_values.index[len(df_values.index) - 1]

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end + dt.timedelta(days=1), dt_timeofday)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Prepare the symbols to fetch from data source
    ls_symbols = [ s_benchmark_symbol ]

    # Create an object of DataAccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Copying the close price into separate DataFrame
    df_close = d_data['close'].copy()

    # Filling the gaps in data.
    df_close[s_benchmark_symbol] = df_close[s_benchmark_symbol].fillna(method='ffill')
    df_close[s_benchmark_symbol] = df_close[s_benchmark_symbol].fillna(method='bfill')
    df_close[s_benchmark_symbol] = df_close[s_benchmark_symbol].fillna(1.0)

    # Extract just the data
    na_benchmark_close = df_close.values

    # Normalize the price
    na_norm_benchmark_close = na_benchmark_close / na_benchmark_close[0, :]

    # Returnize the values
    na_benchmark_returns = na_norm_benchmark_close.copy()
    tsu.returnize0(na_benchmark_returns)

    # Calculate statistical values
    f_benchmark_avg_return = np.mean(na_benchmark_returns)
    f_benchmark_tot_return = np.prod(na_benchmark_returns + 1.0)
    f_benchmark_stdev = np.std(na_benchmark_returns)
    f_benchmark_sharpe = math.sqrt(252.0) * f_benchmark_avg_return / f_benchmark_stdev

    print 'The final value of the portfolio using the sample file is -- %d,%d,%d,%.0f' % \
          (dt_end.year, dt_end.month, dt_end.day, df_values['TOTAL'].ix[dt_end])
    print
    print 'Details of the Performance of the portfolio'
    print
    print 'Date Range : ', min(ldt_timestamps), ' to ', max(ldt_timestamps)
    print
    print 'Sharpe Ratio of Fund : %-.12g' % f_sharpe
    print 'Sharpe Ratio of %s : %-.12g' % (s_benchmark_symbol, f_benchmark_sharpe)
    print
    print 'Total Return of Fund :  %-.12g' % f_tot_return
    print 'Total Return of %s : %-.12g' % (s_benchmark_symbol, f_benchmark_tot_return)
    print
    print 'Standard Deviation of Fund :  %-.12g' % f_stdev
    print 'Standard Deviation of %s : %-.12g' % (s_benchmark_symbol, f_benchmark_stdev)
    print
    print 'Average Daily Return of Fund :  %-.12g' % f_avg_return
    print 'Average Daily Return of %s : %-.12g' % (s_benchmark_symbol, f_benchmark_avg_return)

    # Plotting the prices with x-axis=timestamps
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(ldt_timestamps, na_values)
    plt.plot(ldt_timestamps, na_norm_benchmark_close * na_values[0], alpha=0.4)
    plt.legend(['Fund', s_benchmark_symbol], loc='best')
    plt.ylabel('Value')
    plt.xlabel('Date')
    fig.autofmt_xdate(rotation=45)
    s_pdf_filename = s_values_filename[ : s_values_filename.rfind('.')] + '.pdf'
    plt.savefig(s_pdf_filename, format='pdf')

    # Plot again using normalized values
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(ldt_timestamps, na_norm_values)
    plt.plot(ldt_timestamps, na_norm_benchmark_close, alpha=0.4)
    plt.legend(['Fund', s_benchmark_symbol], loc='best')
    plt.ylabel('Normalized Value')
    plt.xlabel('Date')
    fig.autofmt_xdate(rotation=45)
    s_pdf_filename = s_values_filename[ : s_values_filename.rfind('.')] + '_normalized.pdf'
    plt.savefig(s_pdf_filename, format='pdf')




if __name__ == '__main__':
    if len(sys.argv) == 1:
        s_values_filename = 'values.csv'
        s_benchmark_symbol = '$SPX'
        main()
        s_values_filename = 'values2.csv'
        s_benchmark_symbol = '$SPX'
        main()
    elif len(sys.argv) == 3:
        s_values_filename = sys.argv[1]
        s_benchmark_symbol = sys.argv[2]
        main()
    else:
        print "\nUsage: python analyze.py <values.csv file> <benchmark>\n"