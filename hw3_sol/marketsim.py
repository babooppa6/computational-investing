"""
Author     : Tolunay Orkun <tolunay(at)orkun(dot)us>
Date       : Oct 7, 2013
Description: Market Simulator for Computational Investing HW3
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


def read_orders(orderfile):
    """Read orders in orderfile to a Panda DataFrame"""

    ls_names = ['YEAR', 'MONTH', 'DAY', 'SYMBOL', 'TYPE', 'QUANTITY']
    li_cols = [0, 1, 2, 3, 4, 5]
    d_date_columns = {'DATE': ['YEAR', 'MONTH', 'DAY']}
    s_index_column = 'DATE'
    df_orders = pd.read_csv(orderfile, \
                            sep=',', \
                            comment='#', \
                            skipinitialspace=True, \
                            header=None, \
                            usecols=li_cols, \
                            names=ls_names, \
                            parse_dates=d_date_columns, \
                            index_col=s_index_column)
    df_orders.index = df_orders.index + dt.timedelta(hours=16)
    if not df_orders.index.is_monotonic:
        df_orders.sort_index(inplace=True)

    # Uppercase string data
    df_orders['SYMBOL'] = df_orders['SYMBOL'].str.upper()
    df_orders['TYPE'] = df_orders['TYPE'].str.upper()

    # Adjust quantity based on order type
    na_buy_orders = df_orders['TYPE'].values == 'BUY'
    na_sell_orders = df_orders['TYPE'].values == 'SELL'
    na_valid_orders = np.logical_or(na_buy_orders, na_sell_orders)
    na_invalid_orders = np.logical_not(na_valid_orders)
    df_orders['QUANTITY'][na_sell_orders] = df_orders['QUANTITY'].values * -1
    df_orders['QUANTITY'][na_invalid_orders] = 0

    # No longer need order type column
    del df_orders['TYPE']

    return df_orders


def get_trades(ldt_timestamps, ls_symbols, df_orders):
    """Create a trade matrix for each day"""

    # Create a DataFrame for trades
    df_trades = pd.DataFrame(index=ldt_timestamps, columns=ls_symbols)

    # Initialize all data to zeros
    df_trades[:] = np.zeros((len(df_trades.index), len(df_trades.columns)), dtype=np.int32)

    for i in range(len(df_orders.index)):
        na_order = df_orders.iloc[i].values
        s_sym = na_order[0]
        i_qty = na_order[1]
        df_trades[s_sym].loc[df_orders.index[i]] += i_qty

    return df_trades


def augment_trades_with_cash(df_trades, df_price):
    """Calculate daily cash change"""

    df_trades['cash'] = 0.0
    df_trades['cash'] = np.sum(df_trades.values * df_price.values, axis=1) * -1.0


def get_holdings(f_start_cash, df_trades):
    """Transform daily trades matrix to daily holdings"""

    df_holdings = df_trades.copy()
    df_holdings[:] = np.cumsum(df_holdings.values, axis=0)
    df_holdings['cash'] += f_start_cash
    return df_holdings


def get_values(df_holdings, df_price):
    """Sum each row of portfolio to determine daily values"""

    df_values = pd.DataFrame(index=df_holdings.index)
    df_values['value'] = np.sum(df_holdings.values * df_price.values, axis=1)
    return df_values


def print_holdings(dt_date, df_port):
    """Print portfolio holdings on a give date"""

    print 'Date:\t', dt_date
    for s_sym in df_port.columns:
        print s_sym.upper() + ":\t", df_port[s_sym].ix[dt_date]


def write_values(valuesfile, df_values):
    """Dump the portfolio daily totals to a CSV file"""

    print "Generating %s..." % s_values_filename
    df_output = pd.DataFrame(index=range(len(df_values)), columns=['YEAR', 'MONTH', 'DAY', 'VALUE'])
    df_output['YEAR'] = [dt_date.year for dt_date in df_values.index]
    df_output['MONTH'] = [dt_date.month for dt_date in df_values.index]
    df_output['DAY'] = [dt_date.day for dt_date in df_values.index]
    df_output['VALUE'] = list(df_values['value'].values)
    #df_output.to_csv(valuesfile, header=False, index=False, float_format='%.0f')
    df_output.to_csv(valuesfile, header=False, index=False)


def main():
    """Main Function"""
    print "\nProcessing %s...\n" % s_orders_filename

    # Reading the orders
    df_orders = read_orders(s_orders_filename)

    # Get a list of symbols we are going to trade
    ls_symbols = list(set(df_orders['SYMBOL'].values))
    ls_symbols.sort()
    print "SYMBOLS   : ", ls_symbols

    # Create an object of DataAccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')
    ls_all_symbols = c_dataobj.get_all_symbols()

    # Bad symbols are symbols present in portfolio but not in all symbols
    ls_bad_symbols = list(set(ls_symbols) - set(ls_all_symbols))
    if len(ls_bad_symbols) != 0:
        print "Orders contain bad symbols : ", ls_bad_symbols
        return

    # Reading the historical data.
    dt_start = df_orders.index.min()
    print "START DATE: ", dt_start
    dt_end = df_orders.index.max()
    print "END DATE  : ", dt_end
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
    print "DAYS      : ", len(ldt_timestamps)

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Copying the close price into separate DataFrame
    df_close = d_data['close'].copy()

    # Filling the gaps in data.
    for s_symbol in ls_symbols:
        df_close[s_symbol] = df_close[s_symbol].fillna(method='ffill')
        df_close[s_symbol] = df_close[s_symbol].fillna(method='bfill')
        df_close[s_symbol] = df_close[s_symbol].fillna(1.0)

    # Add in the price for cash
    df_close['cash'] = 1.0

    # Build a trade matrix
    df_trades = get_trades(ldt_timestamps, ls_symbols, df_orders)
    augment_trades_with_cash(df_trades, df_close)

    # Transform trades to a holdings matrix
    df_holdings = get_holdings(f_start_cash, df_trades)

    # Get daily portfolio values
    df_values = get_values(df_holdings, df_close)

    # Print out final portfolio
    print "\nFinal Portfolio:"
    print_holdings(dt_end, df_holdings)

    # Print out some statistics
    f_port_final = round(df_values['value'].loc[dt_end])
    f_port_start = round(df_values['value'].loc[dt_start])
    f_total_return = f_port_final / f_port_start
    print "Total Return: ", f_total_return
    print

    # Write out portfolio values
    write_values(s_values_filename, df_values)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        f_start_cash = 1000000.0
        s_orders_filename = 'orders.csv'
        s_values_filename = 'values.csv'
        main()
        f_start_cash = 1000000.0
        s_orders_filename = 'orders2.csv'
        s_values_filename = 'values2.csv'
        main()
    elif len(sys.argv) == 4:
        try:
            f_start_cash = float(int(sys.argv[1]))
        except ValueError:
            f_start_cash = float(sys.argv[1])
        s_orders_filename = sys.argv[2]
        s_values_filename = sys.argv[3]
        main()
    else:
        print "\nUsage: python marketsim.py <starting cash> <orders.csv file> <values.csv file>\n"
