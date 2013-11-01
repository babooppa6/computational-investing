a ben 6 days ago New edits;
# Creating an empty dataframe
# df_events = copy.deepcopy(df_close)
df_events = df_close * np.NAN

## copy and deepcopy do nothing here.

now
for s_key in ls_keys:
d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
d_data[s_key] = d_data[s_key].fillna(1.0)

better
for s_key in ls_keys:
d_data[s_key] = d_data[s_key].fillna(method = 'ffill').fillna(method = 'bfill').fillna(1.0)

Remove excess Index lookups...
i_shares = 100
f_cutoff = 10.0
for s_sym in ls_symbols:
for i in range(1, len(ldt_timestamps)):
if df_close[s_sym].ix[ldt_timestamps[i]] < f_cutoff:
# Calculating the returns for this timestamp
f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1

# Event is found if the symbol is down more then 3% while the
# market is up more then 2%
# if f_symreturn_today <= -0.03 and f_marketreturn_today >= 0.02:
# df_events[s_sym].ix[ldt_timestamps[i]] = 1

if f_symprice_today < f_cutoff and f_symprice_yest >= f_cutoff:
df_events[s_sym].ix[ldt_timestamps[i]] = 1
row_to_enter = [str(ldt_timestamps[i].year), str(ldt_timestamps[i].month), \
str(ldt_timestamps[i].day), s_sym, 'Buy', i_shares]
writer.writerow(row_to_enter)
try:
time_n = ldt_timestamps[i + 5]
except:
time_n = ldt_timestamps[-1]
row_to_enter = [str(time_n.year), str(time_n.month), \
str(time_n.day), s_sym, 'Sell', i_shares]
writer.writerow(row_to_enter)

return df_events

To see the code with proper python indent use edit mode


------------------------------------------------------------------------------------------
Ben Bailey 3 days ago

In examining the SolutionToHomework4.py, within the find_events function, within the loop "for i in range(1,len(ldt_timestamps)):

I noticed that the SPY symbol is in the same loop as the symbol. So for each symbol, the values for SPY are being recalculated. E.g., if there are 200 symbols, then the f_marketprice_X values are being recalculated 200 times, instead of once.  This means the function will take twice as long to run as needed.

I've moved the assignment for f_marketprice_today, f_marketprice_yest, and f_marketreturn_today to be prior to the "for i in range..." section, so it is only calculated once.

Here's my code for the affected area:

##############CODE CHANGE START#############3
"""

Module: SolutionToHomework4.py

Section: def find_events(ls_symbols, d_data):


Original code was looping SPY for each unique symbol
"""
f_marketprice_today = None
f_marketprice_yest = None
f_marketreturn_today = None

 

for i in range(1,len(ldt_timestamps)):
    f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
    f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
    f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1


for s_sym in ls_symbols: #for each symbol in the list of symbols
    for i in range(1, len(ldt_timestamps)): #for each date within the rows of dates
        # Calculating the returns for this timestamp
        #for the date indexed by i, obtain the specified closing value for symbol
        f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
        f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
        #<deleted> f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
        #<deleted> f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
        #now calculat the returns
        f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
        #<deleted> f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) - 1
        i_shares = 1000

#########################CODE CHANGE END################

<remainder is same>
