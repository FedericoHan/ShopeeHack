import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

file = 'order_brush_order.csv'
df = pd.read_csv(file)
df = df.sort_values('orderid')
#need to set index to datetime obj to bucket by time
df["event_time"] = pd.to_datetime(df['event_time'])
df = df.set_index(pd.DatetimeIndex(df['event_time'])).drop('event_time', axis=1).sort_index()

MAGIC_NUMBER = 3.0

# return unique shoplist
def get_shop_list(full_data):
    shop_list = full_data['shopid']
    shop_list = pd.Series(shop_list.unique())
    return shop_list

# return dictionary for shopid
def get_shop_dict(data, shopid_in):
    shop_dict = data[(data['shopid'] == shopid_in)]
    return shop_dict

# return hour chunk from starttime.
# if less than MAGIC_NUMBER return same dict cause it ain't brushing
def get_hr_dict(shop_dict_in, starttime):
    if len(shop_dict_in) < MAGIC_NUMBER:
        return shop_dict_in
    hr_dict = shop_dict_in
    #selection by timedate range
    start = datetime.fromisoformat(starttime)
    end = start + timedelta(hours=1)
    return hr_dict[start.isoformat():end.isoformat()]

# return concentration rate
def get_concentration_rate(hr_window):
    rate = 0.0
    users = hr_window['userid']
    rate = len(hr_window.index)/len(list(set(users)))
    return rate

# return string concat with & from given array
def concat_userid(data):
    if len(data) <= 0:
        return "0"
    result = '&'.join(str(x) for x in data)
    data
    return result

# return string of userid(s) during brushing or empty array if no brushing
def get_order_brush_userids(shopid_dict):
    userids = []
    if len(shopid_dict) < MAGIC_NUMBER:
        return concat_userid(userids)
    for i in shopid_dict.index:
        starttime = i.isoformat()
        this_hr = get_hr_dict(shopid_dict,starttime)
        endtime = this_hr.index[len(this_hr.index)-1].isoformat()
        #if not brushing go to next
        if get_concentration_rate(this_hr) < MAGIC_NUMBER:
            continue
        these_users = this_hr['userid']
        user_counts = these_users.value_counts()
        #add userids with highest count. more than one only if tied.
        userids.extend(user_counts[user_counts==user_counts.max()].index.tolist())
    userids = np.unique(userids)
    return concat_userid(userids)

# unique shopid list loop through and get dict for each shopid
shopid_list = get_shop_list(df)
brush_userids = []
for i in shopid_list:
    this_shop_dict = get_shop_dict(df,i)
    brush_userids.append(get_order_brush_userids(this_shop_dict))

submission = pd.DataFrame({'shopid': shopid_list, 'userid': brush_userids})
submission_sorted = submission.sort_values('userid')
submission_sorted.to_csv("submission.csv", index=False)
# order brushing shops should be 315?
