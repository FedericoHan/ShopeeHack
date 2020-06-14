
import csv
import datetime

def parse_file(fname):
    d_list = []

    with open(fname, 'r') as f:
        r = csv.reader(f)
        headerDone = False
        for row in r:
            if not headerDone:
                headerDone = True
                continue
            dt = datetime.datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
            if dt.year != 2019:
                print("ERROR Y")
            if dt.month != 12:
                print("ERROR M")
            if dt.day < 27:
                print("ERROR D")
            d = {
                'orderid' : int(row[0]),
                'shopid' : int(row[1]),
                'userid' : int(row[2]),
                'event_time' : dt
            }
#            print(str(d['event_time']))
            d_list.append(d)

    return d_list

def check_dups(order_list):
    for i in range(len(order_list)):
        for j in range(i+1, len(order_list)):
            if order_list[i]['orderid'] == order_list[j]['orderid']:
                print("DUP ORDER")

def check_dups_by_shop(shop_list):
    for shopid in shop_list:
        order_list = shop_list[shopid]
        for i in range(len(order_list)):
            for j in range(i+1, len(order_list)):
                if order_list[i]['orderid'] == order_list[j]['orderid']:
                    print("DUP ORDER")

def segment_data_by_shop(full_data):
    segment_d = {}
    for ele in full_data:
        s_id = ele['shopid']
        if s_id not in segment_d:
            segment_d[s_id] = []
        segment_d[s_id].append(ele)
    return segment_d

def shop_ts_key(record):
    return record['event_time']

# this gets a list of transactions for a single shop into timestamp order
def sort_shop_list(shop_list):
    s = sorted(shop_list, key=shop_ts_key)
    return s

def sort_all_shop_lists(big_list):
    d = {}
    for shopid in big_list:
        sorted_list = sort_shop_list(big_list[shopid])
        d[shopid] = sorted_list
    return d

# takes in an ordered shop_list
# computes the highest observed concentrate rate
def max_concentrate_rate(shop_list):
    n_orders = len(shop_list)
    res_d = {
        'max rate' : -1,
        'instances' : []
    }
    for start_index in range(n_orders):
        start_time = shop_list[start_index]['event_time']
        buyers = []
        orders_in_range = []
        last_seconds = 0
        this_c_rate = 0.0
        for this_idx in range(start_index, n_orders):
            this_time = shop_list[this_idx]['event_time']
            dt = this_time - start_time
            seconds = dt.total_seconds()
            if seconds < last_seconds:
                print('ERROR ERROR ERROR')
                print('this ts: '+str(this_time))
                print('start ts: '+str(start_time))
            last_seconds = seconds
            if seconds < 3600.0:
                buyers.append(shop_list[this_idx]['userid'])
                orders_in_range.append(shop_list[this_idx])
                candidate_rate = len(orders_in_range)*1.0/len(list(set(buyers)))
                this_c_rate = max(candidate_rate, this_c_rate)
            else:
                # break out of loop
                continue
        if this_c_rate > 3.0:
            res_d['instances'].append(orders_in_range)
        if this_c_rate > res_d['max rate']:
            res_d['max rate'] = this_c_rate
    return res_d

def get_all_instances(big_dict):
    res = {}

    for shopid in big_dict:
        res[shopid] = max_concentrate_rate(big_dict[shopid])

    return res

def does_shop_brush(shop_list):
    c_rate_d = max_concentrate_rate(shop_list)
    if c_rate_d["max rate"] >= 3.0:
        return True
    return False

def get_shops_that_brush(big_dict):
    brush_list = []

    for shopid in big_dict:
        ele = big_dict[shopid]
        if does_shop_brush(ele):
            brush_list.append(shopid)

    return brush_list

def check_sorted(shop_list):
    prev_dt = datetime.datetime(2000,1,1)
    for ele in shop_list:
        this_dt = ele['event_time']
        if this_dt < prev_dt:
            return False
        prev_dt = this_dt
    return True

def check_all_sorted(big_dict):
    for shopid in big_dict:
        res = check_sorted(big_dict[shopid])
        if not res:
            return False
    return True

def analyze_instances(res_d):
    brushers_d = {}
    for shopid in res_d:
        brushers_d[shopid] = []
        res = res_d[shopid]
        instances = res["instances"]
        per_user_orders = {}

        all_orders = []
        for instance in instances:
            for order in instance:
                all_orders.append(order)

        uniq_orders = []
        for i in range(len(all_orders)):
            has_future_entry = False
            for j in range(i+1, len(all_orders)):
                if all_orders[i]['event_time'] == all_orders[j]['event_time']:
                    if all_orders[i]['userid'] == all_orders[j]['userid']:
                        has_future_entry = True
            if not has_future_entry:
                uniq_orders.append(all_orders[i])

#        for instance in instances:
#            for order in instance:
#                uid = order['userid']
#                if uid not in per_user_orders:
#                    per_user_orders[uid] = 0.0
#                per_user_orders[uid] += 1.0
        for order in uniq_orders:
            uid = order['userid']
            if uid not in per_user_orders:
                per_user_orders[uid] = 0.0
            per_user_orders[uid] += 1.0
        max_orders = 0
        for uid in per_user_orders:
            if per_user_orders[uid] > max_orders:
                max_orders = per_user_orders[uid]
        for uid in per_user_orders:
            if max_orders == per_user_orders[uid]:
                brushers_d[shopid].append(uid)
    return brushers_d
