import sys

import parse_data

fname = "../data/order_brush_order.csv"

print('loading')
d_list = parse_data.parse_file(fname)

#print(str(d_list[0]))

print('segmenting')
segmented_list = parse_data.segment_data_by_shop(d_list)
#print(str(len(segmented_list)))

shop_list = []
for ele in d_list:
    shop_list.append(ele['shopid'])
shop_list = list(set(shop_list))
#print(str(len(shop_list)))
if len(shop_list) != len(segmented_list):
    print('ERROR LENGTH')
    sys.exit(0)
print('checked num shops: '+str(len(shop_list)))

sorted_shop_lists = parse_data.sort_all_shop_lists(segmented_list)
print('sorted lists')

n_brush = 0
for s_id in shop_list:
    per_shop_list = segmented_list[s_id]
    already_sorted = sorted_shop_lists[s_id]
    if len(per_shop_list) > 1:
        s_sorted = parse_data.sort_shop_list(per_shop_list)
        if s_sorted != already_sorted:
            print('ERROR')
print('checked sorting works')
res = parse_data.check_all_sorted(sorted_shop_lists)
if res:
    print('all sorted')
else:
    print('sort problem')
    sys.exit(0)

print('computing concentrate rates')

#for s_id in shop_list:
    #already_sorted = sorted_shop_lists[s_id]
    #does_brush = parse_data.does_shop_brush(already_sorted)
    #if does_brush:
        #n_brush += 1
shops_that_brush = parse_data.get_shops_that_brush(sorted_shop_lists)
#if n_brush != len(shops_that_brush):
    #print("brush count mismatch")
    #sys.exit(0)
print('n_brush: '+str(len(shops_that_brush)))

all_info = parse_data.get_all_instances(sorted_shop_lists)

d = parse_data.analyze_instances(all_info)
parse_data.check_dups_by_shop(sorted_shop_lists)
sys.exit(0)

for shopid in d:
    res = d[shopid]
    res_sorted = sorted(res)
    res_s = [str(x) for x in res_sorted]
    if res == []:
        print(str(shopid)+','+'0')
    else:
        print(str(shopid)+','+'&'.join(res_s))

