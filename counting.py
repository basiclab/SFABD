import json

def iou(time1, time2):
    union = max(time1[1], time2[1]) - min(time1[0], time2[0])
    intersection = min(time1[1], time2[1]) - max(time1[0], time2[0])
    return round(intersection / union, 2)

with open('data/QVHighlights/train.json') as f:
    data = json.load(f)

count = 0
target_count = 0
for idx, (vid, anno) in enumerate(data.items()):
    duration = anno["duration"]
    unit_time = duration/64
    for query_label in anno["annotations"]:
        timestamps = query_label["timestamps"]
        for time in timestamps:
            length = time[1] - time[0]
            if length <= 2:
                target_count += 1
                ## check if moment is aligned with grid unit
                start = time[0]
                start_index = int(round(start / unit_time, 0))
                closest_grid = [start_index*unit_time, (start_index+1)*unit_time]
                gt_iou = iou(time, closest_grid)
                print(f"GT:{time}, closest grid:{closest_grid}, iou: {gt_iou}")
                if gt_iou >= 0.75:
                    count += 1
                


print(f"count: {count}/{target_count}, {round(count/target_count*100, 2)}%")