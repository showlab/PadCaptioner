# Split a JSON list into N equal shards (padded with should_use=False copies).
# Usage: python split_data.py <data.json> <num_shards> <out_dir>
import json
import sys
import os
import copy

if __name__ == "__main__":
    data_file = sys.argv[1]
    split_num = sys.argv[2]
    target_dir = sys.argv[3]

    with open(data_file, "r") as f:
        data = json.load(f)

    per_len = len(data) // int(split_num)

    remainder = len(data) % int(split_num)

    if remainder > 0:
        should_add = int(split_num) - remainder
    else:
        should_add=0

    for i in range(should_add):
        data_append = copy.deepcopy(data[-1])
        data_append["should_use"] = False
        data.append(data_append)

    per_len = len(data) // int(split_num)

    group_counts = [per_len] * int(split_num)

    os.makedirs(target_dir, exist_ok=True)

    total_count = 0

    for i in range(int(split_num) - 1):
        data_file = os.path.join(target_dir, f"{i}.json")
        print(data_file)
        with open(data_file, "w") as f:
            json.dump(data[total_count:total_count+group_counts[i]], f, indent=2)
        print(group_counts[i])
        total_count += group_counts[i]

    data_file = os.path.join(target_dir, f"{int(split_num) - 1}.json")
    with open(data_file, "w") as f:
        json.dump(data[total_count:], f, indent=2)
