import json
from pprint import pprint
json_path = "/Users/htplex/Desktop/data_new/datasets/PIDray/annotations/test.json"
with open(json_path, 'r') as fp:
    data = json.load(fp)
    
pprint(data.keys())
pprint(len(data['images']))
pprint(len(data['annotations']))
pprint(data['categories'])
data['annotations'] = []
data['images'] = data['images'][:10]
pprint(data)

with open("/Users/htplex/Desktop/data_new/datasets/PIDray/annotations/test_one.json", 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4, ensure_ascii=False)