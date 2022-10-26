import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    data_vmr = json.load(open(args.input, 'r'))
    data_qv = []
    qid_cnt = 0
    with open(args.output, 'w') as f:
        for vid, data in data_vmr.items():
            for anno in data['annotations']:
                f.write(json.dumps({
                    "qid": qid_cnt,
                    "query": anno['query'],
                    "duration": data['duration'],
                    "vid": vid,
                    "relevant_windows": anno['timestamps'],
                }) + "\n")
                qid_cnt += 1
