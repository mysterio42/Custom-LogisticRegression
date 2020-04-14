import argparse

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scalar', type=float,
                        help='scalar value to predict binary response')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_args()
    if arg.scalar:
        url = 'http://127.0.0.1:5000/classify'
        payload = {"scalar": arg.scalar}
        r = requests.post(url, json=payload)
        print(r.json())