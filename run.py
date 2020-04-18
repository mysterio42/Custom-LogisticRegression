import argparse

import numpy as np

from utils.data import generate_data
from utils.model import train_model, latest_modified_weight, load_model
from utils.plot import model_plot, prediction_plot,data_plot


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(0)
    args = parse_args()

    x1, y1, x2, y2 = generate_data()

    if args.load:
        weight = latest_modified_weight()
        model = load_model(weight)
        data_plot(x1, y1, x2, y2)
        model_plot(model, x1, y1, x2, y2)
        prediction_plot(model, x1, y1, x2, y2, 5.53)
        prediction_plot(model, x1, y1, x2, y2, 3.42)
        prediction_plot(model, x1, y1, x2, y2, 4.3)
    else:
        X = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        model = train_model(X, y)
        model_plot(model, x1, y1, x2, y2)
        prediction_plot(model, x1, y1, x2, y2, 5.53)
        prediction_plot(model, x1, y1, x2, y2, 3.42)
        prediction_plot(model, x1, y1, x2, y2, 4.3)
