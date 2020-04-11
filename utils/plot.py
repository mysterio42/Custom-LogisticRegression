from matplotlib import pyplot as plt

from utils.model import logistic


def model_plot(model, x1, y1, x2, y2):
    plt.title(f'Logistic Regression')
    plt.plot(x1, y1, 'ro', color='blue')
    plt.plot(x2, y2, 'ro', color='red')

    for i in range(25, 100):
        plt.plot(i / 10 - 2, logistic(model, i / 10.0 - 2), 'ro', color='green')

    plt.axis([-2, 10, -0.5, 2])
    plt.show()


def prediction_plot(model, x1, y1, x2, y2, x):
    plt.plot(x1, y1, 'ro', color='blue')
    plt.plot(x2, y2, 'ro', color='red')

    for i in range(25, 100):
        plt.plot(i / 10 - 2, logistic(model, i / 10.0 - 2), 'ro', color='green')

    plt.axis([-2, 10, -0.5, 2])
    ans = model.predict([[x]])[0]
    plt.plot(x, ans, 'ro', color='black')
    plt.title(f'Logistic Regression, Predict class for {x} is {ans}')
    plt.show()
