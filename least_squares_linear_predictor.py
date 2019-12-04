import numpy as np


def least_squares_linear_predictor_coefficients(data, order):
    """
    :param data: Data to do a least squares regression on
    :param order: order of the Least Squares regression
    :return: returns coefficients of LSR [a0, a1, a2... a_order-1]
    """
    X_low = np.ones((len(data) - order, order))
    for i in range(len(data) - order):
        X_low[i, 0:order] = data[i: i + order]
    x = np.ones((len(data) - order, 1))
    for i in range(order, len(data)):
        x[i - order] = data[i]
    coeffs = np.linalg.lstsq(-X_low, x, rcond=None)[0]
    coeffs = [coeffs[ind][0] for ind, val in enumerate(coeffs)]
    return np.asarray(coeffs)


def correlated_lse(data, order):
    num_data_sets = len(data)
    X_matrix = np.ones((num_data_sets, len(data[0]) - order, order))
    for index in range(num_data_sets):
        for i in range(len(data[index]) - order):
            X_matrix[index][i, 0:order] = data[index][i:i + order]
    final_X = np.concatenate((tuple(X_matrix[:])), axis=1)
    coeff_array = np.ones((num_data_sets, num_data_sets * order))
    for index in range(num_data_sets):
        x = np.ones(((len(data[index]) - order), 1))
        for i in range(order, len(data[0])):
            x[i - order] = data[index][i]
            coeffs = np.linalg.lstsq(-final_X, x, rcond=None)[0]
            coeffs = [coeffs[ind][0] for ind, val in enumerate(coeffs)]
            coeff_array[index] = coeffs
    output = []
    for i in range(num_data_sets):
        output.append(np.split(coeff_array[i], num_data_sets))
    return np.asarray(output)


def predict_next_point_array(data, order, coefficients):
    """
    function to predict the next point from a data set using only the coefficients of
    one member of the data set
    :param data: array of arrays (of the different variables)
    :param order: order of the filter
    :param coefficients: coefficients of the variable being predicted. Should be in an array
    :return: returns the prediction of the next point
    """
    prediction = 0
    for i in range(len(data)):
        points = np.asarray(data[i][-order:])
        prediction += np.sum(points * -coefficients[i])
    return prediction


def predict_x_next_points_array(data, order, coefficients, x):
    data = np.asarray(data)
    output = np.zeros((x,len(data)))
    for i in range(0, x):
        predicted_set = np.zeros((len(data),1))
        for index in range(len(coefficients)):
            predicted = predict_next_point_array(data, order, coefficients[index])
            predicted_set[index] = predicted
        output[i] = np.transpose(predicted_set)
        data = np.append(data, np.asarray(predicted_set), axis=1)
    return output


def least_squares_linear_predictor_coefficients_double(low_data, high_data, order):
    """
    :param data: Data to do a least squares regression on
    :param order: order of the Least Squares regression
    :return: returns coefficients of LSR [a0, a1, a2... a_order-1]
    """
    X_low = np.ones((len(low_data) - order, order))
    for i in range(len(low_data) - order):
        X_low[i, 0:order] = low_data[i: i + order]
    x_low = np.ones((len(low_data) - order, 1))
    for i in range(order, len(low_data)):
        x_low[i - order] = low_data[i]

    X_High = np.ones((len(high_data) - order, order))
    for i in range(len(high_data) - order):
        X_High[i, 0:order] = high_data[i: i + order]
    x_high = np.ones((len(high_data) - order, 1))
    for i in range(order, len(high_data)):
        x_high[i - order] = high_data[i]

    cat_arr = np.concatenate((X_low, X_High), axis=1)
    coeffs = np.linalg.inv(np.multiply(np.transpose(cat_arr), cat_arr)) * np.transpose(cat_arr)
    coeffs = np.linalg.lstsq(-X_low, x_low, rcond=None)[0]
    coeffs = [coeffs[ind][0] for ind, val in enumerate(coeffs)]
    return np.asarray(coeffs)


def predict_next_point(data, order, coefficients):
    points = np.asarray(data[-order:])
    return np.sum(points * -coefficients)


def predict_all_points(data, order, coefficients):
    """
    :param data: input data to create least squares prediction of order(order) of
    :param order: order for least squares prediction
    :param coefficients: coefficients of LPC
    :return: returns estimation of entire data set. Will be of length (len(data) - order)
    """
    predicted_set = np.zeros((1, len(data) - order))
    index = 0
    for i in np.arange(order, len(data)):
        y = data[i - order:i]
        predicted_set[0][index] = np.sum(np.multiply(data[i - order:i], -coefficients))
        index += 1
    return predicted_set[0]


def predict_x_next_points(data, order, coefficients, x):
    data = list(data)
    predicted_set = []
    for i in range(0, x):
        predicted = predict_next_point(data, order, coefficients)
        predicted_set.append(predicted)
        data.append(predicted)
    return predicted_set
