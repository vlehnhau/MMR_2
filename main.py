import numpy as np
import matplotlib.pyplot as plt
import sympy as sympy

############################################
# Aufgabe 1.1

def readfile(filename):
    return np.loadtxt(filename, skiprows=3)

def plot_1_1(x, tx_val, rr_val):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Zeitpunkt')
    ax1.set_ylabel('Temp', color=color)
    ax1.plot(x, tx_val, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Niederschlag', color=color)
    ax2.plot(x, rr_val, 'b.')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.show()

# toDo: Fragen beantworten und Punkte verbinden(kp was mit punkte verbinden gemeint ist ... sind sie doch schon)

############################################
# Aufgabe 1.2

def plot_1_2(x_data, y_data):

    # Vektor x mit allen Stellen, an denen das Polynom ausgewertet werden soll
    x = np.linspace(min(x_data), max(x_data), 100)

    # Berechnung des Lagrange Polynoms
    def lagrange_polynom(x_data, i, x):
        result = 1
        for j in range(len(x_data)):
            if j != i:
                result *= (x - x_data[j]) / (x_data[i] - x_data[j])
        return result

    # Berechnung des Polynoms p(x)
    def polynomial_interpolation(x_data, y_data, x):
        p = 0
        for i in range(len(x_data)):
            p += y_data[i] * lagrange_polynom(x_data, i, x)
        return p

    # Auswertung des Polynoms an den echten Messstellen
    y_interpolated = polynomial_interpolation(x_data, y_data, x)

    # Plot der Interpolation und der echten Datenpunkte
    plt.plot(x, y_interpolated, color='orange', label='Interpolation')
    plt.scatter(x_data, y_data, color='blue', label='Echte Daten')

    # Plot der Lagrange-Polynome für jeden Datenpunkt
    for i in range(len(x_data)):
        l_i = lagrange_polynom(x_data, i, x)
        plt.plot(x, y_data[i] * l_i, color='orange', alpha=0.2)

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynom-Interpolation')
    plt.show()

    return y_data

# toDo: Aufgabe 1.2 Theoriefragen

############################################
# Aufgabe 1.3

def plot_1_3a(x_data, y_data):

    m = np.mean(y_data)
    my = np.full(len(y_data), m)

    plt.plot(x_data, my, color='orange')
    plt.scatter(x_data, y_data, color='blue')
    plt.show()


def plot_1_3b(x_data, y_data, m):
    mx = x_data
    for x in x_data:
        if x % m != 0:
            mx = np.delete(mx, np.where(mx == x)[0])

    my = []
    for x in mx:
        arg = y_data[x:x+m]
        my.append(np.mean(arg))

    for i in range(len(mx)):
        mx[i] += int(m/2)

    plt.plot(mx, my, color='orange')
    plt.scatter(x_data, y_data, color='blue')
    plt.show()

def plot_1_3c(x_data, y_data, m):
    my = []
    m2 = m
    for i in range(len(x_data)):
        arg = y_data[i:m2]
        m2 += 1
        my.append(np.mean(arg))

    plt.plot(x_data, my, color='orange')
    plt.scatter(x_data, y_data, color='blue')
    plt.show()

# toDo: Theoriefragen

############################################
# Aufgabe 1.4

def plot_1_4(x, y):
    n = len(x)

    m = (n * np.mean(x) * np.mean(y) - np.sum(x * y)) / (n * np.mean(x) ** 2 - np.sum(x ** 2))
    b = np.mean(y) - m * np.mean(x)

    plt.plot(x, m * x + b, 'r')
    plt.plot(x, y, 'b.')

    plt.show()


def solve(x, y):
    n = len(x)
    x2 = x ** 2
    x4 = x ** 4
    x6 = x ** 6
    xy = x * y

    sum_x6 = np.sum(x6)
    sum_x4 = np.sum(x4)
    sum_x2 = np.sum(x2)
    sum_xy = np.sum(xy)
    sum_y = np.sum(y)

    A = np.array([[sum_x6, sum_x4, sum_x2], [sum_x4, sum_x2, n], [sum_x2, n, 0]])
    b = np.array([sum_xy, sum_y, 0])

    a, b, c = np.linalg.solve(A, b)

    return a, b, c


def plot_1_4_deg_2(x, y):               # toDo: ^2 geht nicht
    a, b, c = solve(x, y)

    plt.plot(x, a * x ** 2 + b * x + c, 'r')
    plt.plot(x, y, 'b.')

    plt.show()


def plot_1_4_rdm_deg(x, y, deg):

    coeff = np.polyfit(x, y, deg)
    p = np.poly1d(coeff)
    plt.plot(x, p(x), 'r')
    plt.plot(x, y, 'b.')

    plt.show()

    return p

# toDo: Theoriefragen

############################################
# Aufgabe 1.5 toDo: Aufgabe 1.5 bearbeiten

def plot_1_5(x_data, y_data, m):
    coeff = np.polyfit(x_data, y_data, 100)
    p = np.poly1d(coeff)

    my = []
    m2 = m
    for i in range(len(x_data)):
        arg = p(x_data)[i:m2]
        my.append(np.mean(arg))
        m2 += 1

    plt.plot(x_data, my, color='orange')
    plt.scatter(x_data, y_data, color='blue')
    plt.show()

############################################
# Aufgabe 2: toDo: Aufgabe 2 bearbeiten

if __name__ == '__main__':

    # # Aufgabe 1.1:
    datafile = readfile("data.txt")

    tx_val = datafile[:, 6]             # Erdboden
    rr_val = datafile[:, 12]            # Niederschlagsmenge
    #
    # x = np.arange(start=0, stop=len(tx_val), step=1)
    #
    # plot_1_1(x, tx_val, rr_val)
    #
    # # Aufgabe 1.2:
    #
    # # Gegebene Datenpunkte
    # x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])        # x-Koordinaten der Datenpunkte
    # y_data = np.array([13.0, 12.0, 16.0, 12.0, 13.0])   # y-Koordinaten der Datenpunkte
    #
    # plot_1_2(x_data, y_data)
    #
    # # Aufgabe 1.3:
    #
    # plot_1_3a(x, tx_val)
    # plot_1_3b(x, tx_val, 10)
    # plot_1_3c(x, tx_val, 10)
    #
    # # Aufgabe 1.4:
    y_val = tx_val
    x_val = np.arange(0, len(y_val), 1)
    #
    # plot_1_4(x_val, y_val)
    # plot_1_4_deg_2(np.array([1,2,3,4,5,6]), np.array([10,12,30,4,15,60]))
    # plot_1_4_rdm_deg(np.array([1,2,3,4,5,6]), np.array([10,12,30,4,15,60]), 2)
    #
    # plot_1_4_rdm_deg(x_val, y_val, 100)

    # # Aufgabe 1.5
    # plot_1_5(x_val, y_val, 10)

    # Aufgabe 2:

    # Interpolation Data

    rdm_x_val = x_val.copy()
    rdm_x_val = np.random.permutation(rdm_x_val)

    rdm_x_val_split=np.split(rdm_x_val, 2)

    x_val_rdm_train = rdm_x_val_split[0]
    x_val_rdm_test = rdm_x_val_split[1]

    x_val_rdm_train.sort()
    x_val_rdm_test.sort()

    y_val_rdm_train = []
    y_val_rdm_test = []

    for i in x_val_rdm_train:
        y_val_rdm_train.append(y_val[i])

    for i in x_val_rdm_test:
        y_val_rdm_test.append(y_val[i])

    # Extrapolation Data
    x_val_split = np.split(x_val, 2)
    y_val_split = np.split(y_val, 2)

    x_val_train = x_val_split[0]
    x_val_test = x_val_split[1]

    y_val_train = y_val_split[0]
    y_val_test = y_val_split[1]

    # test Regression

    # interpolation
    p_interpolation = plot_1_4_rdm_deg(x_val_rdm_train, y_val_rdm_train, 10)

    fehler_interpolation = 0
    for i in x_val_rdm_test:
        fehler_interpolation += (p_interpolation(i) - y_val[i]) ** 2
    fehler_interpolation = (1/len(x_val)) * fehler_interpolation

    print(str(fehler_interpolation))

    # extrapolation
    p_extrapolation = plot_1_4_rdm_deg(x_val_train, y_val_train, 10)

    fehler_extrapolation = 0
    for i in x_val_test:
        fehler_extrapolation += (p_extrapolation(i) - y_val[i]) ** 2
    fehler_extrapolation = (1/len(x_val)) * fehler_extrapolation


    print(str(fehler_extrapolation))

    # test Polynominterpolation

    # interpolation
    # f_in = plot_1_2(x_val_rdm_train, y_val_rdm_train)
    #
    # fehler_interpolation_2 = 0
    # for i in x_val_rdm_test:
    #     fehler_interpolation_2 += (f_in[i] - y_val[i]) ** 2
    # fehler_interpolation_2 = (1 / len(x_val)) * fehler_interpolation_2
    #
    # print(str(fehler_interpolation_2))

    # extrapolation
    # f_ex = plot_1_2(x_val_train, y_val_train)
    #
    # fehler_extrapolation_2 = 0
    # for i in x_val_test:
    #     fehler_extrapolation_2 += (f_ex[i] - y_val[i]) ** 2
    # fehler_extrapolation_2 = (1 / len(x_val)) * fehler_extrapolation_2
    #
    # print(str(fehler_extrapolation_2))

    # loop interpolation regression test
    p_interpolation = []

    for i in range(10):
        rdm_x_val = x_val.copy()
        rdm_x_val = np.random.permutation(rdm_x_val)

        rdm_x_val_split = np.split(rdm_x_val, 2)

        x_val_rdm_train = rdm_x_val_split[0]
        x_val_rdm_test = rdm_x_val_split[1]

        x_val_rdm_train.sort()
        x_val_rdm_test.sort()

        y_val_rdm_train = []
        y_val_rdm_test = []

        for i in x_val_rdm_train:
            y_val_rdm_train.append(y_val[i])

        for i in x_val_rdm_test:
            y_val_rdm_test.append(y_val[i])

        p_interpolation.append(plot_1_4_rdm_deg(x_val_rdm_train, y_val_rdm_train, 10))

    p_interpolation_fehler = []

    for x in range(len(p_interpolation)):
        p_interpolation_fehler.append(0)
        for i in x_val_rdm_test:
            p_interpolation_fehler[x] += (p_interpolation[x](i) - y_val[i]) ** 2
        p_interpolation_fehler[x] = (1 / len(x_val)) * p_interpolation_fehler[x]

        print(str(p_interpolation_fehler[x]))

    print("Durchschnitt: " + str(np.mean(p_interpolation_fehler)))





