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

# toDo: Aufgabe 1.2 Theoriefragen

############################################
# Aufgabe 1.3 toDo: Aufgabe 1.3 bearbeiten

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



############################################
# Aufgabe 1.4 toDo: Aufgabe 1.4 bearbeiten

############################################
# Aufgabe 1.5 toDo: Aufgabe 1.5 bearbeiten

# def plot_1_4(x_data, y_data, m):
#     my = []
#     m2 = m
#     for i in range(len(x_data)):
#         arg = y_data[i:m2]
#         m2 += 1
#         my.append(???)
#
#     plt.plot(x_data, my, color='orange')
#     plt.scatter(x_data, y_data, color='blue')
#     plt.show()

# numpy kann gleichungssysteme lösen (polyfit)

############################################
# Aufgabe 2: toDo: Aufgabe 2 bearbeiten

if __name__ == '__main__':

    # Aufgabe 1.1:
    datafile = readfile("data.txt")

    tx_val = datafile[:, 6]             # Erdboden
    rr_val = datafile[:, 12]            # Niederschlagsmenge

    x = np.arange(start=0, stop=len(tx_val), step=1)

    plot_1_1(x, tx_val, rr_val)

    # Aufgabe 1.2:

    # Gegebene Datenpunkte
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])        # x-Koordinaten der Datenpunkte
    y_data = np.array([13.0, 12.0, 18.0, 12.0, 13.0])   # y-Koordinaten der Datenpunkte

    plot_1_2(x_data, y_data)
    # plot_1_2(x, rr_val)

    # Aufgabe 1.3:

    plot_1_3a(x, tx_val)
    plot_1_3b(x, tx_val, 10)
    plot_1_3c(x, tx_val, 10)