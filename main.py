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

    # Berechnung der Lagrange-Polynome l_i(x)
    def lagrange_basis(x_data, i, x):
        result = np.ones_like(x)
        for j in range(len(x_data)):
            if j != i:
                result *= (x - x_data[j]) / (x_data[i] - x_data[j])
        return result

    # Berechnung des Polynoms p(x)
    def polynomial_interpolation(x_data, y_data, x):
        n = len(x_data)
        p = np.zeros_like(x)
        for i in range(n):
            p += y_data[i] * lagrange_basis(x_data, i, x)
        return p

    # Auswertung des Polynoms an den echten Messstellen
    y_interpolated = polynomial_interpolation(x_data, y_data, x)

    # Plot der Interpolation und der echten Datenpunkte
    plt.plot(x, y_interpolated, label='Interpolation')
    plt.scatter(x_data, y_data, color='red', label='Echte Daten')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynom-Interpolation')
    plt.show()

# toDo: Aufgabe 1.2 Theoriefragen

############################################
# Aufgabe 1.3 toDo: Aufgabe 1.3 bearbeiten

############################################
# Aufgabe 1.4 toDo: Aufgabe 1.4 bearbeiten

############################################
# Aufgabe 1.5 toDo: Aufgabe 1.5 bearbeiten


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