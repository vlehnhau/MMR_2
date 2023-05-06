import numpy as np
import matplotlib.pyplot as plt

############################################
# Aufgabe 1.1

datafile = np.loadtxt("data.txt", skiprows=3)

tx_val = datafile[:, 6]                 # Erdboden
rr_val = datafile[:, 12]                # Niederschlagsmenge

# tx_val = tx_val[100:300]
# rr_val = rr_val[100:300]


x = np.arange(start=0, stop=len(tx_val), step=1)

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
# Aufgabe 1.2 toDo: Aufgabe 1.2 bearbeiten

############################################
# Aufgabe 1.3 toDo: Aufgabe 1.3 bearbeiten

############################################
# Aufgabe 1.4 toDo: Aufgabe 1.4 bearbeiten

############################################
# Aufgabe 1.5 toDo: Aufgabe 1.5 bearbeiten

