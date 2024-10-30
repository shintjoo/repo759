import pandas as pd
import matplotlib.pyplot as plt

# Load data from the text files
static_times = pd.read_csv('zhuHW4T4static.txt', header=None).squeeze("columns")
dynamic_times = pd.read_csv('zhuHW4T4dynamic.txt', header=None).squeeze("columns")
guided_times = pd.read_csv('zhuHW4T4guided.txt', header=None).squeeze("columns")

# Define the x-axis values (1 through 8)
x_values = range(1, 9)

# Plot the data
plt.plot(x_values, static_times, label='Static', marker='o', linestyle='-')
plt.plot(x_values, dynamic_times, label='Dynamic', marker='o', linestyle='-')
plt.plot(x_values, guided_times, label='Guided', marker='o', linestyle='-')

# Set axis labels, title, and legend
plt.xlabel('Number of CPU Cores')
plt.ylabel('Time Taken (ms)')
plt.title('Execution Time when N = 200 vs. Number of CPU Cores')
plt.legend()

# show grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the plot as a PNG file
plt.savefig('task4.png')
plt.show()