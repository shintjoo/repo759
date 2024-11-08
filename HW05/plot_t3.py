import matplotlib.pyplot as plt

# Array sizes from 2^10 to 2^29
sizes = [2**exp for exp in range(10, 30)]

# Read execution times from output files
times_512 = []
times_16 = []

# Parse the output files for time data
with open("zhuHW5T3.out", "r") as file_512, open("zhuHW5T316.out", "r") as file_16:
    for line in file_512:
        # Extract the numeric time part from "untime: <time>"
        time_str = line.strip().split(":")[-1]
        times_512.append(float(time_str))

    for line in file_16:
        # Extract the numeric time part from "untime: <time>"
        time_str = line.strip().split(":")[-1]
        times_16.append(float(time_str))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_512, label="512 Threads per Block", marker='o')
plt.plot(sizes, times_16, label="16 Threads per Block", marker='s')
plt.xlabel("Array Size (n)")
plt.ylabel("Execution Time (ms)")
plt.title("CUDA vscale Kernel Execution Time vs Array Size")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("task3.png")
plt.show()