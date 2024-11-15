import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


x = np.linspace(2, 7, 100)
y = 10*np.exp(-(x-4.5)**2/2**2)
exp_x = np.exp(x)
exp_y = np.exp(y)
xx = np.linspace(10, 1000, 100)
pdf = np.exp(10*np.exp(-(np.log(xx)-4.5)**2/2**2))


# Create a figure with two subplots: linear and log-log
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Regular plot
ax1.plot(xx, pdf)
ax1.set_title("Regular Plot")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Log-log plot
ax2.plot(xx, pdf)
#ax2.set_ylim([10000, 20000])
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("Log-Log Plot")
ax2.set_xlabel("log(x)")
ax2.set_ylabel("log(y)")

plt.tight_layout()
plt.show()

u = np.random.uniform(10, 1000, 10000)
pdf = np.exp(10*np.exp(-(np.log(u)-4.5)**2/2**2))

plt.hist(pdf, bins=100)
plt.xscale('log')
plt.yscale('log')

plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from ipywidgets import interact
# import ipywidgets as widgets

# # Function to plot the histograms with the given Pareto parameter
# def plot_pareto_histogram(a_value=3.0):
#     # Generate data based on the current shape parameter 'a'
#     data = np.random.pareto(a=a_value, size=1000)

#     # Create the figure and axis
#     plt.figure(figsize=(10, 5))

#     # Regular histogram
#     plt.subplot(1, 2, 1)
#     plt.hist(data, bins=50, color='blue', edgecolor='black', alpha=0.7)
#     plt.title('Regular Histogram')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')

#     # Log-log histogram
#     plt.subplot(1, 2, 2)
#     plt.hist(data, bins=50, color='blue', edgecolor='black', alpha=0.7)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.title('Log-Log Histogram')
#     plt.xlabel('Value (log scale)')
#     plt.ylabel('Frequency (log scale)')

#     plt.tight_layout()
#     plt.show()

# # Create a slider widget for the shape parameter 'a' of the Pareto distribution
# a_slider = widgets.FloatSlider(
#     value=3.0,  # default value for the shape parameter
#     min=1.0,    # minimum value for the shape parameter
#     max=5.0,    # maximum value for the shape parameter
#     step=0.1,   # step size for the parameter
#     description='Shape Parameter (a):',
#     continuous_update=False
# )

# # Display the plot and interact with the slider
# interact(plot_pareto_histogram, a_value=a_slider)
