'''import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# Parameters for the log-normal distribution
mean = 0         # Mean of the underlying normal distribution
sigma = 1        # Standard deviation of the underlying normal distribution
size = 100000    # Number of samples

# Generate log-normal distributed data
data = np.random.lognormal(mean, sigma, size)

# Plot histogram on a log-log scale
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(data, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.7)

# Apply log scales to the histogram plot
plt.yscale('log')
plt.xscale('log')
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Histogram of Log-Normal Distribution on Log-Log Scale")
plt.grid(True)
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters for the log-normal distribution
mu = 1.0      # Mean of the log-normal distribution (for log(x))
sigma = 0.5   # Standard deviation of the log-normal distribution (for log(x))
num_samples = 10000  # Number of samples to draw

# Generate samples from a log-normal distribution
samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)

# Plot the histogram of the original log-normal samples
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, color='skyblue', edgecolor='black', density=True)
plt.title("Histogram of Log-Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Take the logarithm of the samples
log_samples = np.log(samples)

# Plot the histogram of the log-transformed samples
plt.subplot(1, 2, 2)
plt.hist(log_samples, bins=50, color='salmon', edgecolor='black', density=True)

# Overlay a Gaussian curve for comparison
x = np.linspace(log_samples.min(), log_samples.max(), 100)
gaussian_pdf = stats.norm.pdf(x, mu, sigma)
plt.plot(x, gaussian_pdf, color='darkred', lw=2, label='Gaussian fit')

plt.title("Histogram of Log-Transformed Samples (Gaussian)")
plt.xlabel("Log(Value)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

