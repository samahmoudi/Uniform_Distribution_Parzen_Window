# Project Title: Uniform Distribution & Parzen Window Estimation

## Overview
This project demonstrates the estimation of a uniform distribution using the Parzen window technique. We derive the mean estimate and plot the results for different values of `h_n` using both mathematical formulations and Python code.

## Contents
- `Uniform_Distribution_Parzen_Window.ipynb`: Python code for plotting the distribution with different `h_n` values.
- `README.md`: Documentation for the project.
- `output_plots`: Folder containing the generated plots.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_repo/Uniform_Distribution_Parzen_Estimation.git
   ```
2. Install the required libraries:
   ```bash
   pip install numpy matplotlib
   ```

## Description

### Problem Statement:

We are tasked with estimating a uniform distribution `p(x)` over the range `0 ≤ x ≤ a` using the Parzen window function `φ(x)`. 

The uniform distribution `p(x)` and Parzen window function `φ(x)` are defined as:

- `p(x) = 1/a` for `0 ≤ x ≤ a` and `p(x) = 0` otherwise.
- `φ(x) = exp(-x)` for `x > 0` and `φ(x) = 0` for `x ≤ 0`.

### 1. **Deriving the Mean Estimate of the Parzen Window Estimation:**

The mean estimate of the Parzen window estimation leads to the following result:

$$
\bar{p}_n(x) = 
\begin{cases} 
0, & \text{for } x < 0 \\
\frac{1}{a} \left(1 - e^{-\frac{x}{h_n}}\right), & \text{for } 0 \leq x \leq a \\
\frac{1}{a} \left(e^{-\frac{a}{h_n}} - 1\right) e^{-\frac{x}{h_n}}, & \text{for } x > a
\end{cases}
$$

### 2. **Plotting `p̅ₙ(x)` for `hₙ = {1, 1/4, 1/16}`:**

We will calculate and plot `p̅ₙ(x)` for the given values of `hₙ` over the range `[-1, 4]`.

### 3. **Finding the Bias:**

We aim to find `hₙ` such that the bias remains under 1% in the range `0 < x < a`. This condition is defined as:

$$
e^{-\frac{x}{h_n}} \leq 0.01 \quad \text{for} \quad 0 < x < a
$$

The corresponding `hₙ` value can be calculated as:
$$
hₙ \leq \frac{a}{\ln(100)}
$$

### 4. **Plotting with Calculated `hₙ`:**

Once the appropriate `hₙ` value is found, we will plot `p̅ₙ(x)` in the smaller range `0 < x < 0.05`.

## Steps
1. **Define the piecewise function**: Implement the piecewise function for `p̅ₙ(x)` based on the Parzen window estimation.
2. **Plot the distribution**: Calculate and plot `p̅ₙ(x)` for different `hₙ` values.
3. **Analyze bias**: Find the `hₙ` value to maintain bias less than 1%.
4. **Generate final plot**: Plot `p̅ₙ(x)` for the derived `hₙ` in the smaller range.

## Usage

To run the code and generate the plots, run:

```bash
python Uniform_Distribution_Parzen_Window.ipynb
```

### Example Outputs:

1. **Plot of `p̅ₙ(x)` for `hₙ = {1, 1/4, 1/16}`:**

![Plot for different h_n values](output_plots/different_h_values.png)

2. **Plot for `hₙ` satisfying the 1% bias condition in range `0 < x < 0.05`:**

![Plot for small x range](output_plots/small_x_range.png)
```python
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Define the piecewise function for p̅ₙ(x)
# p̅ₙ(x) is calculated for three cases based on the value of x
# 1. x < 0 -> 0
# 2. 0 <= x <= a -> (1/a) * (1 - exp(-x/h))
# 3. x > a -> (1/a) * (exp(-a/h) - 1) * exp(-x/h)
# ================================
def piecewise_function(x, a, h):
    return np.piecewise(x, [x < 0, (0 <= x) & (x <= a), x > a], [
        lambda x: 0,
        lambda x: 1/a*(1-np.exp(-x/h)),
        lambda x: 1/a*(np.exp(-a/h)-1)*np.exp(-x/h)
    ])

# ================================
# Set up parameters and plot the distribution for h_n values
# h_n = {1, 1/4, 1/16} and a = 1
# Generate and display the plots
# ================================
a_value = 1
x_values = np.linspace(-1, a_value+3, 400)
h_values = [1, 1/4, 1/16]

# ================================
# Create a figure for the plot and loop through h_values
# Each h_value will generate its corresponding plot
# ================================
fig, ax = plt.subplots(figsize=(10, 6))

for h in h_values:
    y_values = piecewise_function(x_values, a_value, h)
    ax.plot(x_values, y_values, label=f'$h={h}$')

# ================================
# Add labels, title, legend, and grid
# Show the plot
# ================================
ax.set_xlabel('$x$')
ax.set_ylabel('$p̅ₙ(x)$')
ax.legend()
plt.title(f'p̅ₙ(x) for Different $h$ Values with $a={a_value}$')
plt.grid(True)
plt.savefig('output_plots/different_h_values.png')
plt.show()

# ================================
# Plot for a smaller x range (0 < x < 0.05)
# Use h_n = 0.0021714724 for bias less than 1%
# ================================
x_values = np.linspace(0, 0.05, 400)
h_value_bias = a_value / np.log(100)

# ================================
# Create a new figure for this smaller range and plot
# ================================
fig, ax = plt.subplots(figsize=(10, 6))

y_values = piecewise_function(x_values, a_value, h_value_bias)
ax.plot(x_values, y_values, label=f'$h={h_value_bias}$')

# ================================
# Add labels, title, legend, and grid
# Show the plot
# ================================
ax.set_xlabel('$x$')
ax.set_ylabel('$p̅ₙ(x)$')
ax.legend()
plt.title(f'p̅ₙ(x) for $h={h_value_bias}$ with $a={a_value}$')
plt.grid(True)
plt.savefig('output_plots/small_x_range.png')
plt.show()
```


