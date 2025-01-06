import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate z values (raw scores) and compute probabilities
z = np.linspace(-6, 6, 500)  # Input range
probabilities = sigmoid(z)

# Example raw score and its probability
example_raw_score = 2.0
example_probability = sigmoid(example_raw_score)

# Plot the sigmoid function
plt.figure(figsize=(10, 6))
plt.plot(z, probabilities, label="Sigmoid Activation Function", linewidth=2)

# Highlight the example point
plt.scatter(example_raw_score, example_probability, color="red", label=f"Example Point (z={example_raw_score}, p≈{example_probability:.2f})")

# Add threshold lines
plt.axvline(0, color="gray", linestyle="--", label="Decision Threshold (z=0, p=0.5)")
plt.axhline(0.5, color="blue", linestyle="--", linewidth=0.8, label="Threshold Probability (p=0.5)")

# Add labels, title, and legend
plt.title("Sigmoid Activation Function with Loan Approval Example")
plt.xlabel("Raw Score (z)")
plt.ylabel("Probability (p)")
plt.legend()
plt.grid(alpha=0.4)

# Show the plot
plt.show()