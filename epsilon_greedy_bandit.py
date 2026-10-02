"""
Epsilon-greedy K-armed bandit for WiFi channel selection.

Each CSV column is a channel (arm) and each row a signal-strength reading in dBm.
A reading below THRESHOLD counts as success (channel free, no interference).

USAGE:
    python3 epsilon_greedy_bandit.py FILENAME EPSILON TRAIN% THRESHOLD
    python3 epsilon_greedy_bandit.py data/wifi_channels.csv 0.3 30 -90
"""

# import libraries
import os
import sys
import csv
import random
import matplotlib.pyplot as plt


# args: filename, epsilon, train_percent, threshold

if len(sys.argv) != 5:
    print("ERROR: Not enough or too many input arguments.")
    sys.exit()

filename = sys.argv[1]

# EPSILON
try:
    epsilon = float(sys.argv[2])
    if epsilon < 0 or epsilon > 1:
        epsilon = 0.3
except:
    epsilon = 0.3

# TRAIN %
try:
    train_percent = int(sys.argv[3])
    if train_percent < 0 or train_percent > 50:
        train_percent = 50
except:
    train_percent = 50

# THRESHOLD
try:
    threshold = float(sys.argv[4])
except:
    print("ERROR: invalid threshold.")
    sys.exit()

# load data
try:
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
except:
    print("ERROR: cannot read input file.")
    sys.exit()

# extract labels and numeric data
labels = data[0]
rows = data[1:]

# convert to float
numeric_data = []
for row in rows:
    numeric_data.append([float(x) for x in row])

num_rows = len(numeric_data)
num_bandits = len(labels)

# split into train and test

train_size = int((train_percent / 100) * num_rows)

train_data = numeric_data[:train_size]
test_data = numeric_data[train_size:]

# epsilon-greedy algorithm

counts = [0] * num_bandits
successes = [0] * num_bandits

for t in range(train_size):
    # exploration vs exploitation
    if random.random() < epsilon:
        action = random.randint(0, num_bandits - 1)
    else:
        # choose best estimated bandit
        estimates = [
            (successes[i] / counts[i]) if counts[i] > 0 else 0
            for i in range(num_bandits)
        ]
        action = estimates.index(max(estimates))

    reward = 1 if train_data[t][action] < threshold else 0

    counts[action] += 1
    successes[action] += reward

# compute success probabilities
probabilities = []
for i in range(num_bandits):
    if counts[i] > 0:
        probabilities.append(successes[i] / counts[i])
    else:
        probabilities.append(0)

# choose best bandit
best_index = probabilities.index(max(probabilities))
best_label = labels[best_index]

# test on remaining data
test_success = 0
test_total = len(test_data)

for row in test_data:
    if row[best_index] < threshold:
        test_success += 1

if test_total > 0:
    success_percentage = (test_success / test_total) * 100
else:
    success_percentage = 0

# PRINT RESULTS
print(f"epsilon: {epsilon}")
print(f"Training data percentage: {train_percent} %")
print(f"Success threshold: {threshold}")
print()
print("Success probabilities:")

for i in range(num_bandits):
    print(f"P({labels[i]}) = {probabilities[i]}")

print()
print(f"Bandit [{best_label}] was chosen to be played for the rest of data set.")
print(f"{best_label} Success percentage: {success_percentage}")


# EXPERIMENT (TABLES + PLOTS) same strategy from before but with different parameters

def run_bandit(data, labels, epsilon, train_percent, threshold):
    num_rows = len(data)
    num_bandits = len(labels)

    train_size = int((train_percent / 100) * num_rows)
    train_data = data[:train_size]
    test_data = data[train_size:]

    counts = [0] * num_bandits
    successes = [0] * num_bandits

    # --- TRAINING ---
    for t in range(train_size):
        if random.random() < epsilon:
            action = random.randint(0, num_bandits - 1)
        else:
            estimates = [
                (successes[i] / counts[i]) if counts[i] > 0 else 0
                for i in range(num_bandits)
            ]
            action = estimates.index(max(estimates))

        reward = 1 if train_data[t][action] < threshold else 0
        counts[action] += 1
        successes[action] += reward

    # probabilities
    probs = [
        (successes[i] / counts[i]) if counts[i] > 0 else 0
        for i in range(num_bandits)
    ]

    best_index = probs.index(max(probs))

    # --- TESTING ---
    cumulative_success = []
    success_count = 0

    for i, row in enumerate(test_data):
        if row[best_index] < threshold:
            success_count += 1
        cumulative_success.append(success_count / (i + 1))

    final_success_rate = cumulative_success[-1] * 100 if cumulative_success else 0

    return final_success_rate, cumulative_success


# PARAMETERS epsilon and training % values to test
epsilons = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
train_percents = [10,20,30,40,50]

# store results
all_runs = []

for run in range(3):
    print(f"\n================ RUN {run+1} ================\n")

    table = []

    for e in epsilons:
        row_results = []
        for tp in train_percents:
            final_rate, _ = run_bandit(numeric_data, labels, e, tp, threshold)
            row_results.append(final_rate)
        table.append(row_results)

    all_runs.append(table)

    # PRINT TABLE FORMAT FOR THIS RUN
    print("e \\ TRAIN% -> 10   20   30   40   50")
    for i, e in enumerate(epsilons):
        row_str = f"{e:.1f}    "
        for val in table[i]:
            row_str += f"{val:.2f}  "
        print(row_str)


# Plots for each run

plot_pairs = [(0.3,50), (0.5,50), (0.8,50)]

for run in range(3):
    plt.figure()

    for (e,tp) in plot_pairs:
        _, cumulative = run_bandit(numeric_data, labels, e, tp, threshold)
        plt.plot(cumulative, label=f"e={e}, TRAIN={tp}%")

    plt.xlabel("Time / Row")
    plt.ylabel("Cumulative Success Rate")
    plt.title(f"Run {run+1}")
    plt.legend()
    plt.grid()

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"run_{run+1}_plot.png")
    plt.savefig(plot_path)
    print()
    print(f"Plot saved: {plot_path}")

    plt.close()
