import numpy as np

best_value = 0

number = 1000

for _ in range(number):
    x = np.random.randint(low=0, high=10, dtype=int)
    y = np.random.randint(low=0, high=10, dtype=int)
    z = np.random.randint(low=0, high=10, dtype=int)

    if x + y <= 10 and 2 * x + z <= 9 and y + 2 * z <= 11:
        calculate_value = 3 * x + 2 * y + 5 * z

        if calculate_value > best_value:
            best_value = calculate_value

print("best_value: ", best_value)
