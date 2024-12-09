import numpy as np
from tensorflow.keras import layers, models
import random

lotto_data = [
    [3, 12, 23, 34, 35, 42],
    [5, 14, 19, 22, 33, 45],
    [1, 6, 11, 16, 31, 41],
    [7, 12, 23, 28, 39, 43],
    [8, 15, 20, 26, 37, 44],
    [4, 13, 25, 32, 38, 40]
]

flattened_numbers = [num for draw in lotto_data for num in draw]

unique, counts = np.unique(flattened_numbers,return_counts=True)
number_frequency = dict(zip(unique, counts))

total_numbers = 45
weights = np.zeros(total_numbers)
for num, freq in number_frequency.items():
    weights[num - 1] = freq

weights = weights / weights.sum()

lotto_data_normalized = np.array(lotto_data) / total_numbers

X = lotto_data_normalized
Y = lotto_data_normalized

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(6,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, Y, epochs=100, batch_size=4, verbose=1)

def generate_lotto_numbers(model, weights):
    random_input = np.random.rand(6).reshape(1,-1)
    predicted = model.predict(random_input)[0]
    predicted = (predicted * total_numbers).astype(int)

    adjusted_numbers = np.random.choice(
        range(1, total_numbers + 1), size=6, replace=False, p=weights
    )

    combined_numbers = list(set(predicted) | set(adjusted_numbers))
    combined_numbers = sorted(list(set(np.clip(combined_numbers,1,total_numbers))))

    while len(combined_numbers) < 6:
        combined_numbers.append(random.randint(1, total_numbers))

    return sorted(combined_numbers[:6])

new_lotto_numbers = generate_lotto_numbers(model, weights)
new_lotto_numbers = [int(num) for num in new_lotto_numbers]

print('생성된 로또 번호:', new_lotto_numbers)