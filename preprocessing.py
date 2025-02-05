# Load spike times and cursor velocity data
spike_times = f['/units/spike_times'][:]
spike_times_index = f['/units/spike_times_index'][:]
timestamps = f['/processing/behavior/Velocity/cursor_vel/timestamps'][:]
cursor_velocity = f['/processing/behavior/Velocity/cursor_vel/data'][:]

# Define binning parameters
window_size = 0.1  # 100ms window
bin_size = 0.01  # 10ms bins
threshold_velocity = 45  # Filter threshold for |Vx| or |Vy|

# Create bin edges
t_min, t_max = timestamps[0], timestamps[-1]
bin_edges = np.arange(t_min, t_max, bin_size)
num_bins = len(bin_edges) - 1

binned_spikes = np.zeros((len(spike_times_index), num_bins))

for i, (start, end) in enumerate(zip(spike_times_index[:-1], spike_times_index[1:])):  
    spike_hist, _ = np.histogram(spike_times[start:end], bins=bin_edges)
    binned_spikes[i] = spike_hist



# Binning cursor velocity
digitized = np.digitize(timestamps, bin_edges) - 1
binned_velocity = np.zeros((num_bins, 2))
for i in range(num_bins):
    mask = digitized == i
    if np.any(mask):
        binned_velocity[i] = np.mean(cursor_velocity[mask], axis=0)
    else:
        binned_velocity[i] = [0, 0]

# Filtering abnormal velocities
valid_indices = np.all(np.abs(binned_velocity) <= threshold_velocity, axis=1)
binned_spikes = binned_spikes[:, valid_indices]
binned_velocity = binned_velocity[valid_indices]

# Save processed data
np.savez("processed_data.npz", spikes=binned_spikes, velocity=binned_velocity)

# Ensure equal length for binned spikes and velocity data
min_length = min(binned_spikes.shape[1], data.shape[0])

# Trim both datasets to the same length
binned_spikes = binned_spikes[:, :min_length]
timestamps = timestamps[:min_length]
data = data[:min_length]


# Tokenize binned spikes
tokens = binned_spikes.T  

# Temporal Embedding
embedding_dim = 16  
time_steps = tokens.shape[0]
position_embedding = np.zeros((time_steps, embedding_dim))

# Generate temporal embeddings (e.g., sinusoidal or linear)
for t in range(time_steps):
    position_embedding[t] = t / time_steps  # normalized position

# Add embedding to tokens
tokens_with_embedding = np.concatenate((tokens, position_embedding), axis=1)

sequence_length = 50  # Number of time steps in each input sequence

X_seq = []
Y_seq = []

for i in range(len(tokens_with_embedding) - sequence_length):
    X_seq.append(tokens_with_embedding[i : i + sequence_length])  # Shape (50, 77)
    Y_seq.append(data[i + sequence_length])  # Shape (2,)

# Convert to NumPy arrays
X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)


# Define the split index
split_index = int(len(X_seq) * 0.8)  # 80% for training, 20% for testing

# Split the data sequentially
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = Y_seq[:split_index], Y_seq[split_index:]
