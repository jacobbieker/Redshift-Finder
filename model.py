from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import os
from pprint import pprint

data = np.loadtxt("data/data.txt")

print(len(data))

# Remove IDs, as not needed
data = np.delete(data, 0, axis=1)
labels = data[:,-1]
print(labels)
data = np.delete(data, -1, axis=1)
print(data.shape)
print(data)


model = Sequential()
model.add(Conv1D(16, 3, activation="relu", input_shape=(10,1)))
model.add(Conv1D(32, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(500)) # Needed for the 500 redshift bins
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
model.summary()

# Now create the distribution for the redshift data
width = 0.5
number_pulls = 10000
bin_edges = np.arange(0.0, 5.01, 0.01)
print(len(bin_edges))
redshift_distributions = []
for redshift in labels:
    dist = np.random.normal(redshift, scale=width, size=number_pulls)
    dist = dist[dist > 0.0]
    redshift_distributions.append(np.histogram(dist, bins=bin_edges, density=True)[0])

redshift_distributions = np.asarray(redshift_distributions)
print(redshift_distributions.shape)



# Create train/val/test set
data = np.expand_dims(data, axis=2)
train = data[0:int(len(data)*0.8)]
#validate = data[int(len(data)*0.6):int(len(data)*0.8)]
test = data[int(len(data)*0.8):]

# And labels for them
#redshift_distributions = np.expand_dims(redshift_distributions, axis=2)
train_y = redshift_distributions[0:int(len(redshift_distributions)*0.8)]
#validate_y = redshift_distributions[int(len(redshift_distributions)*0.6):int(len(redshift_distributions)*0.8)]
test_y = redshift_distributions[int(len(redshift_distributions)*0.8):]

model.fit(train, train_y, epochs=500, batch_size=64, validation_split=0.2, shuffle=True)
model.predict(test)
#loading the photometry templates, which are previously produced and saved as .pickle. Multi-band photometry template can be parametrized with only two parameters: spectroscopic redshift (zs) and template number (tn). Saving them in templateScale
#templateScale = {}
#for i in range(50):
#    with open(os.path.join("data", "photoZ", f"{i}templatez01t001.pickle"), 'rb') as handle:
#        templateScale.update(pickle.load(handle, encoding='latin1'))
#print(len(templateScale))

#pprint(templateScale[list(templateScale.items())[0]])