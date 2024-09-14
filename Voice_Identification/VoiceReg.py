# Imports
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import time
import multiprocessing  
import tensorflow as tf
import librosa
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# Setup environment 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Make consitantly reproduceable results
# SEED = -1
# os.environ['PYTHONHASHSEED']=str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# Upscale a dataset by copying and trimming.
def oversample_dataset(dataset, upscaled_length):
    if (len(dataset) >= upscaled_length): return
    new_dataset = list(dataset)
    random.shuffle(new_dataset)
    for i in range(upscaled_length // len(dataset)):
        new_dataset.extend(list(dataset))
    random.shuffle(new_dataset)
    return new_dataset[:upscaled_length]
    
# Get files
NEGITIVE_DIR = dir_path + '/Negitive/'
POSITIVE_DIR = dir_path + '/Positive/'
TEST_DIR = dir_path + '/Test/'
test_files = glob.glob(TEST_DIR + "*.json")
positive_files = glob.glob(POSITIVE_DIR + "*.json")
negitive_files = glob.glob(NEGITIVE_DIR + "*.json")

# Balance the dataset if needed
if (len(positive_files) > len(negitive_files)):
    negitive_files = oversample_dataset(negitive_files, len(positive_files))
elif (len(positive_files) > len(negitive_files)):
    positive_files = oversample_dataset(positive_files, len(negitive_files))

# Get the splits
files = positive_files + negitive_files
random.shuffle(files)
X_train, X_val_unsplit = train_test_split(files, test_size=0.4)
X_val, X_test_raw = train_test_split(X_val_unsplit, test_size=(0.2))

# List examples amounts
print('# All examples: {}'.format(len(files)))
print('# Training examples: {}'.format(len(X_train)))
print('# Validation examples: {}'.format(len(X_val)))
print('# Test examples: {}'.format(len(X_test_raw)))


# Network shapes/length
n_features = 20
max_length = 500
n_classes = 1

# Network training variables
learning_rate = 0.0001
n_epochs = 100
batch_size = 64
dropout = 0.7

input_shape = (max_length, n_features, 1)

# Other variables
checkpoint_filepath = "checkpoints/voice_recognition_best_model.keras"

# Model architecture
model = tf.keras.models.Sequential([
    # CNN network architecture 
    tf.keras.layers.Conv2D(filters=32*2, kernel_size=(7,7), activation='relu', input_shape=input_shape,padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(filters=32*2, kernel_size=(5,5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(filters=32*2, kernel_size=(3,3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(dropout), #seed=SEED
    tf.keras.layers.Dense(128*4, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(n_classes, activation='sigmoid')
])

# Compiling and logging the model
opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy']) #categorical_crossentropy

model.summary()

#tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)

# Get label and data from a path to json MFCC
def get_json_data(mfcc_json, i):
    label = mfcc_json.split('/')[-2]
    label_formatted = float(label == "Positive") 
    data = []
    with open(mfcc_json, "rb") as jsonfile:
        data = json.load(jsonfile)
    return data, label_formatted
    
# Parallelly process and get all json data from files
def get_preprocessed_data_from_files(data):
    start = time.time()

    num_cores = multiprocessing.cpu_count()
    return_data = Parallel(n_jobs=num_cores)(delayed(get_json_data)(data[i], i) for i in range(len(data)))
    
    # Turn an array of tuples into 2 arrays of label and data
    return_data_unformatted = [list(i) for i in zip(*return_data)]
    input_data = return_data_unformatted[0]
    output_data = return_data_unformatted[1]

    # Give total processing time
    end = time.time()
    print("Preprocessing elapsed: " + str(end - start))

    # Return the data in formats/shapes which the network will understand
    return np.array(input_data).reshape((len(input_data), max_length, n_features, 1)), np.array(output_data).reshape((len(output_data), 1))

# Preprocess and get data
X,y = get_preprocessed_data_from_files(X_train)
Xval,Yval = get_preprocessed_data_from_files(X_val)
X_test, y_test = get_preprocessed_data_from_files(X_test_raw)

# Have callbacks to stop the model early if it starts overfitting
callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)]
history = model.fit(X,y,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(Xval,Yval),
                    callbacks=callbacks
                    )

model.load_weights(checkpoint_filepath)

# Evaluate success
print("Best model evaluated on test data:")
results = model.evaluate(X_test, y_test, batch_size=128)

# Predict on X, and give me the sum and total to see if it is just always being 1 or 0
predict = model.predict(X)
prediction = np.array(predict).flatten()
print("Prediction sum: " + str(sum(list(map(round, prediction)))) + " total: " + str(len(X)))

# Graph Accuracy (validation included)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')  
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Loss (validation included)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# X_custom_test,_ = get_preprocessed_data_from_files(test_files)
# prediction_test = model.predict(X_custom_test)
# print(list(map(np.round, np.reshape(prediction_test, len(prediction_test)))))
# print(list(map(os.path.basename, test_files)))