import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np


def get_data(filename, window_size, stride):
    f = open(filename, 'r')
    text = f.read()


    encoded = []

    # taken from example rnn code
    chars = sorted(list(set(text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    reverse_map = {v: k for k, v in mapping.items()}
    encoded = np.array([mapping[char] for char in text])

    # create the slices
    starting_index = 0
    all_slices = []
    while(starting_index + window_size+1 < len(encoded)):
        all_slices.append(encoded[starting_index:starting_index+window_size+1])
        starting_index += stride
    

    # turn the slices into x slices and y slices
    x_slices = []
    y_slices = []
    for slice in all_slices:
        x = slice[:-1]
        y = slice[1:]

        x_slices.append(x)
        y_slices.append(y)

    # one hot encode the slices and return them
    x = np.array(to_categorical(x_slices))
    y = np.array(to_categorical(y_slices))

    return x, y, reverse_map


def predict_chars(model, x, samp_temp, num_predictions, reverse_map):
    # create a new model with the same layers except an added lambda layer to take temp into account
    new_model = Sequential()
    new_model.add(model.layers[0])
    new_model.add(layers.Lambda(lambda x: x / samp_temp))
    new_model.add(model.layers[1])
    new_model.compile(loss=CategoricalCrossentropy(), optimizer='adam')

    # print the initial input
    for i in range(x.shape[1]):
        letter = np.argmax(x[0][i])
        letter = reverse_map[letter]
        print(letter, end='')

    for i in range(num_predictions):
        # get the prediction
        y_pred = new_model.predict(x)
        last_letter = y_pred[0][-1]
        letter = np.argmax(last_letter)

        # add a dimension to make the dims match (1, vocab_size)
        last_letter = np.expand_dims(last_letter, 0)

        # update x to add on the new letter and pop off the first letter so we stay size (1, window_size, vocab_size)
        x[0] = (np.append(x[0], last_letter, axis=0))[1:]
                
        # print the letter
        letter = reverse_map[letter]
        print(letter, end='')

    print()
    

def train_model(model, x, y, num_epochs, reverse_map, temp, type_model, hidden_size):

    # THIS IS JUST TO MAKE TESTING EASIER BY MAKING IT QUICKER
    # NEED TO REMOVE EVENTUALLY
    # x = x[:5000]
    # y = y[:5000]

    model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    # go ahead and grab a sample from the training data so we can see how the predictions evolve over time
    sample = x[0]
    sample = np.expand_dims(sample, 0)

    # set up the tensorboard before we fit
    # creating unique name for tensorboard directory
    # log_dir = "logs/class/" + f'window_size={x.shape[1]}-hidden={hidden_size}-model={type_model}'
    # #Tensforboard callback function
    # tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    # while loop so you can generate text every certain number of epochs
    i = 0
    while(i < 5):
        model.fit(x, y, epochs=int(num_epochs/5), callbacks=[tensorboard_callback])
        i += 1

        sample = x[np.random.randint(len(x))]
        sample = np.expand_dims(sample, 0)

        predict_chars(model, sample, temp, 300, reverse_map)

    # for generating the graphs
    # model.fit(x, y, epochs=num_epochs, callbacks=[tensorboard_callback])

    # sample = x[np.random.randint(len(x))]
    # sample = np.expand_dims(sample, 0)

    # predict_chars(model, sample, temp, 300, reverse_map)

def main():
    # check for command line args
    if len(sys.argv) < 7:
        print('USAGE: python main.py [input_file] [lstm/simple] [hidden_state_size (int)] [window_size (int)] [stride (int)] [sampling_temp (float)]')
        return

    input_file = sys.argv[1]
    type_model = sys.argv[2]
    hidden_state_size = int(sys.argv[3])
    window_size = int(sys.argv[4])
    stride = int(sys.argv[5])
    samp_temp = float(sys.argv[6])

    x, y, reverse_map = get_data(input_file, window_size, stride)

    if(type_model == 'simple'):
        model = Sequential()
        model.add(layers.SimpleRNN(hidden_state_size, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(layers.Dense(x.shape[2], activation='softmax'))
        train_model(model, x, y, 100, reverse_map, samp_temp, 'simple', hidden_state_size)
    elif(type_model == 'lstm'):
        model = Sequential()
        model.add(layers.LSTM(hidden_state_size, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(layers.Dense(x.shape[2], activation='softmax'))
        train_model(model, x, y, 100, reverse_map, samp_temp, 'lstm', hidden_state_size)

if __name__ == '__main__':
    main()