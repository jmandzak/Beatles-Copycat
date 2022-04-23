import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
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

    # print(x.shape)
    # print(y.shape)

    # x = x.reshape((len(x_slices), 1, window_size, len(mapping)))
    # y = y.reshape((len(x_slices), 1, window_size, len(mapping)))

    # print(x.shape)
    # print(y.shape)

    return x, y


def train_model(model, x, y, num_epochs):

    # THIS IS JUST TO MAKE TESTING EASIER BY MAKING IT QUICKER
    # NEED TO REMOVE EVENTUALLY
    x = x[:5000]
    y = y[:5000]

    model.compile(loss=CategoricalCrossentropy(), optimizer='adam')

    # while loop so you can generate text every certain number of epochs
    i = 0
    while(i < 5):
        model.fit(x, y, epochs=int(num_epochs/5))
        i += 1

def main():
    # check for command line args
    if len(sys.argv) < 7:
        print('USAGE: python main.py [input_file] [lstm/simple] [hidden_state_size (int)] [window_size (int)] [stride (int)] [sampling_temp (int)]')
        return

    input_file = sys.argv[1]
    type_model = sys.argv[2]
    hidden_state_size = int(sys.argv[3])
    window_size = int(sys.argv[4])
    stride = int(sys.argv[5])
    samp_temp = float(sys.argv[6])

    x, y = get_data(input_file, window_size, stride)

    print(x[0].shape)
    if(type_model == 'simple'):
        model = Sequential()
        model.add(layers.SimpleRNN(hidden_state_size, return_sequences=True, input_shape=(5, x.shape[2])))
        model.add(layers.Dense(x.shape[2], activation='softmax'))

        train_model(model, x, y, 20)
        # print(model.summary())

        # test = x[0]
        # test = test.reshape(1, 5, 48)
        # ans = model.predict(x[0])
        # print(ans.shape)
        # # print(ans)
        # print(ans[0,0:4])

if __name__ == '__main__':
    main()