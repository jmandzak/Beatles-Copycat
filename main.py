from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
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

    return x, y

def main():
    x, y = get_data('beatles.txt', 5, 3)

if __name__ == '__main__':
    main()