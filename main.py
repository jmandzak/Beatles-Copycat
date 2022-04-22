from sklearn.preprocessing import OneHotEncoder
import numpy as np


def get_data(filename, window_size, stride):
    f = open(filename, 'r')
    text = f.read()

    # mapping = {}
    # unique_char = 0
    encoded = []
    
    # for c in text:
    #     if c in mapping:
    #         encoded.append(mapping[c])
    #     else:
    #         mapping[c] = unique_char
    #         unique_char += 1
    #         encoded.append(mapping[c])

    # taken from example rnn code
    chars = sorted(list(set(text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    encoded = np.array([mapping[char] for char in text])
    print(encoded)

    starting_index = 0
    all_slices = []
    while(starting_index + window_size+1 < len(encoded)):
        all_slices.append(encoded[starting_index:starting_index+window_size+1])
        starting_index += window_size+1
    
    x_slices = []
    y_slices = []
    for slice in all_slices:
        x = slice[:-1]
        y = slice[1:]
        string_x = [str(int) for int in x]
        string_y = [str(int) for int in y]
        
        string_x = "".join(string_x)
        string_y = "".join(string_y)

        x_slices.append(string_x)
        y_slices.append(string_y)

    # print(x_slices)
    print(len(y_slices))
    print(str(x_slices[0]))
    print(str(y_slices[0]))

    # encoder = OneHotEncoder()
    # encoded = encoder.fit_transform(x_slices)
    # print((encoded.toarray()).shape)

def main():
    get_data('beatles.txt', 5, 1)

if __name__ == '__main__':
    main()