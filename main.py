DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), ('L'))
        img.save(path)


DATA_DIR = 'data/'
TEST_DIR = 'test/'
TEST_DATA_FILENAME    = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME  = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME   = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels.idx1-ubyte'


n_train = 10000
n_test = 100

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # skip magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = bytes_to_int(f.read(1))
                    row.append(pixel)
                image.append(row)
            images.append(image)
    print(n_images)
    print(n_rows)
    print(n_columns)
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = len(X[0])	
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, X_train, Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



def main():
    X_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    X_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = read_labels(TEST_LABELS_FILENAME, n_test)


    if len(X_test) < 100:
    	for idx, test_sample in enumerate(X_test):
        	write_image(test_sample, f'{TEST_DIR}{idx}.png')


    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    X_train = np.array(X_train).T
    y_train = np.array(y_train)
    X_train = X_train / 255.
    
    X_test = np.array(X_test).T
    y_test = np.array(y_test)
    X_test = X_test / 255.


    W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 500)


    test_prediction(0, W1, b1, W2, b2, X_train, y_train)
    test_prediction(3, W1, b1, W2, b2, X_train, y_train)
    test_prediction(1, W1, b1, W2, b2, X_train, y_train)
    test_prediction(2, W1, b1, W2, b2, X_train, y_train)

    print('\n========================TEST DATA=======================\n')

    # X_test = [read_image(f'{DATA_DIR}test.png')]
    # X_test = extract_features(X_test)
    # y_test = [5]
    # y_test = np.array(y_test)
    # X_test = np.array(X_test).T
    # X_test = X_test / 255.


    dev_predictions = make_predictions(X_test, W1, b1, W2, b2)

    print(f'Predictions: {dev_predictions}')
    print(f'Y: {y_test}')
    accuracy = get_accuracy(dev_predictions, y_test)
    print(f'Accurace on test data: {accuracy * 100}%')


        
if __name__ == '__main__':
    main()


'''

	7 2 1 0 4 1 4 9 6 9 0 2 9 0 1 8 9 7 8 4 9 6 6 5 4 0 7 4 0 1 3 1 3 6 7 2 7

	7 2 1 0 4 1 4 9 6 9 0 8 9 0 1 5 9 7 5 4 7 6 6 5 9 0 7 4 0 1 3 1 3 6 7 2 7 0.822


'''




