import pandas as pd
import numpy as np

def X_Y_from_data(data):
    """
    make X and Y with preprocessed data
    
    Args:
        data : preprocessed data
    
    Returns:
        X : input data
        Y : label data
    """

    X = []
    Y = []
    for datum in data:
        X.append(datum[:-1])
        try:
            Y.append(datum[1:, 0])
        except:
            Y.append(datum[1:])

    return X, Y

def X_Y2train_test(X, Y, output_window = 6, input_size = 3, stride = 1):
    """
    make X and Y to X_train, Y_train, X_test, Y_test
    
    Args:
        X : preprocessed input data(CGM, CHO, Insulin)
        Y : preprocessed label data(CGM)
        input_window : encoder side input data count
        output_window : decoder side output data count
        stride : stride
    
    Returns:
        X_train : train input data
        Y_train : train label data
        X_test : test input data
        Y_test : test output data
    """
    
    X_train = np.empty((1, output_window, input_size))
    Y_train = np.empty((1, output_window))

    for i in range(0, len(X)-1): 
        win_X, win_Y = data2Window(X = X[i],
                                   Y = Y[i],
                                   output_window = output_window,
                                   stride = stride)

        if input_size == 1:
            win_X = win_X.reshape(-1, 6, 1)

        try:
            X_train = np.vstack((X_train, win_X))
            Y_train = np.vstack((Y_train, win_Y))
            
        except:
            continue

    X_test, Y_test = data2Window(X = X[-1],
                                 Y = Y[-1],
                                 output_window = output_window,
                                 stride = stride)
    
    
    return X_train[1:], Y_train[1:], X_test, Y_test

def data2Window(X, Y, output_window = 6, stride = 1):
    """
    change the full train and test data to window dataset for seq2seq
    
    Args :
        X : feature(CGM, CHO, Insulisn)
        Y : label(CGM)
        input_window : input window size
        output_window : output window size(default : 6)
        stride : window stride(default : 1)

    Returns :
        window dataset
    """

    #print("Stride : ", stride)
    win_X = []
    win_Y = []
    #number of data
    L = len(X)
    #number of samples by using window with stride.
    num_samples = (L - output_window) // stride + 1
    
    #input & output : shape = (sizeof window, number of samples)
    idx = 0
    for i in range(num_samples):
        win_X.append(np.array(X[idx:idx+output_window])) #t ~ t + input_window + output_window
        win_Y.append(np.array(Y[idx:idx+output_window])) #t+input_window ~ t+input_window+output_window use for teacher forcing
        idx = idx + stride
        
    return np.array(win_X), np.array(win_Y)