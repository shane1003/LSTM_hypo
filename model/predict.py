import numpy as np
import torch

def predict(model, X_test, output_window, scaler, option, device):


    print(option)
    # if scaler is used, transform X_test scaled
    if option == 1:
        X_test = scaler.fit_transform(X_test)
    elif option == 2:
        X_test = scaler.fit_transform(X_test)

    prediction = np.empty((6,))

    for idx in range(X_test.shape[0]):
        predict = model.predict(torch.tensor(X_test[idx]).to(device).float(), target_len=output_window)
        prediction = np.vstack((prediction, predict))

    prediction = np.delete(prediction, 0, axis = 0)


    # if scaler is used, transform prediction unscaled
    if option == 1:
        prediction = scaler.inverse_transform(prediction)
    elif option == 2:
        prediction = scaler.inverse_transform(prediction)

    return prediction
    