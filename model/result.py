from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shutil
import csv
import numpy as np

def result_jpg(prediction, Y_test, PATH):

    #save for first 30min(t = 0 ~ 5) prediction
    plt.plot(prediction[0], label='Prediction')
    plt.plot(Y_test[0], label='Real Value')
    plt.xlabel('Time indices(per 5min)')
    plt.ylabel('Blood Glucose Level(mg/dL)')
    plt.legend()
    plt.savefig(PATH + '/1_pred_fisrt_30min.jpg')
    

    #save for 250 min, (t = 0 ~ 49) prediction
    plt.figure(figsize=(15,9))
    X_case = [0, 5, 9, 11, 18, 25, 30, 33]
    X_prediction_range = []
    Y_prediction = []

    for start in X_case:
        X_prediction_range = X_prediction_range + list(range(start, start+6))
        Y_prediction = Y_prediction + prediction[start+5].tolist()

    plt.plot(Y_test[0:50,-1], label='Real Value' , color="#727272", linewidth="3.2")
    plt.plot(prediction[5:55,0], label='Prediction_5')
    plt.plot(prediction[0:50,5], label='Prediction_30', linewidth="3.2")
    plt.scatter(X_prediction_range, Y_prediction, label="Prediction Samples", s = 15, color="red") 

    plt.xlabel('Time indices(per 5min)')
    plt.ylabel('Blood Glucose Level(mg/dL)')
    plt.legend()
    plt.savefig(PATH + '/2_pred_sample.jpg')

    #save for 250 min, (t = 0 ~ 49) prediction
    plt.figure(figsize=(15,9))
    plt.plot(prediction[5:55,0], label='Prediction_5')
    plt.plot(prediction[4:54,1], label='Prediction_10')
    plt.plot(prediction[3:53,2], label='Prediction_15')
    plt.plot(prediction[2:52,3], label='Prediction_20')
    plt.plot(prediction[1:51,4], label='Prediction_25')
    plt.plot(prediction[0:50,5], label='Prediction_30', linewidth="3.2")
    plt.plot(Y_test[0:50,-1], label='Real Value' , color="#727272", linewidth="3.2")
    plt.xlabel('Time indices(per 5min)')
    plt.ylabel('Blood Glucose Level(mg/dL)')
    plt.legend()
    plt.savefig(PATH + '/3_pred_50.jpg')

    #save for 500 min, (t = 0 ~ 99) prediction  
    plt.figure(figsize=(15,9))
    plt.plot(prediction[5:105,0], label='Prediction_5')
    plt.plot(prediction[4:104,1], label='Prediction_10')
    plt.plot(prediction[3:103,2], label='Prediction_15')
    plt.plot(prediction[2:102,3], label='Prediction_20')
    plt.plot(prediction[1:101,4], label='Prediction_25')
    plt.plot(prediction[0:100,5], label='Prediction_30', linewidth="3.2")
    plt.plot(Y_test[0:100,-1], label='Real Value' , color="#727272", linewidth="3.2")
    plt.xlabel('Time indices(per 5min)')
    plt.ylabel('Blood Glucose Level(mg/dL)')
    plt.legend()
    plt.savefig(PATH + '/4_pred_100.jpg')

    #save for 750 min, (t = 0 ~ 99) prediction 
    plt.figure(figsize=(15,9))
    plt.plot(prediction[5:155,0], label='Prediction_5')
    plt.plot(prediction[4:154,1], label='Prediction_10')
    plt.plot(prediction[3:153,2], label='Prediction_15')
    plt.plot(prediction[2:152,3], label='Prediction_20')
    plt.plot(prediction[1:151,4], label='Prediction_25')
    plt.plot(prediction[0:150,5], label='Prediction_30', linewidth="3.2")
    plt.plot(Y_test[0:150,-1], label='Real Value' , color="#727272", linewidth="3.2")
    plt.xlabel('Time indices(per 5min)')
    plt.ylabel('Blood Glucose Level(mg/dL)')
    plt.legend()
    plt.savefig(PATH + '/5_pred_150.jpg')

    #save for 2500 min, (t = 0 ~ 99) prediction
    plt.figure(figsize=(15,9))
    plt.plot(prediction[5:505,0], label='Prediction_5')
    plt.plot(prediction[4:504,1], label='Prediction_10')
    plt.plot(prediction[3:503,2], label='Prediction_15')
    plt.plot(prediction[2:502,3], label='Prediction_20')
    plt.plot(prediction[1:501,4], label='Prediction_25')
    plt.plot(prediction[0:500,5], label='Prediction_30', linewidth="3.2")
    plt.plot(Y_test[0:500,-1], label='Real Value' , color="#727272", linewidth="3.2")
    plt.xlabel('Time indices(per 5min)')
    plt.ylabel('Blood Glucose Level(mg/dL)')
    plt.legend()
    plt.savefig(PATH + '/6_pred_500.jpg')
    
def save_yaml_and_error(yaml_path, save_path, date, flag = 1):
    if flag == 2:
        source = yaml_path + "Settings2.yaml"
    else:
        source = yaml_path + "Settings.yaml"
    destination = save_path + "/Settings.yaml"
    
    shutil.copyfile(source, destination)

def save_numeric_result(prediction, Y_test, date):

    mse_all = np.sqrt(mean_squared_error(Y_test.reshape(-1,6), prediction))
    mae_all = mean_absolute_error(Y_test.reshape(-1,6), prediction)
    mse_30 = np.sqrt(mean_squared_error(Y_test[:,-1], prediction[:,-1]))
    mae_30 = mean_absolute_error(Y_test[:,-1], prediction[:,-1])

    f = open('./result.csv', 'a', encoding = 'utf-8', newline='')
    data = [str(date), mae_30, mae_all, mse_30, mse_all]
    print(data)
    writer = csv.writer(f)
    writer.writerows([data])
    f.close()