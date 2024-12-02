from sklearn.datasets import fetch_openml
import numpy as np
import ssl
 
def get_data():
    ssl._create_default_https_context = ssl._create_unverified_context
    mnist = fetch_openml(name='mnist_784', version=1)
    x_all = mnist['data'].astype(np.float32) / 255
    y_all = mnist['target'].astype(np.int32)
    x_train, x_test = np.split(x_all, [60000])
    y_train, y_test = np.split(y_all, [60000])

    x_train = x_train.reshape([-1,1,28,28])
    x_test = x_test.reshape([-1,1,28,28])
    return x_train,y_train,x_test,y_test