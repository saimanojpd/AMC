################################################# Importing the libraries ##################################################
import numpy as np
import pandas as pd
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils_keras import KerasModelWrapper
###################################################### Trained Model #######################################################
backend.set_learning_phase(False)
keras_model=load_model('/home/labadmin/Downloads/Mnist_Digits_Model_Malware.h6')
############################################## Importing the data #############################################################
dataset_1 = pd.read_csv('training_dataset.csv')
dataset_2 = pd.read_csv('testing_dataset.csv')
############################################### Merging the dataframe #############################################
dataset=dataset_1.append(dataset_2)
################################################ 8 feature selection ##########################################
#dataset= dataset[['LLC-load-misses','branch-instructions','cache-references','branch-misses','node-stores','branch-loads','cache-misses','LLC-loads','class']]
dataset  = dataset[dataset['LLC-load-misses'] < 100000]
############################################## Spiltting into X and Y ###############################################
X_data=dataset.iloc[:, 0:16].values
from sklearn.preprocessing import Imputer #Handling NAN Values
X_data = Imputer().fit_transform(X_data)
Y_data=dataset.iloc[:,16].values
############################################## Randomzing the data ###############################################
idx=np.random.RandomState(seed=45).permutation(len(X_data))
X_data=X_data[idx]
Y_data=Y_data[idx]
############################################## Spiltting in X-Y test and X-Y train ##################################
X_train= X_data[0:40000]
X_test=X_data[40001:49456]

Y_train= Y_data[0:40000]
Y_test= Y_data[40001:49456]
############################################ Categorical varaiables ##############################################
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()                                         
Y_train=Y_train[:]=labelencoder_X.fit_transform (Y_train[:]) 
Y_train=np.reshape(Y_train,(-1,1))
onehotencoder= OneHotEncoder(categorical_features=[0])                  
Y_train=onehotencoder.fit_transform(Y_train).toarray()
Y_train=Y_train[:,] #Avoid Dummy Variable Trap

from sklearn.preprocessing import LabelEncoder 
labelencoder_X= LabelEncoder()                                          
Y_test=Y_test[:,]=labelencoder_X.fit_transform (Y_test[:,]) 
Y_test=np.reshape(Y_test,(-1,1))
onehotencoder= OneHotEncoder(categorical_features=[0])                  
Y_test=onehotencoder.fit_transform(Y_test).toarray()
Y_test=Y_test[:,] #Avoid Dummy Variable Trap
############################################# Normalizing the data #########################################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
################################################ Testing on  All Samples ###################################################

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

#Definning the session
sess =  backend.get_session()

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 16))
y = tf.placeholder(tf.float32, shape=(None, 6))

pred = np.argmax(keras_model.predict(X_test,None), axis = 1)
acc =  np.mean(np.equal(pred, Y_test.argmax(axis=1)))

print("The Test accuracy is: {}".format(acc))

################################################# Adversarial Attack  #####################################################
#JSMA
wrap=KerasModelWrapper(keras_model)
jsma = SaliencyMapMethod(wrap, back='tf', sess=sess)
jsma_params = {'theta':0.4, 'gamma':0.4,
                   'clip_min':0., 'clip_max': 1.,
                   'y_target':None}

adv_x = jsma.generate_np(X_test, **jsma_params)
adv_conf = keras_model.predict(adv_x)
adv_pred = np.argmax(adv_conf, axis = 1)
adv_pred=np.reshape(adv_pred,(-1,1))
adv_acc =  np.mean(np.equal(adv_pred, Y_test.argmax(axis=1)))
print("The adversarial  accuracy is: {}".format(adv_acc))

################################################# De-Normalizing Adv samples ##############################################
adv_x_unscaled = scaler.inverse_transform(adv_x)
a=np.ceil(adv_x_unscaled)
a=adv_x_unscaled.astype(np.int32)
adv_conf_1 = keras_model.predict(a)
adv_pred_1 = np.argmax(adv_conf_1, axis = 1)
adv_pred_1=np.reshape(adv_pred_1,(-1,1))
adv_acc_1 =  np.mean(np.equal(adv_pred_1,Y_test.argmax(axis=1)))
print("The adversarial  accuracy is: {}".format(adv_acc_1))
############################################### De-Normalizing the test set ###################################
x_test_unscaled=scaler.inverse_transform(X_test)
b=x_test_unscaled.astype(np.int32)
############################################### Saving the data ###############################################
import pickle
filename = '/home/labadmin/Desktop/a_date.lol'
pickle.dump(a, open(filename, 'wb'))



