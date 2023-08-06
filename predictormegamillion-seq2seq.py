import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.metrics import mse
from keras.regularizers import l1
import matplotlib.pyplot as plt
from keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner import HyperModel, RandomSearch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

colnames = ['gamename', 'month', 'day', 'year', 'num1', 'num2', 'num3', 'num4', 'num5', 'megaball', 'megaplier']
df = pd.read_csv("https://www.texaslottery.com/export/sites/lottery/Games/Mega_Millions/Winning_Numbers/megamillions.csv",names=colnames)

def sort_numbers(row):
    numbers = row[['num1', 'num2', 'num3', 'num4', 'num5']]
    numbers = sorted(numbers)
    row[['num1', 'num2', 'num3', 'num4', 'num5']] = numbers
    return row

df = df.apply(sort_numbers, axis=1)

df1 = df.copy()

df1["Date"] = pd.to_datetime(df[['year','month','day']].astype(str).agg('-'.join, axis=1))

df1.set_index('Date', inplace=True)
df1.sort_index(inplace=True)
df1 = df1[df1.index > '2017-10-31']

df1 = df1.drop(columns = ['gamename', 'month', 'day','year','megaplier'])
df = df.drop(columns = ['gamename', 'month', 'day','year','megaplier'])

window_length = 6
train_full = df1.copy()
number_of_features = train_full.shape[1]

train, val_data = train_test_split(train_full, test_size=0.2, random_state=42) #reducing validation set

train_rows = train.values.shape[0]
train_samples = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)
train_labels = np.empty([ train_rows - window_length, number_of_features], dtype=float)

for i in range(0, train_rows-window_length):
    train_samples[i] = train.iloc[i : i+window_length, 0 : number_of_features]
    train_labels[i] = train.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

scaler = StandardScaler()
transformed_dataset = scaler.fit_transform(train.values)
scaled_train_samples = pd.DataFrame(data=transformed_dataset, index=train.index)

minimum_value = scaled_train_samples.min().min()
maximum_value = scaled_train_samples.max().max()

#Let’s, create our x_train and y_train data sets.
x_train = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)
#y_train = np.empty([ train_rows - window_length, number_of_features], dtype=float)
y_train = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)

for i in range(0, train_rows-window_length):
    x_train[i] = scaled_train_samples.iloc[i : i+window_length, 0 : number_of_features]
    y_train[i] = scaled_train_samples.iloc[i+window_length : i+window_length+1, 0 : number_of_features]


val_rows = val_data.values.shape[0]
val_samples = np.empty([ val_rows - window_length, window_length, number_of_features], dtype=float)
val_labels = np.empty([ val_rows - window_length, number_of_features], dtype=float)

#Let’s, create our x_train and y_train data sets.
x_val = np.empty([ val_rows - window_length, window_length, number_of_features], dtype=float)
#y_val = np.empty([ val_rows - window_length, number_of_features], dtype=float)
y_val = np.empty([ val_rows - window_length, window_length, number_of_features], dtype=float)

transformed_val_dataset = scaler.fit_transform(val_data.values)
scaled_val_samples = pd.DataFrame(data=transformed_val_dataset, index=val_data.index)

for i in range(0, val_rows-window_length):
    x_val[i] = scaled_val_samples.iloc[i : i+window_length, 0 : number_of_features]
    y_val[i] = scaled_val_samples.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

def custom_loss(y_true, y_pred):
    mask = tf.math.logical_and(
        tf.math.logical_and(y_true >= minimum_value, y_true <= maximum_value),
        tf.math.logical_and(y_pred >= minimum_value, y_pred <= maximum_value)
    )

    loss1 = tf.math.squared_difference(y_true, y_pred)

    masked_loss = tf.where(mask, 1000 * loss1, tf.zeros_like(y_pred))
    masked_mean_loss = tf.reduce_mean(masked_loss)

    loss2 = keras.losses.mean_squared_error(y_true, y_pred)
    mean_loss2 = tf.reduce_mean(loss2)

    return masked_mean_loss + mean_loss2


def custom_loss_integer(y_true, y_pred):
    # Mean Squared Error term
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

    # Integer Intersection
    y_true_int = K.round(y_true)
    y_pred_int = K.round(y_pred)

    same_values = tf.cast(tf.equal(y_true_int, y_pred_int), dtype=tf.float32)
    intersection_count = tf.reduce_sum(same_values)

    intersection_term = 1.0 / (1.0 + intersection_count)

    return mse + intersection_term

from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

class MyHyperModel(HyperModel):

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()

        model.add(Bidirectional(LSTM(240,
                                     input_shape=(window_length, number_of_features),
                                     return_sequences=True,
                                     recurrent_dropout=0.1)))  # Added recurrent dropout

        model.add(Dropout(0.3))

        model.add(Bidirectional(LSTM(units=hp.Int('units',
                                                  min_value=240,
                                                  max_value=1024,
                                                  step=10),
                                     input_shape=self.input_shape,
                                     return_sequences=True,
                                     recurrent_dropout=hp.Float('recurrent_dropout',
                                                                min_value=0.1,
                                                                max_value=0.5,
                                                                step=0.1))))  # Added recurrent dropout

        model.add(Dropout(rate=hp.Float(
            'dropout',
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
        )))  # Adding a dropout layer

        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))

        optimizer_choice = hp.Choice('optimizer', ['adam', 'RMSprop'])
        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=0.0001)
                #learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5]))
        elif optimizer_choice == 'RMSprop':
            optimizer = keras.optimizers.RMSprop(
                learning_rate=0.0001)
                #learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5]))

        model.add(Dense(number_of_features))

        model.compile(
            optimizer=optimizer,
            loss=custom_loss_integer,
            metrics=['accuracy'])

        return model


hypermodel = MyHyperModel(input_shape=(window_length, number_of_features), num_classes=6)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1)

# Adding early stopping callback
stop_early = EarlyStopping(monitor='val_accuracy', patience=500)

tuner.search(x_train, y_train,
             epochs=100,
             validation_data=(x_val, y_val),
             callbacks=[stop_early] # Added early stopping in tuner search
             )

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train,
                    epochs=5000,
                    validation_data=(x_val, y_val),
                    callbacks=[stop_early],  # Also added early stopping in model fit
                    batch_size=32)

model.save('megamillion-seq2seq.keras')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.show()

print('-' * 40)
print('Prediction vs. GoundTruth with rounding up')
for i in range(1,10):
  test = df1.copy()
  test = test.tail((window_length+10-i))
  test = test.head((window_length+1))
  #test_Date = df1.iloc[ (test.tail().index[-1]) ]['Date']
  #test_Date = df1.loc[(test.tail().index[-1]), 'Date']
  test_Date = test.tail(1).index
  test1 = test.head((window_length)).copy()
  #test1.drop(['Date'], axis=1, inplace=True)
  test1 = np.array(test1)
  x_test = scaler.transform(test1)
  y_test_pred = model.predict(np.array([x_test]))

  y_test_pred_reshaped = y_test_pred.reshape(-1, y_test_pred.shape[-1])
  y_test_pred_inverse = scaler.inverse_transform(y_test_pred_reshaped)
  y_test_pred_inverse = y_test_pred_inverse.astype(int)
  y_test_pred_inverse = y_test_pred_inverse.reshape(y_test_pred.shape)

  #y_test_true = test.drop(['Date'], axis=1, inplace=True)
  y_test_true = test.tail(1)
  print('Drawing  Date', test_Date)
  print('Prediction:\t', y_test_pred_inverse[0][0] + 1)
  print('GoundTruth:\t', np.array(y_test_true)[0])
  print('-' * 40)

def run_prediction(drawingstopredict):

    print('Generating ',drawingstopredict, ' predictions, based on ', window_length, ' last numbers:')

    for z in range(1, drawingstopredict):
        next = df.copy()
        print('Predict the Future Drawing')
        next = next.tail(window_length + z - 1).head(window_length)
        lastrow = next.iloc[window_length-1].tolist()
        print('last game out of ', window_length, ' used in the prediction >>>', lastrow)
        next = np.array(next)
        #print(next)
        x_next = scaler.transform(next)
        y_next_pred = model.predict(np.array([x_next]))

        y_next_pred_reshaped = y_next_pred.reshape(-1, y_next_pred.shape[-1])
        y_next_pred_inverse = scaler.inverse_transform(y_next_pred_reshaped)
        y_next_pred_inverse = y_next_pred_inverse.astype(int)
        y_next_pred_inverse = y_next_pred_inverse.reshape(y_next_pred.shape)

        #y_next_pred = custom_predict(model,np.array([x_next]))
        print('Predicted Game #:', z)
        print('Prediction without rounding up:\t', y_next_pred_inverse[0][0])
        print('Prediction with rounding up:\t', y_next_pred_inverse[0][0]+1)
        print('-' * 40)

def run_prediction(drawingstopredict, windowsize):

    print('Generating ',drawingstopredict, ' predictions, based on ', windowsize, ' last numbers:')

    for z in range(1, drawingstopredict):
        next = df.copy()
        print('Predict the Future Drawing')
        next = next.tail(windowsize+ z - 1).head(windowsize)
        print('Window data:', next)
        lastrow = next.iloc[windowsize-1].tolist()
        print('last game out of ', windowsize, ' used in the prediction >>>', lastrow)
        next = np.array(next)
        #print(next)
        x_next = scaler.transform(next)
        y_next_pred = model.predict(np.array([x_next]))

        y_next_pred_reshaped = y_next_pred.reshape(-1, y_next_pred.shape[-1])
        y_next_pred_inverse = scaler.inverse_transform(y_next_pred_reshaped)
        y_next_pred_inverse = y_next_pred_inverse.astype(int)
        y_next_pred_inverse = y_next_pred_inverse.reshape(y_next_pred.shape)

        #y_next_pred = custom_predict(model,np.array([x_next]))
        print('Predicted Game #:', z)
        print('Prediction without rounding up:\t', y_next_pred_inverse[0][0])
        print('Prediction with rounding up:\t', y_next_pred_inverse[0][0]+1)
        print('Prediction with rounding down:\t', y_next_pred_inverse[0][0]-1)
        print('-' * 40)


def run_window_test(window_sample_size, previous_drawings):
    print('Calculating window-size ', window_sample_size,
          ' predictions, based on ', previous_drawings, ' last numbers:')

    mse_dict = {}
    nextw = df1.copy()
    #print('nextw shape out of loop', nextw.shape)

    for n in range(0, previous_drawings):

        last_row = nextw.iloc[-1]
        current_row = nextw.iloc[n - 1]
        print('Last game out of ', window_length,
              ' used in the prediction: ', last_row.tolist(),
              '. Current row: ', current_row.tolist())

        current_row_np = np.array(current_row)

        for w in range(2, window_sample_size):
            train_temp_window = nextw.tail(w+1)[:-1]
            print('Window size:', w)

            nextw_np = np.array(train_temp_window)
            x_nextw = scaler.transform(nextw_np)
            y_next_predw = model.predict(np.array([x_nextw]))

            pred = scaler.inverse_transform(y_next_predw).astype(int)[0]+1
            pred_round_up = scaler.inverse_transform(y_next_predw).astype(int)[0] + 1
            pred_round_down = scaler.inverse_transform(y_next_predw).astype(int)[0] - 1

            set1 = set(current_row.tolist())
            set2 = set(pred)

            intersection = len(set1.intersection(set2))
            mse_pred = intersection / float(len(set1) + len(set2) - intersection)

            print('Similarity: ', mse_pred)
            mse_dict[w] = mse_pred

            print('Predicted value: ', pred)
            print('Ground Truth: ', set1)

    # Find window size with minimum MSE
    if mse_dict:  # Checks if the dictionary is not empty
        ideal_window_size = max(mse_dict, key=mse_dict.get)
        print('Ideal window size based on MSE:', ideal_window_size)
    else:
        print("mse_dict is empty. Cannot calculate minimum.")
        ideal_window_size = None

    if mse_dict:  # Checks if the dictionary is not empty
        sorted_dict = dict(sorted(mse_dict.items(), key=lambda item: item[1], reverse=True))
        top_30_window_sizes = list(sorted_dict.keys())[:30]
        print('Top 30 window sizes based on MSE:', top_30_window_sizes )
    else:
        print("mse_dict is empty. Cannot calculate minimum.")
        top_30_window_sizes = None

    return ideal_window_size

#from tensorflow.keras.models import load_model
#model = load_model('/Users/tiagolee/PycharmProjects/MegaMillion/megamillion.keras')


run_prediction(10,window_length)
#run_prediction(3,70)

#run_window_test(10,1)
#detect_window_size(model,x_train, y_train)