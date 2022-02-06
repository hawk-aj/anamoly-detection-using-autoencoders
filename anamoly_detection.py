import streamlit as st

train_area = st.empty()

"""

# Anamoly Detection using Autoencoders

In this demo we use a basic autoencoder architecture to perform anamoly detection on ECG dataset, the dataset comprises of various ecgs, each with 140 data points
## Let's first take a look at imports

"""

with st.echo():

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, losses

"""

## Loading the Dataset

We will load the ecg dataset. 

"""

with st.echo():

    dataframe = pd.read_csv('C:/Users/aj240/Downloads/ecg.csv', header=None)
    raw_data = dataframe.values
    dataframe.head()
    # The last element contains the labels
    labels = raw_data[:, -1]
    # The other data points are the electrocadriogram data
    data = raw_data[:, 0:-1]

"""

This will load the entire data in the `data` and 'labels' variables as you can see below

"""

st.subheader('Input Features')

data[:5]

st.subheader('Output Labels')

labels[:5]

"""

## Splitting the data into Train and Test sets

Here we split our data into train and test sets

"""

with st.echo():

    from sklearn.model_selection import train_test_split

    train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21)

"""

The `train_test_split()` function splits the data into 2 sets where the test set is 25% of the total dataset. We have used the same function again on the train_full to split it into train and validation sets. 25% is a default parameter and you can tweak it as per your needs. Take a look at it from the [Scikit-Learn's Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

## Taking a look at the train data


"""

st.write(train_data[:5])

"""
## Preprocessing

Here we normalize the data before forwarding it to the autoencoder

"""

with st.echo():

    #normalization
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

"""
#the model will be trained to recognize the rythms classified as 1, the labels are thus converted to bool
"""

with st.echo():
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

"""
*Normal ecg*
"""
st.line_chart(pd.DataFrame(normal_train_data[0]))

"""
Abnormal ecg
"""
st.line_chart(pd.DataFrame(anomalous_train_data[0]))

"""
## Creating a model

"""

st.sidebar.title('Hyperparameters')

n_neurons_1 = st.sidebar.slider('Neurons encoder layer 1', 1, 128, 8)

n_neurons_2 = st.sidebar.slider('Neurons encoder layer 2', 1, 64, 8)

n_neurons_3 = st.sidebar.slider('Neurons encoder layer 3', 1, 32, 8)

n_neurons_4 = st.sidebar.slider('Neurons decoder layer 1', 1, 32, 8)

n_neurons_5 = st.sidebar.slider('Neurons decoder layer 2', 1, 64, 8)

# l_rate = st.sidebar.selectbox('Learning Rate', (0.0001, 0.001, 0.01), 1)

n_epochs = st.sidebar.number_input('Number of Epochs', 1, 50, 20)

#The n_neurons, l_rate, and _nepochs are the inputs taken from the user for training the model. The default values for them are also set. Default value for n_neurons is 30, the default value for l_rate is 0.01 and the default value for n_epochs is 20. So at the beginning the model will have 30 neurons in the first layer, the learning rate will be 0.01 and the number of epochs for which the model will train for is 20. 

with st.echo():

    import tensorflow as tf
    from tensorflow.keras.models import Model
    
    class AnomalyDetector(Model):
        def __init__(self):
            super(AnomalyDetector, self).__init__()
            self.encoder = tf.keras.Sequential([
            layers.Dense(n_neurons_1, activation="relu"),
            layers.Dense(n_neurons_2, activation="relu"),
            layers.Dense(n_neurons_3, activation="relu")])

            self.decoder = tf.keras.Sequential([
            layers.Dense(n_neurons_4, activation="relu"),
            layers.Dense(n_neurons_5, activation="relu"),
            layers.Dense(140, activation="sigmoid")])

        def call(self, x):
            encoded = self.encoder(x)
            # self.encoded = encoded
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = AnomalyDetector()

"""

## Compiling the model

Tensorflow keras API provides us with the `model.compile()` function to assign the optimizers, loss function and a few other details for the model.

"""

with st.echo():

    autoencoder.compile(optimizer='adam', loss='mae')

"""

## Training the model

In order to train the model you simply have to call the `fit()` function on the model with training and validation set and a number of epochs you want the model to train for.

**Try playing with the hyperparameters from the sidebar on the left side and click on the `Train Model` button given below to start the training.**

"""

train = st.checkbox('Train Model')

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

if train:

    with st.spinner('Training Model…'):

        # with st.echo():

        # model.summary(print_fn=lambda x: st.write("{}".format(x)))

        history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=n_epochs, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)

        st.success('Model Training Complete!')

        """

        ## Model Performance

        """

        # with st.echo():

        st.line_chart(pd.DataFrame(history.history))

        encoded_data = autoencoder.encoder(normal_test_data).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()

        plt.plot(normal_test_data[0], 'b')
        plt.plot(decoded_data[0], 'r')
        plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
        plt.legend(labels=["Input", "Reconstruction", "Error"])
        st.pyplot(plt)

        """"""
        reconstructions = autoencoder.predict(normal_train_data)
        train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
        threshold = np.mean(train_loss) + np.std(train_loss)
        reconstructions = autoencoder.predict(anomalous_test_data)
        test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

        """
        > This loss on the test set is a little worse than that on the validation set, which is as expected, as the model has never seen the images from the test set.

        ## Predictions using the Model

        """

        # with st.echo():

        def predict(model, data, threshold):
            reconstructions = model(data)          #calls the call funciton of the autoencoder class that returns the result of the decoder
            loss = tf.keras.losses.mae(reconstructions, data)
            return tf.math.less(loss, threshold)   #return true if loss is less than the threshold
        
        preds = predict(autoencoder, test_data, threshold)
        
        st.write("Accuracy = {}".format(accuracy_score(test_labels, preds))
        ,"Precision = {}".format(precision_score(test_labels, preds))
        ,"Recall = {}".format(recall_score(test_labels, preds)))
        

        """

        ### Predictions

        """

        a = tf.make_tensor_proto(preds[10:15])
        a = tf.make_ndarray(a)
        a
        """

        ### Ground Truth

        """

        test_labels[10:15]