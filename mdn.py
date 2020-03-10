
from keras.layers import Input, Dense, Lambda, RepeatVector, Flatten, Multiply
from keras.models import Model

from keras import backend as k
import numpy as np
import keras

def mixture_likelyhood(y, center, sigma, weights, numGaussians):
    #y shape: [batch_dim, 1]
    #center shape: [batch_dim, numGaussians]
    #sigma shape: [batch_dim, numGaussians]
    #weights shape: [batch_dim, numGaussians]
    y = RepeatVector(numGaussians)(y)
    y = Flatten()(y)
    error = Lambda(lambda inputs: inputs[0] - inputs[1], output_shape=lambda shapes:shapes[0])([y, center])
    inv_sigma = Lambda(lambda x: 1/(x + .00001))(sigma)
    scaled_error = Multiply()([error, inv_sigma])
    z = Multiply()([scaled_error, scaled_error])
    z = Lambda(lambda x: 1/ np.sqrt(2 * np.pi) * np.e **(-x/2))(z)
    normal = Multiply()([z, inv_sigma])
    
    weighted_normal = Multiply()([normal, weights])
    
    likelyhood = Lambda(lambda x: -k.log(k.sum(x, axis=1, keepdims=True)))(weighted_normal)
    
    return likelyhood
    
    
    
    
"""
def mixureDensityNetwork(numGaussians=3):
    inputs = Input((1,))
    
    truth = Input((1,))
    
    middleLayer = Dense(200, activation="relu")(inputs)
    middleLayer = Dense(200, activation="relu")(inputs)
    centers = Dense(numGaussians, activation="tanh")(middleLayer)
    sigmas = Lambda(lambda x: x + 1)(Dense(numGaussians, activation="tanh")(middleLayer))
    weights = Dense(numGaussians, activation="softmax")(middleLayer)
    
    likelyhood = mixture_likelyhood(truth, centers, sigmas, weights, numGaussians)
    
    train_model = Model([inputs, truth], [likelyhood])
    predict_model = Model([inputs], [centers, sigmas, weights])
    
    def mdn_loss(y_true, y_true_likelyhood):
        return k.mean(y_true_likelyhood)
    
    train_model.compile(keras.optimizers.adam(), mdn_loss)
    
    return train_model, predict_model

"""


def direct_minimize_loss(y_true, y_true_likelyhood):
    return k.mean(y_true_likelyhood)

def MDNModel(input_, output, numGaussians=20, numOutputDims=1, optimizer=keras.optimizers.adam()):
    assert numOutputDims == 1
    centers = Dense(numGaussians, activation="tanh")(output)
    sigmas = Lambda(lambda x: x + 1)(Dense(numGaussians, activation="tanh")(output))
    weights = Dense(numGaussians, activation="softmax")(output)
    
    truth = Input([numOutputDims])
    
    likelyhood = mixture_likelyhood(truth, centers, sigmas, weights, numGaussians)
    
    train_model = Model([input_, truth], [likelyhood])
    predict_model = Model([input_], [centers, sigmas, weights])
    

    
    train_model.compile(optimizer, direct_minimize_loss)
    
    return train_model, predict_model
    
def predict1D():
    inputs = Input((1,))
    
    middleLayer = Dense(200, activation="relu")(inputs)
    middleLayer = Dense(200, activation="relu")(middleLayer)
    
    return MDNModel(inputs, middleLayer)


def sample(x, predict_model):
    centers, sigmas, weights = predict_model.predict(np.array([[x]]))
    threshhold = np.random.random()
    cumsum = 0
    for i, weight in enumerate(weights[0]):
        cumsum += weight
        if cumsum > threshhold:
            break
    
    return np.random.normal(centers[0][i], sigmas[0, i])
from scipy import stats
def probabilityDensity(x, y, predict_model):
    centers, sigmas, weights = predict_model.predict(np.array([x]))
    density = 0
    for i, weight in enumerate(weights[0]):
        density += weight * stats.norm.pdf((y - centers[0][i]) / sigmas[0, i]) / sigmas[0, i]
    
    return density