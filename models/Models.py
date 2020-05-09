from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import  Conv1D, Dense, Dropout, Embedding,ThresholdedReLU,Flatten, Lambda,  \
    MaxPooling1D,  Input, LSTM, Bidirectional
from tensorflow.keras import backend as K

def Endgame_model(max_features, max_len, use_gap=True):
    main_input = Input(shape=(75,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    lstm = LSTM(128, return_sequences=False)(embedding)
    drop = Dropout(0.5)(lstm)
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cmu_model(max_features, max_len, use_gap=True):
    main_input = Input(shape=(75,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    bi_lstm = Bidirectional(layer=LSTM(64, return_sequences=False), merge_mode='concat')(embedding)
    output = Dense(1, activation='sigmoid')(bi_lstm)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def nyu_model(max_features, max_len, use_gap=True):
    main_input = Input(shape=(75,), dtype='int32 ', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    conv1 = Conv1D(filters=128, kernel_size=3, padding='same', strides=1)(embedding)
    thresh1 = ThresholdedReLU(1e-6)(conv1)
    max_pool1 = MaxPooling1D(pool_size=2, padding='same')(thresh1)
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', strides=1)(max_pool1)
    thresh2 = ThresholdedReLU(1e-6)(conv2)
    max_pool2 = MaxPooling1D(pool_size=2, padding='same')(thresh2)
    flatten = Flatten()(max_pool2)
    fc = Dense(64)(flatten)
    thresh_fc = ThresholdedReLU(1e-6)(fc)
    drop = Dropout(0.5)(thresh_fc)
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss=' binary_crossentropy ', optimizer='adam', metrics=['accuracy'])
    return model


def invincea_model(max_features, max_len, use_gap=True):
    def getconvmodel(self, kernel_size, filters):
        model = Sequential()
        model.add(
            Conv1D(
                filters=filters, input_shape=(128, 128), kernel_size=kernel_size,
                padding='same', activation='relu', strides=1
            )
        )
        model.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(filters,)))
        model.add(Dropout(0.5))
        return model

    main_input = Input(shape=(75,), dtype='int32 ', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    conv1 = getconvmodel(2, 256)(embedding)
    conv2 = getconvmodel(3, 256)(embedding)
    conv3 = getconvmodel(4, 256)(embedding)
    conv4 = getconvmodel(5, 256)(embedding)
    merged = K.Concatenate()([conv1, conv2, conv3, conv4])
    middle = Dense(1024, activation=' relu ')(merged)
    middle = Dropout(0.5)(middle)
    middle = Dense(1024, activation=' relu ')(middle)
    middle = Dropout(0.5)(middle)
    output = Dense(1, activation='sigmoid')(middle)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss=' binary_crossentropy ', optimizer='adam', metrics=['accuracy'])
    return model


def mit_model(max_features, max_len, use_gap=True):
    main_input = Input(shape=(75,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', strides=1)(embedding)
    max_pool = MaxPooling1D(pool_size=2, padding='same')(conv)
    encode = LSTM(64, return_sequences=False)(max_pool)
    output = Dense(1, activation='sigmoid')(encode)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def crnn1d(max_features, max_len, use_gap=True):
    main_input = Input(shape=(75,), dtype='int32', name='main_input')
    embedding = Embedding(input_dim=128, output_dim=128, input_length=75)(main_input)
    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', strides=1)(embedding)
    max_pool = MaxPooling1D(pool_size=2, padding='same')(conv)
    bi_lstm = Bidirectional(layer=LSTM(64, return_sequences=False), merge_mode='concat')(max_pool)
    output = Dense(1, activation='sigmoid')(bi_lstm)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def import_test_mdl():
    print('you have imported Model successfully:)')