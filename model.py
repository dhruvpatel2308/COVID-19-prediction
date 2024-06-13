from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=input_shape,kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu',padding='same',kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu',padding='same',kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu',padding='same',kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu',kernel_regularizer='l2'),
        #BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu',kernel_regularizer='l2'),
        #BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Assuming 3 classes
    ])

    return model
