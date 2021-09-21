import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

batch_size = 100
num_classes = 10

# Cantidad de veces que el modelo ve toda la data
epochs = 10

# Dimensiones de la imagen en px
filas, columnas = 28, 28

# Hacemos directamente el split
(xt, yt), (xtest, ytest) = mnist.load_data()

# Reshape
print(f'Original Shape Images xt: {xt.shape}')
print(f'Original Shape Labels yt: {yt.shape}')
print(f'Original Shape Images xtest: {xtest.shape}')
print(f'Original Shape Labels ytest: {ytest.shape}')

# E.g shape esperado: (60000, 28, 28, 1)
xt = xt.reshape(xt.shape[0], filas, columnas, 1)
xtest = xtest.reshape(xtest.shape[0], filas, columnas, 1)

print('-'*60)
print(f'New Shape Images xt: {xt.shape}')
print(f'New Shape Images xtest: {xtest.shape}')

print('*'*60)
print('\n')
print(f'Default type xt: {xt.dtype}')
print(f'Default type xtest: {xtest.dtype}')
xt = xt.astype('float32')
xtest = xtest.astype('float32')

print('-'*60)
print(f'New type xt: {xt.dtype}')
print(f'New type xtest: {xtest.dtype}')

print('*'*60)
print('\n')
print(f'Default max value xt: {xt.max()}')
print(f'Default max value xtest: {xtest.max()}')
xt = xt / 255
xtest = xtest / 255
print('-'*60)
print(f'Normalized max value xt: {xt.max()}')
print(f'Normalized max value xtest: {xtest.max()}')

yt = tensorflow.keras.utils.to_categorical(yt, num_classes)
ytest = tensorflow.keras.utils.to_categorical(ytest, num_classes)

modelo = Sequential()
modelo.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
modelo.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(68))
modelo.add(Dropout(0.25))
modelo.add(Dense(20))
modelo.add(Dropout(0.25))
modelo.add(Dense(num_classes, activation='softmax'))

modelo.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),
               metrics=['categorical_accuracy'])

modelo.fit(xt, yt, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))

puntuacion = modelo.evaluate(xtest, ytest, verbose=1)

print(puntuacion)
