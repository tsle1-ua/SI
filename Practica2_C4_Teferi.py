#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Practica 2 C4 - Sistemas Inteligentes
Alumno: Teferi
Asignatura: Sistemas Inteligentes (Vision artificial y aprendizaje)
Convocatoria: 2024/2025
Fecha limite: 2024-06-01
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

USE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras import models, layers, utils, callbacks, regularizers
    USE_TF = True
except ImportError:
    print("TensorFlow no disponible. Algunas funciones no funcionaran.")

from PIL import Image

# ------------------------------ UTILIDADES ------------------------------

def cargar_y_preprocesar_cifar10():
    """Carga el dataset CIFAR-10 y devuelve los conjuntos de entrenamiento y 
    prueba normalizados junto con informacion de forma y numero de clases."""
    if not USE_TF:
        raise RuntimeError("TensorFlow es requerido para cargar CIFAR-10")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    input_shape = X_train.shape[1:]
    num_classes = 10
    return X_train, y_train, X_test, y_test, input_shape, num_classes

def plot_history(history, title='Model loss'):
    """Grafica la historia de entrenamiento de un modelo Keras."""
    if history is None:
        print('No history to plot')
        return
    plt.figure()
    plt.plot(history.history.get('loss', []), label='train')
    plt.plot(history.history.get('val_loss', []), label='val')
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def plot_bar_comparison(values, labels, title='Comparison', ylabel='accuracy'):
    """Grafica una comparativa de valores en barras."""
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """Muestra una matriz de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ------------------------------ TAREAS ------------------------------

def tarea_A(epochs=5, batch_size=32):
    """MLP simple para CIFAR-10."""
    if not USE_TF:
        print('TensorFlow requerido para tarea A')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Tarea A - Test accuracy: {acc:.4f}")
    return history

def tarea_B(epochs=50, batch_size=32, patience=3):
    """MLP con EarlyStopping."""
    if not USE_TF:
        print('TensorFlow requerido para tarea B')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    es = callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, callbacks=[es], verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Tarea B - Test accuracy: {acc:.4f}")
    return history

def tarea_C(batch_sizes=[32, 64, 128], epochs=5):
    """Comparacion de distintos batch_size."""
    if not USE_TF:
        print('TensorFlow requerido para tarea C')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    results = []
    for bs in batch_sizes:
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(32, activation='sigmoid'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results.append(acc)
        print(f"batch_size={bs} -> acc={acc:.4f}")
    plot_bar_comparison(results, [str(b) for b in batch_sizes], title='Batch size comparison')
    return results

def tarea_D(activaciones=['sigmoid', 'tanh', 'relu'], epochs=5, batch_size=32):
    """Comparacion de funciones de activacion."""
    if not USE_TF:
        print('TensorFlow requerido para tarea D')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    results = []
    for act in activaciones:
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(32, activation=act),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results.append(acc)
        print(f"activation={act} -> acc={acc:.4f}")
    plot_bar_comparison(results, activaciones, title='Activation comparison')
    return results

def tarea_E(neuronas=[16,32,64,128], epochs=5, batch_size=32):
    """Comparacion del numero de neuronas."""
    if not USE_TF:
        print('TensorFlow requerido para tarea E')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    results = []
    for n in neuronas:
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(n, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results.append(acc)
        print(f"neurons={n} -> acc={acc:.4f}")
    plot_bar_comparison(results, [str(n) for n in neuronas], title='Neurons comparison')
    return results

def tarea_F(configs=[[32],[64,32]], epochs=5, batch_size=32):
    """MLP con multiples capas ocultas."""
    if not USE_TF:
        print('TensorFlow requerido para tarea F')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    results = []
    for cfg in configs:
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        for units in cfg:
            model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results.append(acc)
        print(f"config={cfg} -> acc={acc:.4f}")
    plot_bar_comparison(results, [str(c) for c in configs], title='MLP layer config')
    return results

def tarea_G(epochs=5, batch_size=32):
    """CNN sencilla."""
    if not USE_TF:
        print('TensorFlow requerido para tarea G')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Tarea G - Test accuracy: {acc:.4f}")
    return history

def tarea_H(kernels=[(3,3),(5,5)], epochs=5, batch_size=32):
    """Comparacion de kernels en CNN."""
    if not USE_TF:
        print('TensorFlow requerido para tarea H')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    results = []
    for k in kernels:
        model = models.Sequential([
            layers.Conv2D(32, k, activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, k, activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results.append(acc)
        print(f"kernel={k} -> acc={acc:.4f}")
    plot_bar_comparison(results, [str(k) for k in kernels], title='Kernel size comparison')
    return results

def tarea_I(epochs=5, batch_size=32):
    """CNN optimizada con regularizacion y Dropout."""
    if not USE_TF:
        print('TensorFlow requerido para tarea I')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Tarea I - Test accuracy: {acc:.4f}")
    return history

def cargar_dataset_propio(path='dataset_propio', dims=(32,32)):
    """Carga un dataset propio organizado en carpetas por clase."""
    if not os.path.isdir(path):
        print(f"Directorio {path} no encontrado")
        return None, None
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))])
    X = []
    y = []
    for idx, cls in enumerate(classes):
        folder = os.path.join(path, cls)
        for file in os.listdir(folder):
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                img_path = os.path.join(folder, file)
                img = Image.open(img_path).resize(dims)
                X.append(np.array(img)/255.0)
                y.append(idx)
    X = np.array(X, dtype='float32')
    y = utils.to_categorical(y, len(classes)) if USE_TF else np.array(y)
    return (X, y), classes

def tarea_J(path='dataset_propio'):
    """Carga el dataset propio."""
    data, classes = cargar_dataset_propio(path)
    if data[0] is None:
        return None
    print(f"Dataset propio: {len(data[0])} imagenes, {len(classes)} clases")
    return data, classes

def tarea_K(model, data, classes):
    """Evalua un modelo sobre el dataset propio y muestra matriz de confusion."""
    if not USE_TF:
        print('TensorFlow requerido para tarea K')
        return None
    X, y = data
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    plot_confusion_matrix(y_true, y_pred, classes)

def tarea_L(epochs=5, batch_size=32):
    """CNN con data augmentation, BatchNormalization y EarlyStopping."""
    if not USE_TF:
        print('TensorFlow requerido para tarea L')
        return None
    X_train, y_train, X_test, y_test, input_shape, num_classes = cargar_y_preprocesar_cifar10()
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(X_train)
    es = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              epochs=epochs, validation_split=0.1, callbacks=[es], verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Tarea L - Test accuracy: {acc:.4f}")
    return model

# ------------------------------ MAIN ------------------------------

if __name__ == "__main__":
    start = time.time()
    print("Pruebas de humo para tareas A-L (1 epoch cada una)")
    if USE_TF:
        tarea_A(epochs=1)
        tarea_B(epochs=1)
        tarea_C(epochs=1)
        tarea_D(epochs=1)
        tarea_E(epochs=1)
        tarea_F(epochs=1)
        tarea_G(epochs=1)
        tarea_H(epochs=1)
        tarea_I(epochs=1)
        data_classes = tarea_J()
        if data_classes:
            data, classes = data_classes
            model = tarea_G(epochs=1)
            tarea_K(model, data, classes)
        tarea_L(epochs=1)
    else:
        print("TensorFlow no disponible, solo se prueban funciones basicas")
    end = time.time()
    print(f"Tiempo total de pruebas: {end-start:.2f}s")
