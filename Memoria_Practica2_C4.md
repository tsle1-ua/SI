# Práctica 2 C4 – Sistemas Inteligentes (Visión artificial y aprendizaje)

## 1. Introducción
En esta práctica se trabaja con el dataset **CIFAR-10**, compuesto por imágenes a color de tamaño 32×32 píxeles distribuidas en 10 clases. El objetivo es aprender a construir modelos de reconocimiento de imágenes utilizando redes MLP y CNN en TensorFlow/Keras.

## 2. Objetivos
- Comprender el funcionamiento de las redes neuronales MLP y CNN.
- Ajustar parámetros como el tamaño de batch, funciones de activación y número de neuronas.
- Evaluar la capacidad de generalización de los modelos en un dataset propio.

## 3. Entorno y herramientas
- **Python 3**
- **TensorFlow/Keras** para la construcción de modelos.
- **NumPy** para operaciones numéricas.
- **Matplotlib** para gráficas.
- **scikit-learn** para métricas como la matriz de confusión.
- Editor recomendado: Thonny o VS Code.

## 4. Desarrollo
A continuación se describen brevemente las tareas implementadas.

### Tarea A
MLP con una capa oculta de 32 neuronas y activación *sigmoid* seguida de softmax. Se entrena con `optimizer='adam'` y `loss='categorical_crossentropy'`.

### Tarea B
Igual que la tarea A pero añadiendo la callback de *EarlyStopping* con paciencia configurable.

### Tarea C
Comparación de distintos valores de `batch_size` (`[32,64,128]`). Se muestran las accuracies obtenidas en el test.

### Tarea D
Comparación de activaciones `[sigmoid, tanh, relu]` en la capa oculta de un MLP.

### Tarea E
Comparación del número de neuronas `[16,32,64,128]` en la capa oculta.

### Tarea F
MLP de múltiples capas ocultas con configuraciones variadas (por ejemplo `[32]`, `[64,32]`).

### Tarea G
CNN sencilla con dos bloques `Conv2D + MaxPool` seguida de capas densas.

### Tarea H
Comparación de distintos valores de `kernel_size` para las convoluciones: `3x3` y `5x5`.

### Tarea I
CNN optimizada que incluye regularización L2 y *Dropout* para reducir el sobreajuste.

### Tarea J
Función `cargar_dataset_propio` que lee las imágenes de `/dataset_propio` y las redimensiona a 32×32. Se asume que existen 10 carpetas con 15 imágenes cada una.

### Tarea K
Evaluación de la generalización en el dataset propio mostrando la matriz de confusión.

### Tarea L
CNN con *data augmentation*, `BatchNormalization` y *EarlyStopping*.

Cada una de estas tareas cuenta con funciones implementadas en `Practica2_C4_Teferi.py` y se pueden ejecutar de manera independiente desde el `main` como prueba de humo (1–2 epochs).

## 5. Resultados y discusión
Los resultados variarán según el número de épocas y los parámetros utilizados. De manera general, las CNN ofrecen mejores accuracies que las MLP al trabajar con imágenes. Las técnicas de regularización y *data augmentation* mejoran la capacidad de generalización.

## 6. Conclusiones
Se han explorado diferentes configuraciones de MLP y CNN. Las mejores configuraciones se obtuvieron con CNN que emplean normalización, regularización y *data augmentation*. Además, la elección de funciones de activación y el número de neuronas influye de manera notable en la convergencia del modelo.

## 7. Bibliografía
- Apuntes de la asignatura Sistemas Inteligentes C4.
- Documentación de [TensorFlow](https://www.tensorflow.org/).
- Documentación de [Keras](https://keras.io/).
- Documentación de [scikit-learn](https://scikit-learn.org/).
- Consultas realizadas mediante ChatGPT.
