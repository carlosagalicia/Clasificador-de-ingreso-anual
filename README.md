# Clasificador de ingreso anual
## Descripción de proyecto
Este proyecto tiene el propósito de crear un modelo clasificador del potencial adquisitivo de la persona de acuerdo a factores como la edad, su estado civil, profesión, etc.
## Contenido del repositorio

## Descripción del dataset
**Nombre:** “Adult”

**Links a dataset:** https://archive.ics.uci.edu/dataset/2/adult o https://www.kaggle.com/datasets/wenruliu/adult-income-dataset?resource=download

**Autores de dataset:** Barry Becker y Ronny Kohavi

**Objetivo:** Predecir si el ingreso anual de un individuo excede los 50.000 dólares.

**Características del dataset:** Dataset multivariable

**Número de variables:** 14

**Tipos de variables:** Variables numéricas (6), binarias (2),  categóricas (6) y target (1)

**Número de instancias:** 48,842

**¿Contiene valores faltantes?** Sí

### Tabla de variables

| Nombre | Rol | Tipo | Demográfico | Descripción | ¿Valores faltantes? |
|--------|-----|------|-------------|-------------|---------------------|
| age | Feature | Numérico | Age | Edad de la persona. | No |
| workclass | Feature | Categórico | Income | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. | Sí |
| fnlwgt | Feature | Numérico | - | Peso final asignado por el censo (cuántas personas de la población están representadas por este registro). | No |
| education | Feature | Categórico | Education Level | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. | No |
| educational-num | Feature | Numérico | Education Level | Versión numérica de la variable “education” (ej. Bachelors = 13, HS-grad = 9, etc.). | No |
| marital-status | Feature | Categórico | Other | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. | No |
| occupation | Feature | Categórico | Other | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. | Sí |
| relationship | Feature | Categórico | Other | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. | No |
| race | Feature | Categórico | Race | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. | No |
| gender | Feature | Binario | Gender | Female, Male. | No |
| capital-gain | Feature | Numérico | - | Ganancias de capital. | No |
| capital-loss | Feature | Numérico | - | Pérdidas de capital. | No |
| hours-per-week | Feature | Numérico | - | Número de horas trabajadas por semana. | No |
| native-country | Feature | Categórico | Other | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. | Sí |
| income | Target | Binario | Income | >50K, <=50K. | No |

### Correlación
No se observó una correlación fuerte entre las variables numéricas del dataset (máximo de 0.14) por lo que se decidió conservarlas para su posterior preprocesamiento:
<p align="center">
  <img src="https://github.com/user-attachments/assets/0b595b2f-615b-432b-acf4-21f772337a4d" alt="matriz de correlación" width="50%" />
  <br>
  <em> Matriz de correlación sobre las variables numéricas</em>
</p>


## Proceso
### Preprocesamiento de los datos antes del split de entrenamiento
Antes de realizar la división del dataset, se realizaron algunas técnicas de preprocesamiento. Las anteriores no representaron un "data leakage" debido a que <a href="https://pub.towardsai.net/data-leakage-in-machine-learning-why-you-must-split-before-preprocessing-3ddc3dcde4e9"><u>los cambios que no analizan relaciones entre los datos evitan que el modelo aprenda patrones o medidas que introduzcan un sesgo</u></a>. Las técnicas de preprocesamiento realizadas fueron las siguientes:
- Eliminación de la variable irrelevante "fnlwgt", debido a que describe el número de personas representadas por esa instancia, no una característica de la persona de la cual el modelo puede aprender.
- Eliminación de la variable duplicada “educational-num”, ya que es una versión numérica de la variable “education”, por lo que utilizar ambas sería redundante y generaría multicolinealidad. La multicolinearidad es una condición donde <a href="https://www.mdpi.com/2227-7390/10/8/1283#:~:text=2.,The%20authors%20of%20%5B4%5D."><u>una o dos variables independientes tienen una relación lineal, lo que afectaría a la interpretabilidad del modelo (como identificar las variables que tienen mayor impacto en la predicción)</u></a>.
- Imputación de los valores faltantes de las instancias en las variables categóricas "workclass", "occupation" y "native-country" con el valor "Unknown"; ya que <a href="https://ursmaheshj.medium.com/effective-strategies-for-handling-missing-data-a215056a07e3"><u> representan más del 13.2% de las instancias totales del dataset</u></a>, por lo que la distribución podría alterarse si sólo se eliminan.
- Reemplazo de tipo de valor de la variable de string a integer (e.g "<=50K" a 0, y ">50K" a 1) por buena práctica, a pesar de que no tenga impacto en el desempeño del modelo.

### Split de entrenamiento
Se decidió usar el split del dataset de 80% de train y 20% de test debido al tamaño medio del dataset (48,842 instancias). Cabe destacar que el dataset se encuentra desbalanceado, es decir, existe una gran diferencia entre el porcentaje de instancias donde el ingreso es menor o igual a 50K (~68.54%) y donde es mayor a 50K (~31.45%), lo que causaría que <a href="https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e"><u> el método aleatorio de splitting por defecto no sea representativo </u></a> (con la posibilidad de que la clase minoría ni siquiera aparezca en el split de entrenamiento). 
<p align="center">
  <img src="https://github.com/user-attachments/assets/6497fce8-a239-489c-85f8-d697090a8022" alt="distribucion de clases" width="50%" />
  <br>
  <em> Distribución de los datos en las clases de ingreso <=50K (0) y >50K (1)</em>
</p>

Debido a esto, mediante el [método de división estratificada "stratify"](https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e), se decidió garantizar que existiera la misma proporción entre estas clases tanto en el split de train como en el de test. Este método, que se utiliza al momento de dividir los datos de entrenamiento y prueba, garantiza que la proporción de las clases en el dataset original también se mantenga en las divisiones de entrenamiento y testing.

### Preprocesamiento de los datos después del split de entrenamiento
- Se normalizaron las variables numéricas para tener una varianza del 0 al 1 mediante [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) de sklearn.
- Se aplicó One-hot Encoding a las variables categóricas y a la variable binaria "gender" para que puedan ser procesadas por el modelo como arrays mediante [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) de sklearn.

Debido a que los modelos que se guardarán y cargarán posteriormente requieren recibir los datos transformados de la misma manera que durante el entrenamiento, es necesario que el proceso de preprocesamiento persista. Una alternativa sería reconstruir manualmente el preprocesador y volver a ajustarlo (fit) con los datos de entrenamiento. Sin embargo, esto requiere mayor esfuerzo computacional.

Ante esto, se decidió guardar el objeto del preprocesador en un archivo mediante "[joblib.dump](https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html#joblib.dump)(preprocessor, ruta_de_guardado)" para reutilizarlo directamente con "[joblib.load(ruta_de_guardado)](https://joblib.readthedocs.io/en/latest/generated/joblib.load.html)", evitando recalcular el proceso cuando el kernel se reinicia o se decide trabajar en otro momento y [siendo eficiente con el tamaño del dataset preprocesado](https://scikit-learn.org/stable/model_persistence.html).

### Implementación del modelo
De acuerdo en el artículo de investigación ["Predicting Annual Income of Individuals using Classification Techniques"](https://d1wqtxts1xzle7.cloudfront.net/119062941/695_report_2_-libre.pdf?1729547185=&response-content-disposition=inline%3B+filename%3DPredicting_Annual_Income_of_Individuals.pdf&Expires=1776033888&Signature=dNPtYHanu~lvL9OcCK~dQkJizOWOmcRZr~7TIKfdhINdVjJl1c4BAlv4ltwAEWomBsCQDT34Pd5VmYHD~e0cJAwV4zFD4iEafSfRabkLuZXWzKW1~ZHClIjIlc6fdUfT4kJmTpWJ8Z9xDe0de7QpQW8g5jEt8lwB9wRf1IQAU1haPX2JdasDt15NqjL87uvan9JyWRKmaaNmSp-DY3DtjO3HjeqD8j380kvBzII5WDsiTbIXuhGrlzR~H7fkhoVrNyI6pNXHMHMH7X7b9coo3NRfybYtmjMiyhNocTjCEVxdIEylqfYmzSUDe-amR2reYb4-GdPARzWc4xnw1Bku4Q__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) se seleccionó como modelo inicial un modelo de Red Neuronal Artificial (ANN) con la siguiente arquitectura:
- 2 capas ocultas (64 y 32 neuronas).
- Función de activación ReLU.
- Capa de salida con sigmoide.
- Función de pérdida binary crossentropy.
- Optimizador Adam.
- Batch size de 32.
- 18 épocas.

Por otro lado, para permitir la [reutilización de los modelos sin la necesidad de volver a entrenarlos](https://colab.research.google.com/drive/1qquddbZCV-ZjxAG6LY7kCLsMR1PlAiV8?authuser=1#scrollTo=2A8uucqrjDJY) se utilizaron los [métodos](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/) "model.save" (para guardar el modelo en formato .keras) y "tf.keras.models.load_model" (para instanciar el modelo guardado). Esto guarda la arquitectura y pesos para reutilizar el modelo en el mismo estado en el que fue entrenado.
Esto se llevó a cabo con el objetivo de evaluar los modelos y realizar predicciones en nuevas ejecuciones sin entrenamiento previo.

### Evaluación inicial del modelo
[Debido a que el dataset se encuentra desbalanceado, las métricas seleccionadas](https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf) y consideradas relevantes para evaluar el desempeño de este modelo fueron:
- Accuracy: Proporción de todas las clasificaciones que fueron correctas.
- Precision: La proporción de predicciones positivas verdaderas entre todas las predicciones positivas que detectó el modelo.
- Recall: La proporción de predicciones positivas verdaderas entre todos los positivos reales existentes.
- F1-Score: Medida que indica el balance entre Precision y Recall.

Debido a que estas métricas trabajan con la comparación entre las predicciones y los valores reales (TP, FP, TN, y FN), se consideró conveniente utilizar la matriz de confusión para identificar de una mejor manera los resultados, donde:
- TP: Número de personas cuyo ingreso anual es mayor a 50K dólares y fue clasificada como tal.
- FP: Número de personas cuyo ingreso anual es menor a 50K dólares pero fue clasificada como mayor a 50K dólares.
- TN: Número de personas cuyo ingreso anual es menor a 50K dólares y fue clasificada como tal.
- FN: Número de personas cuyo ingreso anual es mayor a 50K dólares pero fue clasificada como menor a 50K dólares.

#### Resultados obtenidos

Se graficaron los valores para "accuracy", "precision", "recall" y "loss" en los datos de entrenamiento y validación a lo largo de las épocas:
<img width="928" height="611" alt="image" src="https://github.com/user-attachments/assets/1ac5f664-3785-416b-a17e-657860308758" />

De acuerdo a las gráficas, se puede apreciar una desviación relativamente pequeña pero evidente sobre las medidas de entrenamiento y validación de las 4 métricas, lo que indica la presencia de overfitting. En el caso de "accuracy", se mejora en el entrenamiento, pero se mantiene estable durante la validación. Con respecto a "precision", se aumenta de forma estable en el entrenamiento; sin embargo, tiene un comportamiento inestable en la validación. En cuanto a "recall", se aumenta progresivamente con los datos de entrenamiento, pero es variable con los datos de validación, lo que sugiere que existe dificultad para detectar todos los casos positivos. Finalmente, durante el entrenamiento el "loss" disminuye, caso contrario a la validación.


| Métrica | Valor
|--------|-----|
|Loss|0.3308|
|Accuracy|0.8558|
|Precision|0.7394|
|Recall|0.6142|
|F1-score|0.6710|

Comparado con el artículo (con Accuracy de 0.860 y F1-score de 0.66), se puede concluir que el modelo inicial implementado reproduce el desempeño del modelo sugerido por el artículo, aunque con una ligera disminución en el Accuracy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/36e3bfb0-5aa4-4112-8318-03bdb88c32ab" alt="matriz de confusión" width="40%" />
  <br>
  <em> Matriz de confusión del modelo inicial</em>
</p>

De acuerdo a la matriz de confusión, el modelo tiende a detectar más falsos negativos que falsos positivos, es decir, casos donde la persona tiene un ingreso mayor a 50K dólares pero fue clasificada como menor a 50K dólares. Si se prioriza que el modelo detecte a todas las personas con ingreso mayor a 50K dólares (recall), entonces actualmente el modelo no detectó correctamente al 38.57% de personas en este segmento, por lo que es una gran área de oportunidad que también se puede observar en las gráficas de train/validation.

### Refinamiento del modelo
Se realizaron intentos para mejorar el F1-score y específicamente el recall del modelo mediante la experimentación con hiperparámetros como "batch size", "epochs", el tipo de optimizer,  métrica de error, etc. A pesar de esto, no se encontraron resultados o mejoras evidentes en el modelo hasta que se investigaron formas de reducir el overfitting, llamadas [métodos de regularización](https://www.ibm.com/think/topics/regularization#1580786321). Ante esto, se consideró mejor trabajar con estas técnicas.

### Métodos de regularización
La regularización consiste en técnicas que mejoran la generalización de un modelo (habilidad para generar predicciones precisas con nuevos datasets) a cambio de una disminución en el accuracy. Los métodos de regularización que se utilizaron y probaron fueron Early stopping (con una paciencia de 5 épocas) y Dropout de Keras (dos capas con un rate de 0.3 y 0.2).

[Early stopping](https://keras.io/api/callbacks/early_stopping/) es un callback que consiste en limitar el número de epochs durante el entrenamiento del modelo, procesando de forma continua los datos de entrenamiento y deteniéndose cuando no hay mejora o existe un deterioro en la métrica bajo la cual se evalúa el desempeño del modelo (pérdida en la validación en este caso). Se seleccionó esta técnica con el objetivo de alcanzar el mejor desempeño antes de que el valor de loss en la validación aumente. "patience" es el número de épocas durante las cuales el modelo espera para que la métrica mejore antes de detener el entrenamiento.

[Dropout](https://keras.io/api/layers/regularization_layers/dropout/) actúa eliminando de forma aleatoria nodos de las redes neuronales (incluyendo sus conexiones de entrada y salida) durante el entrenamiento. Esta técnica simula entrenar múltiples redes neuronales al entrenar diferentes subredes (cada una con diferentes nodos aleatorios excluidos) de la red original. Al final se utiliza la red completa sin Dropout para las pruebas, lo que equivale a hacer predicciones con un promedio de múltiples subredes. Se decidió utilizar este método para prevenir que las neuronas se sobreadapten a los datos en caso de que este fenómeno sea el causante principal del overfitting del modelo inicial. "rate" es el valor de 0 a 1 que representa la fracción de nodos de la red a eliminar.

A diferencia del guardado manual del modelo básico, se utilizó el callback de [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) con "save_best_only=True" para guardar automáticamente la mejor versión del modelo durante el entrenamiento, basada en la métrica de pérdida en la validación con monitor="val_loss". Esto permite reutilizar el mejor modelo sin la obligación de entrenarlo.

### Ajuste de threshold
[El threshold actúa como el umbral que define la decisión de clasificación del modelo](https://scikit-learn.org/stable/modules/classification_threshold.html). En este modelo de clasificación binaria, si la probabilidad de predicción del modelo es mayor al threshold, entonces se clasifica como clase positiva (1), de lo contrario, se clasifica como clase negativa (0). Se experimentó con el ajuste del threshold debido a que [el umbral por defecto de 0.5 no se considera apropiado al trabajar con datasets con clases desbalanceadas](https://developers.google.com/machine-learning/crash-course/classification/thresholding?hl=es-419). Esto causa que las predicciones de los modelos favorezcan a las clases mayoritarias y clasifiquen incorrectamente las clases minoritarias. 

Se iteró sobre un conjunto de thresholds de 0.1 a 0.91, clasificando la probabilidad de predicción del modelo con cada threshold y calculando su desempeño con la métrica F1-score. Finalmente se seleccionó el threshold (0.369) que retornara un mayor F1-score (0.696) para utilizarlo posteriormente en la evaluación final del modelo con el resto de métricas de interés.

### Evaluación del modelo refinado
#### Resultados obtenidos
Se graficaron los valores para "accuracy", "precision", "recall" y "loss" en los datos de entrenamiento y validación a lo largo de las épocas:
<img width="928" height="611" alt="image" src="https://github.com/user-attachments/assets/5233e103-e550-441d-8835-f1d4f6c3e7bd"/>

De acuerdo a las gráficas, se puede observar que todavía existen algunas fluctuaciones en el Recall y Precision a lo largo de los epochs; sin embargo, la diferencia train-validation entre Precision del modelo básico (~0.07) es mayor que la del modelo refinado (~0.01), lo que indica que el modelo refinado tiene un Precision más cercano ante nuevos datos. Al contrario, el Recall del modelo en la validación tiene una gran diferencia con el Recall en el entrenamiento, lo cual es esperado debido a que presenta tendencias inversamente proporcionales a Precision tanto en el modelo inicial como en el refinado.

Por otro lado se observan mejoras en el Accuracy y el Loss, donde ya no existe una gran diferencia entre las métricas de entrenamiento y validación a comparación de las diferencias del modelo inicial, lo que indica una reducción en las tendencias de sobreajuste del modelo (overfitting).

| Métrica | Valor
|--------|-----|
|Loss|0.310|
|Accuracy|0.851|
|Precision|0.681|
|Recall|0.712|
|F1-score|0.696|

En comparación con el modelo inicial, el modelo refinado presentó un aumento significativo en el Recall y F1-score a expensas de una disminución en el Precision y Accuracy, también logrando una disminución en el Loss

<p align="center">
  <img src="https://github.com/user-attachments/assets/734f7a70-7824-4a80-8658-9d9776c3a27d" alt="matriz de confusión" width="40%" />
  <br>
  <em> Matriz de confusión del modelo refinado</em>
</p>

De acuerdo a la matriz de confusión, el modelo ahora tiene la tendencia a detectar más falsos positivos que falsos negativos. La cantidad detectada de falsos negativos se redujo un 25,5% con respecto al modelo anterior. Esto representa una mejora, donde el modelo detecta a más personas con ingreso mayor a 50K dólares (recall) correctamente. Por otro lado, también se puede notar la disminución en la precisión en la matriz, donde se detectaron 274 falsos positivos adicionales en comparación con el modelo inicial.

## Conclusión
Se implementó un modelo de clasificación binaria para predecir si una persona tiene un ingreso mayor o menor a 50K dólares basado en aspectos como su clase de trabajo, nivel educativo, estado civil, edad, etc. Se realizaron técnicas para trabajar con un dataset desbalanceado como la división stratificada durante el split de entrenamiento y se desarrolló un modelo básico con una arquitectura derivada de una investigación previa, para posteriormente refinarlo.
El modelo refinado contó con mejoras en el Loss, Recall y F1-Score mediante el ajuste del umbral de clasificación de las probabilidades de predicción. Sin embargo, se presentó una disminución considerable en el Precision, lo que deja a interpretación del diseñador del modelo cuál es la métrica que tiene prioridad para la evaluación del modelo. Por otro lado, mediante la implementación de técnicas de regularización "Dropout" y "ModelCheckpoint" se logró reducir la presencia de overfitting. No obstante, se considera conveniente seguir optimizando el modelo para mejorar su desempeño y generalización ante nuevos datos.

## Referencias
- Ashraf, K. (2026, 12 de febrero). Data Leakage in Machine Learning: Why You Must Split Before Preprocessing. Towards AI. Recuperado 8 de abril de 2026, de https://pub.towardsai.net/data-leakage-in-machine-learning-why-you-must-split-before-preprocessing-3ddc3dcde4e9
- Chan, J. Y.-L., Leow, S. M. H., Bea, K. T., Cheng, W. K., Phoong, S. W., Hong, Z.-W., & Chen, Y.-L. (2022). Mitigating the Multicollinearity Problem and Its Machine Learning Approach: A Review. Mathematics, 10(8), 1283. https://doi.org/10.3390/math10081283
- Jadhav, M. (2023, 4 de junio). How to handle missing Data | Machine Learning | Data Science. Medium. Recuperado 8 de abril de 2026, de https://ursmaheshj.medium.com/effective-strategies-for-handling-missing-data-a215056a07e3
- Baldé, B. (2023, 13 de abril). Why you should use stratified Split. Medium. Recuperado 9 de abril de 2026, de https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e
- Shuvo, S., Mohanty, J., & Patel, D. (2024). Predicting Annual Income of Individuals using Classification Techniques. Recuperado 12 de abril de 2026, de https://d1wqtxts1xzle7.cloudfront.net/119062941/695_report_2_-libre.pdf?1729547185=&response-content-disposition=inline%3B+filename%3DPredicting_Annual_Income_of_Individuals.pdf&Expires=1776033888&Signature=dNPtYHanu~lvL9OcCK~dQkJizOWOmcRZr~7TIKfdhINdVjJl1c4BAlv4ltwAEWomBsCQDT34Pd5VmYHD~e0cJAwV4zFD4iEafSfRabkLuZXWzKW1~ZHClIjIlc6fdUfT4kJmTpWJ8Z9xDe0de7QpQW8g5jEt8lwB9wRf1IQAU1haPX2JdasDt15NqjL87uvan9JyWRKmaaNmSp-DY3DtjO3HjeqD8j380kvBzII5WDsiTbIXuhGrlzR~H7fkhoVrNyI6pNXHMHMH7X7b9coo3NRfybYtmjMiyhNocTjCEVxdIEylqfYmzSUDe-amR2reYb4-GdPARzWc4xnw1Bku4Q__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
- O. Olawale Awe, PhD. (n.d). Computational Strategies for Handling Imbalanced Data in Machine Learning, LISA 2020 Global Network, USA. https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf
- Baldé, B. (2023, 13 de abril). Why you should use stratified split. Medium. Recuperado 14 de abril de 2026, de https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e
- Scikit-learn. (s. f.). StandardScaler. Recuperado 18 de abril de 2026, de https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Scikit-learn. (s. f.). StandardScaler. Recuperado 18 de abril de 2026, de https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- Scikit-learn. (s. f.). Model persistence. Recuperado 18 de abril de 2026, de https://scikit-learn.org/stable/model_persistence.html
- Joblib. (s. f.). joblib.dump — joblib 1.5.3 documentation. Recuperado 18 de abril de 2026, de https://joblib.readthedocs.io/en/stable/generated/joblib.dump.html#joblib.dump
- Joblib. (s. f.). joblib.load — joblib 1.5.3 documentation. Recuperado 18 de abril de 2026, de https://joblib.readthedocs.io/en/latest/generated/joblib.load.html
- Webster, K. (2020, 15 de septiembre). Callbacks for saving models.ipynb. Google Colab. Recuperado 18 de abril de 2026, de https://colab.research.google.com/drive/1qquddbZCV-ZjxAG6LY7kCLsMR1PlAiV8?authuser=1#scrollTo=XJcHgGhWjDJZ
- Team, K. (s. f.). Keras documentation: Whole model saving & loading. Recuperado 18 de abril de 2026, de https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
- Ph.D., J. M., & Kavlakoglu, E. (2025, 17 de noviembre). What is regularization?. IBM. Recuperado 18 de abril de 2026, de https://www.ibm.com/think/topics/regularization#1580786321
- Team, K. (s. f.-a). Keras documentation: EarlyStopping. Recuperado 18 de abril de 2026, de https://keras.io/api/callbacks/early_stopping/
- Team, K. (s. f.-a). Keras documentation: Dropout layer. Recuperado 18 de abril de 2026, de https://keras.io/api/layers/regularization_layers/dropout/
- Team, K. (s. f.-c). Keras documentation: ModelCheckpoint. Recuperado 18 de abril de 2026, de https://keras.io/api/callbacks/model_checkpoint/
- Google. (s. f.). Umbrales y matriz de confusión. Google For Developers. Recuperado 19 de abril de 2026, de https://developers.google.com/machine-learning/crash-course/classification/thresholding?hl=es-419
- Scikit-learn. (s. f.). 3.3. Tuning the decision threshold for class prediction. Recuperado 19 de abril de 2026, de https://scikit-learn.org/stable/modules/classification_threshold.html
