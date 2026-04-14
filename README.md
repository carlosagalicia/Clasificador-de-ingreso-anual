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
- Se normalizaron las variables numéricas para tener una varianza del 0 al 1
- Se aplicó One-hot Encoding a las variables categóricas y a la variable binaria "gender" para que puedan ser procesadas por el modelo como arrays.

### Implementación del modelo
De acuerdo en el artículo de investigación ["Predicting Annual Income of Individuals using Classification Techniques"](https://d1wqtxts1xzle7.cloudfront.net/119062941/695_report_2_-libre.pdf?1729547185=&response-content-disposition=inline%3B+filename%3DPredicting_Annual_Income_of_Individuals.pdf&Expires=1776033888&Signature=dNPtYHanu~lvL9OcCK~dQkJizOWOmcRZr~7TIKfdhINdVjJl1c4BAlv4ltwAEWomBsCQDT34Pd5VmYHD~e0cJAwV4zFD4iEafSfRabkLuZXWzKW1~ZHClIjIlc6fdUfT4kJmTpWJ8Z9xDe0de7QpQW8g5jEt8lwB9wRf1IQAU1haPX2JdasDt15NqjL87uvan9JyWRKmaaNmSp-DY3DtjO3HjeqD8j380kvBzII5WDsiTbIXuhGrlzR~H7fkhoVrNyI6pNXHMHMH7X7b9coo3NRfybYtmjMiyhNocTjCEVxdIEylqfYmzSUDe-amR2reYb4-GdPARzWc4xnw1Bku4Q__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) se seleccionó como modelo inicial un modelo de Red Neuronal Artificial (ANN) con la siguiente arquitectura:
- 2 capas ocultas (64 y 32 neuronas).
- Función de activación ReLU.
- Capa de salida con sigmoide.
- Función de pérdida binary crossentropy.
- Optimizador Adam.
- Batch size de 32.
- 18 épocas.

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

De acuerdo a la matriz de confusión, el modelo tiende a detectar más falsos negativos, es decir, casos donde la persona tiene un ingreso mayor a 50K dólares pero fue clasificada como menor a 50K dólares. Si se prioriza que el modelo detecte a todas las personas con ingreso mayor a 50K dólares (recall), entonces actualmente el modelo no detectó correctamente al 38.57% de personas en este segmento, por lo que es una gran área de oportunidad que también se puede observar en las gráficas de train/validation.

## Referencias
- Ashraf, K. (2026, 12 febrero). Data Leakage in Machine Learning: Why You Must Split Before Preprocessing. Towards AI. Recuperado 8 de abril de 2026, de https://pub.towardsai.net/data-leakage-in-machine-learning-why-you-must-split-before-preprocessing-3ddc3dcde4e9
- Chan, J. Y.-L., Leow, S. M. H., Bea, K. T., Cheng, W. K., Phoong, S. W., Hong, Z.-W., & Chen, Y.-L. (2022). Mitigating the Multicollinearity Problem and Its Machine Learning Approach: A Review. Mathematics, 10(8), 1283. https://doi.org/10.3390/math10081283
- Jadhav, M. (2023, 4 junio). How to handle missing Data | Machine Learning | Data Science. Medium. Recuperado 8 de abril de 2026, de https://ursmaheshj.medium.com/effective-strategies-for-handling-missing-data-a215056a07e3
- Baldé, B. (2023, 13 abril). Why you should use stratified Split. Medium. Recuperado 9 de abril de 2026, de https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e
- Shuvo, S., Mohanty, J., & Patel, D. (2024). Predicting Annual Income of Individuals using Classification Techniques. Recuperado 12 de abril de 2026, de https://d1wqtxts1xzle7.cloudfront.net/119062941/695_report_2_-libre.pdf?1729547185=&response-content-disposition=inline%3B+filename%3DPredicting_Annual_Income_of_Individuals.pdf&Expires=1776033888&Signature=dNPtYHanu~lvL9OcCK~dQkJizOWOmcRZr~7TIKfdhINdVjJl1c4BAlv4ltwAEWomBsCQDT34Pd5VmYHD~e0cJAwV4zFD4iEafSfRabkLuZXWzKW1~ZHClIjIlc6fdUfT4kJmTpWJ8Z9xDe0de7QpQW8g5jEt8lwB9wRf1IQAU1haPX2JdasDt15NqjL87uvan9JyWRKmaaNmSp-DY3DtjO3HjeqD8j380kvBzII5WDsiTbIXuhGrlzR~H7fkhoVrNyI6pNXHMHMH7X7b9coo3NRfybYtmjMiyhNocTjCEVxdIEylqfYmzSUDe-amR2reYb4-GdPARzWc4xnw1Bku4Q__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
- O. Olawale Awe, PhD. (n.d). Computational Strategies for Handling Imbalanced Data in Machine Learning, LISA 2020 Global Network, USA. https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf
- Baldé, B. (2023b, abril 13). Why you should use stratified split. Medium. Recuperado 14 de abril de 2026, de https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e
