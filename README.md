# Clasificador de ingreso anual
## Descripción de proyecto
Este proyecto tiene el propósito de crear un modelo clasificador del potencial adquisitivo de la persona de acuerdo a factores como la edad, su estado civil, profesión, etc.
## Contenido del repositorio

## Descripción del dataset
**Nombre:** “Adult”

**Link a dataset:** https://archive.ics.uci.edu/dataset/2/adult

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
| education-num | Feature | Numérico | Education Level | Versión numérica de la variable “education” (ej. Bachelors = 13, HS-grad = 9, etc.). | No |
| marital-status | Feature | Categórico | Other | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. | No |
| occupation | Feature | Categórico | Other | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. | Sí |
| relationship | Feature | Categórico | Other | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. | No |
| race | Feature | Categórico | Race | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. | No |
| sex | Feature | Binario | Sex | Female, Male. | No |
| capital-gain | Feature | Numérico | - | Ganancias de capital. | No |
| capital-loss | Feature | Numérico | - | Pérdidas de capital. | No |
| hours-per-week | Feature | Numérico | - | Número de horas trabajadas por semana. | No |
| native-country | Feature | Categórico | Other | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. | Sí |
| income | Target | Binario | Income | >50K, <=50K. | No |

## Proceso
### Preprocesamiento de los datos antes del split de entrenamiento
Antes de realizar la división del dataset, se realizaron algunas técnicas de preprocesamiento. Las anteriores no representaron un "data leakage" debido a que <a href="https://pub.towardsai.net/data-leakage-in-machine-learning-why-you-must-split-before-preprocessing-3ddc3dcde4e9"><u>los cambios que no analizan relaciones entre los datos evitan que el modelo aprenda patrones o medidas que introduzcan un sesgo</u></a>. Las técnicas de preprocesamiento realizadas fueron las siguientes:
- Eliminación de la variable irrelevante "fnlwgt", debido a que describe el número de personas representadas por esa instancia, no una característica de la persona de la cual el modelo puede aprender.
- Eliminación de la variable duplicada “education-num”, ya que es una versión numérica de la variable “education”, por lo que utilizar ambas sería redundante y generaría multicolinealidad. La multicolinearidad es una condicion donde <a href="https://www.mdpi.com/2227-7390/10/8/1283#:~:text=2.,The%20authors%20of%20%5B4%5D."><u>una o dos variables independientes tienen una relación lineal, lo que afectaría a la interpretabilidad del modelo (como identificar las variables que tienen mayor impacto en la predicción)</u></a>.
- Eliminación de las instancias con valores faltantes en las variables "workclass", "occupation", "native-country"; ya que <a href="https://ursmaheshj.medium.com/effective-strategies-for-handling-missing-data-a215056a07e3"><u> sólo representan el 4.51% de las instancias totales del dataset</u></a>, por lo que la distribución no se alterará en gran medida y se disminuirá el tiempo de computo frente a otras técnicas como la imputación.
- Reemplazo de tipo de valor de la variable de string a integer (e.g "<=50K" a 0, y ">50K" a 1) por buena práctica, a pesar de que no tenga impacto en el desempeño del modelo.

### Split de entrenamiento
Se decidió usar el split del dataset de 80% train y 20% test debido al tamaño medio del dataset (48,842 instancias). Cabe destacar que el dataset se encuentra desbalanceado, es decir, existe una gran diferencia entre el porcentaje de instancias donde el ingreso es menor o igual a 50K (~76.07%) y donde es mayor a 50K (~23.92%), lo que causaría que <a href="https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e"><u> el método aleatorio de splitting por defecto no sea representativo </u></a> (con la posibilidad de que la clase minoría ni siquiera aparezca en el split de entrenamiento). Debido a esto se decidió garantizar que existiera la misma proporción entre estas clases tanto en el split de train como en el de test.

### Preprocesamiento de los datos después del split de entrenamiento
- Se normalizaron las variables numéricas para tener una varianza del 0 al 1
- Se aplicó One-hot Encoding a las variables categóricas para representarlos como vectores

## Referencias
- Ashraf, K. (2026, 12 febrero). Data Leakage in Machine Learning: Why You Must Split Before Preprocessing. Towards AI. Recuperado 8 de abril de 2026, de https://pub.towardsai.net/data-leakage-in-machine-learning-why-you-must-split-before-preprocessing-3ddc3dcde4e9
- Chan, J. Y.-L., Leow, S. M. H., Bea, K. T., Cheng, W. K., Phoong, S. W., Hong, Z.-W., & Chen, Y.-L. (2022). Mitigating the Multicollinearity Problem and Its Machine Learning Approach: A Review. Mathematics, 10(8), 1283. https://doi.org/10.3390/math10081283
- Jadhav, M. (2023, 4 junio). How to handle missing Data | Machine Learning | Data Science. Medium. Recuperado 8 de abril de 2026, de https://ursmaheshj.medium.com/effective-strategies-for-handling-missing-data-a215056a07e3
- Baldé, B. (2023, 13 abril). Why you should use stratified Split. Medium. Recuperado 9 de abril de 2026, de https://medium.com/@becaye-balde/why-you-should-use-stratified-split-bddb6dadd34e
