##¿Qué es la regresión y cuál es su propósito en el Machine Learning? 
La regresión es una de las piedras angulares del aprendizaje supervisado en Machine Learning. Su propósito fundamental es predecir un valor numérico continuo a partir de una o más variables de entrada (también llamadas características o features).

A diferencia de otros enfoques que buscan clasificar elementos en categorías cerradas, la regresión intenta responder a preguntas cuantitativas como: ¿Cuánto?, ¿Qué precio? o ¿Qué cantidad?.

El objetivo principal de un algoritmo de regresión es encontrar la función matemática que mejor describa la relación entre las variables predictoras (X) y la variable de salida (Y). Una vez que el modelo aprende este patrón a partir de datos históricos, puede extrapolarlo para realizar predicciones precisas sobre datos completamente nuevos.

##Regresion lineal
##¿En qué se diferencia de otros tipos de modelos supervisados como la clasificación?
Aunque tanto la regresión como la clasificación son las dos grandes ramas del aprendizaje supervisado (lo que significa que ambos modelos entrenan con datos que ya tienen la respuesta correcta o "etiqueta"), se diferencian radicalmente en el tipo de problema que resuelven y en cómo miden su éxito.
La diferencia fundamental se puede resumir en una sola idea: la naturaleza de la variable que intentan predecir.
Regresión: Predice un valor numérico continuo. La respuesta es un número real dentro de un rango infinito. Puede haber infinitos valores posibles entre dos números cualesquiera.
Clasificación: Predice una etiqueta o categoría discreta. La respuesta pertenece a un conjunto finito y predefinido de clases. El modelo decide a cuál de esos "sacos" pertenece el dato.
	##¿Qué es la Regresión Lineal Simple y qué representa la ecuación ŷ = β0 + β1 ·X?
La Regresión Lineal Simple es una técnica estadística y de Machine Learning que se utiliza para modelar la relación entre dos variables:
1.	Una variable independiente o predictora (X), que es la que utilizamos para hacer la predicción.
2.	Una variable dependiente o de respuesta (Y), que es el valor numérico continuo que queremos predecir.
El adjetivo "Simple" se debe a que se utiliza una única variable predictora. El objetivo principal es encontrar la línea recta que mejor se ajuste a la nube de puntos de los datos históricos, permitiéndonos predecir el comportamiento de Y a partir de nuevos valores de X.

¿Qué representa la ecuación ŷ = β0 + β1
Esta ecuación es la fórmula matemática de la línea recta que el modelo calcula para realizar las estimaciones. Cada uno de sus componentes tiene un significado específico dentro del modelo:
•	ŷ : Representa la variable dependiente estimada o predicha. No es el valor real que está en el dataset, sino el valor teórico que el modelo calcula para un determinado valor de X.
•	X: Es la variable independiente o predictora. Es el dato de entrada que ya conocemos y que introducimos en la ecuación para obtener la predicción.
•	β0 (Beta cero / Intercepto): Es la ordenada al origen. Físicamente, representa el punto exacto donde la recta cruza el eje vertical (Y). Matemáticamente, es el valor que tomaría ŷ si la variable X fuera exactamente igual a cero (X = 0).
•	β1 (Beta uno / Pendiente): Es el coeficiente de regresión o la inclinación de la recta. Indica cuánto cambia mecánicamente el valor de ŷ por cada unidad que incrementa la variable X.
o	Si β1 es positivo, la recta sube (a mayor X, mayor Y).
o	Si β1 es negativo, la recta baja (a mayor X, menor Y).
