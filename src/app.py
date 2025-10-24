
"""
PROCEDIMIENTO
- Selección de variable objetivo (salud)
- EDA completo
- División train/test
- Baseline LinearRegression
- Lasso path (R^2 vs alpha) y comparación
- Optimización (GridSearchCV) y recomendaciones

"""


# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import os

# CRISP-DM CRISP-DM significa Cross Industry Standard Process for Data Mining, y es el estándar más usado en la industria para desarrollar proyectos de anális

###############   0. Business Understanding
#Se han recopilado datos socio demográficos y de recursos de salud por condado en los 
# Estados Unidos y queremos descubrir si existe alguna relación entre los recursos 
# sanitarios y los datos socio demográficos. 
# Para ello, es necesario que establezcas una variable objetivo (relacionada con la salud) 
# para llevar a cabo el análisis.

###############   1. Data Understanding

# Paso 1: Obtener Datos de CSV


#total_data = pd.read_csv("/workspaces/machinelearning/data/raw/airbnb2019.csv")  # pasar la data a un data fram

# si la voy a traer de la pagina
total_data = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=733&path=demographic_health_data.csv",sep=',')
total_data.to_csv("/workspaces/machine-learning-python-template/data/raw/total_data.csv", sep=',', index = False)

#Paso 2 Entender o explorar la data
print(total_data.head()) #ver rapidamente si cargo la info

print(total_data.shape)  # 108 columnas o variables y 3,140 filas o cantidad de registros
print("Columnas")
print(total_data.columns) # ver las columnas 
print('\nInfo resumen:')
print(total_data.info()) # ves tipos de datos, valores nulos y memoria usada, 
#todo de un vistazo. float64(61), int64(45), object(2)

print(total_data.describe().T) # ESTADISTICAS de cada columna, 

# PASO 3: LIMPIEZA

#3.1 DUPLICADOS

# Verifica si hay filas duplicadas (en todas las columnas)
duplicados = total_data.duplicated()

print("Duplicados:")
# Muestra cuántas filas duplicadas hay
print(duplicados.sum())
# 0

#3.2 Identificar y eliminar columnas irrelevantes

threshold = 0.5
interim_data = total_data.loc[:, total_data.isnull().mean() < threshold]

# Comprobar nulos
nulls = interim_data.isnull().sum().sort_values(ascending=False)
print('\nValores nulos por columna (top 30):')
print(nulls.head(30))

# Verificar valores nulos
print(interim_data.isnull().sum())

# Eliminar columnas con muchos nulos o irrelevantes
interim_data = interim_data.dropna(thresh=len(total_data)*0.7, axis=1)  # Columnas con >30% nulos
interim_data = interim_data.dropna()  # Filas con valores nulos

antes = len(total_data)
despues = len(interim_data)
print(f"Filas  eliminadas: {antes - despues}")

#---------------SELECCIONAR VARIABLE OBJETIVO

# Se evaluan el listado de las variables del excel y se seleccionan las siguientes de las cuales se visualizan sus datos

# Seleccionar las columnas objetivo
target_cols = [
    "anycondition_prevalence",
    "Obesity_prevalence",
    "Heart disease_prevalence",
    "COPD_prevalence",
    "diabetes_prevalence",
    "CKD_prevalence"
]

# Mostrar las primeras 5 filas de esas columnas
print(interim_data[target_cols].head())
print(interim_data[target_cols].describe())

#---------Se selecciona -------------anycondition_prevalence -----------
# como variable objetivo porque indica el estado general de salud en la poblacion  

#3.2 Identificar y eliminar columnas irrelevantes

#evaluar correlaciones 

# 3 Filtrar por correlación
numeric_df = interim_data.select_dtypes(include='number')
target = "anycondition_prevalence"

corr_with_target = numeric_df.corr()[target].sort_values(ascending=False)
selected_features = corr_with_target[abs(corr_with_target) > 0.4].index.tolist()
selected_features.remove(target)

#  Filtrar dataset
X = interim_data[selected_features]
y = interim_data[target]

print(f"Variables seleccionadas: {len(selected_features)}")
print(selected_features)

#32 variables

# ----- División Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Escalado de variables
#Como Lasso y Ridge penalizan los coeficientes, es fundamental escalar las variables:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelo Base Regresion Lineal
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Base Regresion lineal R2:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# Base Regresion lineal R2: 0.9998964559521065  Sospecha de sobreajuste (overfitting) 
#MAE: 0.051668251019068405
#RMSE: 0.06488087038664787



#----------Lasso Path (R² vs alpha)
alphas = np.logspace(-4, 1, 50)
r2_scores = []

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    r2_scores.append(r2_score(y_test, y_pred))

plt.figure(figsize=(8,5))
plt.plot(alphas, r2_scores, marker='o')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('Lasso Path (R² vs Alpha)')
plt.grid(True)
plt.show()

#--Optimización con GridSearchCV
from sklearn.model_selection import GridSearchCV

lasso = Lasso(max_iter=5000)
param_grid = {'alpha': np.logspace(-4, 1, 50)}

grid = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)

print("Optimizado con alpha Lasso")
print("Mejor alpha Lasso:", grid.best_params_)
print("Mejor R2:", grid.best_score_)

best_lasso = Lasso(alpha=grid.best_params_['alpha'], max_iter=5000)
best_lasso.fit(X_train_scaled, y_train)
y_pred_best = best_lasso.predict(X_test_scaled)

print("Test R2:", r2_score(y_test, y_pred_best))
print("MAE:", mean_absolute_error(y_test, y_pred_best))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best)))

#Filtrar variables derivadas del target 

# --------------------------
# 1️⃣ Eliminar variables derivadas del target (evitar data leakage)
# --------------------------
derived_cols = [col for col in X.columns if any(t.split('_')[0] in col for t in target_cols)]
print("Features eliminadas por derivación del target:", derived_cols)

X_clean = X.drop(columns=derived_cols)
selected_features_clean = X_clean.columns.tolist()
print("Variables finales para entrenamiento:", selected_features_clean)

# --------------------------
# 2️⃣ División Train/Test
# --------------------------
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

# --------------------------
# 3️⃣ Escalado
# --------------------------
scaler = StandardScaler()
X_train_scaled_c = scaler.fit_transform(X_train_c)
X_test_scaled_c = scaler.transform(X_test_c)

# --------------------------
# 4️⃣ Modelo Base LinearRegression
# --------------------------
lr_clean = LinearRegression()
lr_clean.fit(X_train_scaled_c, y_train_c)
y_pred_lr_c = lr_clean.predict(X_test_scaled_c)

print("LinearRegression limpio R2:", r2_score(y_test_c, y_pred_lr_c))
print("MAE:", mean_absolute_error(y_test_c, y_pred_lr_c))
print("RMSE:", np.sqrt(mean_squared_error(y_test_c, y_pred_lr_c)))

# --------------------------
# 5️⃣ Lasso Path
# --------------------------
alphas = np.logspace(-4, 1, 50)
r2_scores_c = []

for a in alphas:
    lasso = Lasso(alpha=a, max_iter=5000)
    lasso.fit(X_train_scaled_c, y_train_c)
    y_pred = lasso.predict(X_test_scaled_c)
    r2_scores_c.append(r2_score(y_test_c, y_pred))

plt.figure(figsize=(8,5))
plt.plot(alphas, r2_scores_c, marker='o')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('Lasso Path limpio (R² vs Alpha)')
plt.grid(True)
plt.show()

# --------------------------
# 6️⃣ GridSearchCV Lasso
# --------------------------
lasso = Lasso(max_iter=5000)
param_grid = {'alpha': np.logspace(-4, 1, 50)}

grid_c = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
grid_c.fit(X_train_scaled_c, y_train_c)

print("sin variables derivadas")

print("Mejor alpha Lasso limpio:", grid_c.best_params_)
print("Mejor R2 CV:", grid_c.best_score_)

best_lasso_c = Lasso(alpha=grid_c.best_params_['alpha'], max_iter=5000)
best_lasso_c.fit(X_train_scaled_c, y_train_c)
y_pred_best_c = best_lasso_c.predict(X_test_scaled_c)

print("Test R2 limpio:", r2_score(y_test_c, y_pred_best_c))
print("MAE limpio:", mean_absolute_error(y_test_c, y_pred_best_c))
print("RMSE limpio:", np.sqrt(mean_squared_error(y_test_c, y_pred_best_c)))

"""R² CV ≈ 0.768 y Test R² ≈ 0.778 → El modelo explica aproximadamente un 77–78% de la variabilidad 
en la prevalencia de condiciones de salud, sin depender de variables derivadas del target.

MAE ≈ 2.39 → En promedio, la predicción se desvía ~2.4 puntos porcentuales de la prevalencia real.

RMSE ≈ 3.0 → La desviación típica de los errores es de ~3 puntos porcentuales.

Sobreajuste mitigado:
Al inicio se tenia R² ≈ 0.9999, lo que era un indicio claro de overfitting. 
Ahora con variables “limpias”, el modelo refleja mejor la realidad y es generalizable.

Alpha Lasso moderado (0.0139):
Esta regularización es suficiente para penalizar coeficientes irrelevantes y mantener las variables más importantes, 
sin eliminar demasiada información.


"""

#--visualizar los coeficientes del Lasso limpio y generar un ranking de variables por importancia

# Obtener coeficientes y nombres de variables
coef = best_lasso.coef_
features = X_train.columns

# Crear un DataFrame para facilitar ordenamiento
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coef})

# Tomar el valor absoluto para ranking de importancia
coef_df['AbsCoefficient'] = np.abs(coef_df['Coefficient'])

# Ordenar por importancia
coef_df_sorted = coef_df.sort_values(by='AbsCoefficient', ascending=False)

# Mostrar las top 10 variables más influyentes
print("Top 10 variables más influyentes según Lasso:")
print(coef_df_sorted.head(10)[['Feature', 'Coefficient']])

# Graficar todos los coeficientes
plt.figure(figsize=(10,6))
plt.barh(coef_df_sorted['Feature'], coef_df_sorted['Coefficient'])
plt.xlabel('Coeficiente Lasso')
plt.title('Importancia de Variables según Lasso')
plt.gca().invert_yaxis()  # Invertir eje para mostrar la más influyente arriba
plt.show()

"""
anycondition_Upper 95% CI (3.538787) y anycondition_Lower 95% CI (2.833571)

Estas son los intervalos de confianza de la prevalencia general de condiciones de salud. 
Es lógico que sean muy influyentes porque directamente representan un rango de tu variable objetivo.

Obesity_Lower 95% CI (0.178905) y Obesity_Upper 95% CI (-0.174505)

El modelo considera la obesidad como relevante: un aumento en la parte inferior del 
CI de obesidad aumenta la prevalencia general, mientras que un aumento en la parte superior del CI tiene efecto contrario.

Percent of adults with a bachelor's degree or higher (-0.012611)

Un mayor porcentaje de adultos con educación universitaria se asocia a
 menor prevalencia de condiciones de salud (coeficiente negativo).

Percent of adults with less than a high school diploma (-0.005330)

Similar, aunque menor efecto; un mayor porcentaje de adultos con poca educación 
tiende a aumentar un poco la prevalencia (el signo negativo indica dirección del modelo tras considerar otras variables).

COPD_prevalence (0.004355), diabetes_prevalence (0.001813)

Prevalencia de enfermedades específicas que contribuyen ligeramente a la prevalencia general.

% Asian-alone (0.004333) y Unemployment_rate_2018 (0.001859)

Factores sociodemográficos que el modelo considera relevantes aunque con menor peso.
"""