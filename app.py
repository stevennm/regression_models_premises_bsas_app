import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.title('Predicción de precios para locales en CABA - 2020')

# carga de datos y limpieza
df = pd.read_csv(
    'https://cdn.buenosaires.gob.ar/datosabiertos/datasets/secretaria-de-desarrollo-urbano/locales-en-venta/locales-en-venta-2020.csv')
df.columns = map(str.lower, df.columns)
df.rename(columns={"propiedads": "m2",
                   "en_galeria": "galeria",
                   "cotiz_": "dolar",
                   "barrios_": "barrios",
                   "comuna_": "comuna",
                   },
          inplace=True)
pd.set_option('display.float_format', lambda x: '%.0f' % x)
df = df.drop(axis=1, columns=['direccion', 'preciopeso', 'pesosm2', 'dolar', 'trimestre_'])
df.galeria.fillna('NO', inplace=True)
df.galeria.replace(to_replace='S1', value='SI', inplace=True)

# ajuste de data
df_ajustada = df[df.preciousd < df.preciousd.quantile(0.99)]

if st.checkbox('Mostrar dataset en tabla'):
    st.dataframe(df_ajustada)

if st.checkbox('Mostrar relación entre el precio y las otras variables'):
    checked_variable = st.selectbox(
        'Seleccione una variable:',
        ('m2', 'barrios', 'antig')
    )
    # Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x=df_ajustada[checked_variable], y=df_ajustada['preciousd'])
    plt.xlabel(checked_variable)
    plt.xticks(fontsize=6, rotation=90)
    plt.ylabel("Precios")
    st.pyplot(fig)

# separamos data
y = np.array(df_ajustada.preciousd)
x = df_ajustada.drop(['preciousd'], axis=1)

'''
### Generamos dummies para variables categóricas
'''

x = pd.get_dummies(data=x)

if st.checkbox('Mostrar data:'):
    st.dataframe(x)

"""
### Separamos en train y test y estandarizamos
"""

left_column, right_column = st.columns(2)

# test size
test_size = left_column.number_input(
    'Tamaño del test dataset (0.0-1.0):',
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
)

# random_seed
random_seed = right_column.number_input('Setear random state (1-):',
                                        value=42, step=1,
                                        min_value=1)

# separamos dataset
xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size=test_size,
    random_state=random_seed
)

# estandarizo
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(xtrain)
xtrain_sc = scaler.transform(X=xtrain)
xtest_sc = scaler.transform(X=xtest)

if st.checkbox('Mostrar data estandarizada:'):
    st.dataframe(xtrain_sc)
    st.dataframe(xtest_sc)

if st.button('Entrenar modelos de regresión y mostrar resultados'):
    '''
    ### K-nearest Neighbours Regression:
    '''
    # knn regressor
    knn_reg = KNeighborsRegressor(weights="distance")
    param_knn = [{'n_neighbors': np.arange(10, 41, 5)}]
    regressor_knn = GridSearchCV(estimator=knn_reg, param_grid=param_knn, n_jobs=-1, refit=True, cv=5)
    regressor_knn.fit(xtrain_sc, ytrain)
    st.write('Los mejores parámetros fueron: ' + str(regressor_knn.best_params_))

    ypred_knnr = regressor_knn.predict(xtest_sc)
    st.write('El error cuadrático medio es: ' + str(np.sqrt(mean_squared_error(ytest, ypred_knnr))))

    '''
    ### Support Vector Regression:
    '''
    # sv regressor
    sv_reg = SVR()
    param_svr = {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [1, 10], 'gamma': [0.01, 1]}
    regressor_svr = GridSearchCV(estimator=sv_reg, param_grid=param_svr, n_jobs=-1, refit=True, cv=5)
    regressor_svr.fit(xtrain_sc, ytrain)
    st.write('Los mejores parámetros fueron: ' + str(regressor_svr.best_params_))

    ypred_svr = regressor_svr.predict(xtest_sc)
    st.write('El error cuadrático medio es: ' + str(np.sqrt(mean_squared_error(ytest, ypred_svr))))

    '''
    ### Random Forest Regression:
    '''
    # rf regressor
    rf_reg = RandomForestRegressor()
    param_rfr = {'n_estimators': [10, 200],
                 'max_features': [0.5, 'sqrt', 'log2'],
                 'min_samples_leaf': [1, 5]}
    regressor_rf = GridSearchCV(estimator=rf_reg, param_grid=param_rfr, refit=True, cv=5)
    regressor_rf.fit(xtrain, ytrain)
    st.write('Los mejores parámetros fueron: ' + str(regressor_rf.best_params_))

    ypred_rfr = regressor_rf.predict(xtest_sc)
    st.write('El error cuadrático medio es: ' + str(np.sqrt(mean_squared_error(ytest, ypred_rfr))))

    ''' ### Diferencias entre ytest e ypredicted'''
    fig1, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].plot(ytest - ypred_knnr, marker='o', linestyle='')
    ax[1].plot(ytest - ypred_svr, marker='o', linestyle='')
    ax[2].plot(ytest - ypred_rfr, marker='o', linestyle='')
    st.pyplot(fig1)

    ''' ### Predicciones vs Valores reales'''
    fig2, ax = plt.subplots(1, 3, figsize=(14, 5))
    ax[0].scatter(ytest, ypred_knnr)
    ax[0].plot(np.arange(np.max(ytest)), np.arange(np.max(ytest)), color='crimson', alpha=0.5)
    ax[0].set_title('KNN Regressor')
    ax[0].set_xlabel('Valores reales')
    ax[0].set_ylabel('Predicciones')

    ax[1].scatter(ytest, ypred_svr)
    ax[1].plot(np.arange(np.max(ytest)), np.arange(np.max(ytest)), color='crimson', alpha=0.5)
    ax[1].set_title('SV Regressor')
    ax[1].set_xlabel('Valores reales')
    ax[1].set_ylabel('Predicciones')

    ax[2].scatter(ytest, ypred_rfr)
    ax[2].plot(np.arange(np.max(ytest)), np.arange(np.max(ytest)), color='crimson', alpha=0.5)
    ax[2].set_title('RFR')
    ax[2].set_xlabel('Valores reales')
    ax[2].set_ylabel('Predicciones')

    st.pyplot(fig2)