import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from apyori import apriori
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.tree import plot_tree

import streamlit as st

def algoritmoEDA(archivo):
	with EDA:
		st.title("**Análisis Exploratorio de Datos**")
		st.write("Una buena práctica, antes de mirar los datos, es hacer un análisis de éstos para resumir sus principales características, a menudo con métodos visuales. El análisis exploratorio de datos, o EDA, implica conocer los datos.")
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			st.subheader("Paso 1: Descripción de la estructura de datos")
			st.markdown("Forma de la matriz (filas, columnas): ") 
			st.text(archivo.shape)
			st.markdown("Tipos de datos: ") 
			st.text(archivo.dtypes)

			st.subheader("Paso 2: Identificación de datos faltantes")
			col1, col2=st.columns(2)
			col1.markdown("Cantidad de datos nulos")
			col1.text(archivo.isnull().sum())
			#col2.markdown("Cantidad de datos NO nulos")
			#col2.table(archivo.info())

			st.subheader("Paso 3: Detección de valores atípicos.")
			st.markdown("Distribución de variables numéricas")
			st.text("Se utilizan histogramas que agrupan los números en rangos. La altura de una barra muestra cuántos números caen en ese rango.")
			archivo.hist(figsize=(14,14), xrot=45)
			st.pyplot(plt.show())
			st.markdown("Resumen estadístico de variables numericas")
			st.table(archivo.describe())
			st.markdown("Distribución de variables categóricas")
			for col in archivo.select_dtypes(include='object'):
				if archivo[col].nunique()<10:
					sns.countplot(y=col, data=archivo)
				st.pyplot(plt.show())

			st.subheader("Paso 4: Identificación de relaciones entre pares de variables")
			plt.figure(figsize=(14,14))
			sns.heatmap(archivo.corr(), cmap='RdBu_r', annot=True)
			st.pyplot(plt.show())
	return

def ACD(archivo):
	with analisisCD:
		st.title("Análisis de Correlacional Datos")
		#Aqui va la descripcion de ACD
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			st.subheader("Paso 1: Tipos de Datos")
			st.text(archivo.dtypes)

			st.subheader("Paso 2: Identificación de datos faltantes")
			st.markdown("Cantidad de datos nulos")
			st.text(archivo.isnull().sum())

			st.subheader("Paso 3: Exploración visual")
			var=st.selectbox("Variable", archivo.columns)
			sns.pairplot(archivo, hue=var)
			st.pyplot(plt.show())

			st.subheader("Paso 4: Obtención de la matriz de correlación")
			st.table(archivo.corr())
			plt.figure(figsize=(14,7))
			matrizInf=np.triu(archivo.corr())
			sns.heatmap(archivo.corr(), cmap='RdBu_r', annot=True, mask=matrizInf)
			st.pyplot(plt.show())
	return

def ACP(archivo):
	with analisisCP:
		st.title("Análisis de Componentes Principales")
		#Descripcion de ACP
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			st.subheader("Paso 1: Estandarización de los datos")
			normalizar=StandardScaler()
			normalizar.fit(archivo)
			MNormalizada = normalizar.transform(archivo)
			st.markdown("Matriz normalizada")
			st.dataframe(pd.DataFrame(MNormalizada, columns=archivo.columns))

			st.subheader("Pasos 2 y 3: Calcular la matriz de covarianzas o correlaciones, calcular los componentes (eigen-vectores) y la varianza (eigen-valores).")
			pca=PCA(n_components=10)
			pca.fit(MNormalizada)
			st.text(pca.components_)

			st.subheader("Paso 4: Decidir el numero de componentes")
			plt.plot(np.cumsum(pca.explained_variance_ratio_))
			plt.xlabel("Numero de componentes")
			plt.ylabel("Varianza acumulada")
			plt.grid()
			st.pyplot(plt.show())
			st.text("El numero de componentes debe contener entre el 75 y 90%  de varianza total")
			numComp=st.selectbox("Numero de componentes", options=[0,1,2,3,4,5,6,7,8,9,10])
			Varianza=pca.explained_variance_ratio_
			st.write('Proporción de varianza: ', Varianza)
			st.write("Varianza acumulada: ", sum(Varianza[0:numComp]))

			st.subheader("Paso 5: Examinar la proporción de relevancias -cargas-")
			st.dataframe(pd.DataFrame(abs(pca.components_), columns=archivo.columns))
	return

def cJerarquico(archivo):
	with jerarquico:
		st.title("Clusterización Jerárquica")
		#Descripcion de cluster jerarquica
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			estandarizar=StandardScaler()
			MEstandarizada=estandarizar.fit_transform(archivo._get_numeric_data())
			st.subheader("Árbol de jerarquia")
			plt.figure(figsize=(10,7))
			plt.xlabel('Observaciones')
			plt.ylabel('Distancia')
			shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
			st.pyplot(plt.show())

			st.text("Seleccione la cantidad de clusters")
			numClus=st.selectbox("Numero de clusters", options=[1,2,3,4,5,6,7,8,9,10])
			MJerarquico = AgglomerativeClustering(n_clusters=numClus, linkage='complete', affinity='euclidean')
			MJerarquico.fit_predict(MEstandarizada)
			archivo['clusterH']=MJerarquico.labels_
			st.text("Cantidad de elementos en los clusters")
			st.write(archivo.groupby(['clusterH'])['clusterH'].count())
			CentroidesH=archivo.groupby('clusterH').mean()
			st.text("Centroides de los clusters")
			st.table(CentroidesH)

			plt.figure(figsize=(10,7))
			plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
			plt.grid()
			st.pyplot(plt.show())

def cParticional(archivo):
	with particional:
		st.title("Clusterización Particional (K-Means)")
		#Descripcion de cluster particional
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			st.subheader("Método del codo (Elbow Method)")
			st.markdown("Definimos la cantidad de clúteres")
			estandarizar=StandardScaler()
			MEstandarizada =estandarizar.fit_transform(archivo._get_numeric_data())
			SSE=[]
			for i in range(2,12):
				km=KMeans(n_clusters=i, random_state=0)
				km.fit(MEstandarizada)
				SSE.append(km.inertia_)
			plt.figure(figsize=(10,7))
			plt.plot(range(2,12), SSE, marker='o')
			plt.xlabel('Cantidad de clusters *k*')
			plt.ylabel('SSE')
			plt.title('Elbow method')
			st.pyplot(plt.show())

			kl=KneeLocator(range(2,12), SSE, curve="convex", direction="decreasing")
			st.write("De acuerdo a la gráfica, el numero de clusters será: ", kl.elbow)

			MParticional=KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
			MParticional.predict(MEstandarizada)
			archivo['clusterP']=MParticional.labels_
			st.text("Cantidad de elementos en los clusters")
			st.write(archivo.groupby(['clusterP'])['clusterP'].count())
			CentroidesP=archivo.groupby('clusterP').mean()
			st.text("Centroides de los clusters")
			st.table(CentroidesP)
			plt.rcParams['figure.figsize']=(10,7)
			plt.style.use('ggplot')
			colores=['red', 'blue', 'green', 'yellow', 'cyan', 'black']
			color=colores[:kl.elbow]
			asignar=[]
			for row in  MParticional.labels_:
				asignar.append(colores[row])
			fig=plt.figure()
			ax=Axes3D(fig)
			ax.scatter(MEstandarizada[:, 0], MEstandarizada[:, 1], MEstandarizada[:, 2], marker='o', c=asignar, s=60)
			ax.scatter(MParticional.cluster_centers_[:,0], MParticional.cluster_centers_[:,1], MParticional.cluster_centers_[:,2], marker='o', c=color, s=1000)
			st.pyplot(plt.show())

def reglasAsociacion(archivo):
	with reglas:
		st.title("**Reglas de Asociación**")
		st.write("Las reglas de asociación es un algoritmo de aprendizaje automático basado en reglas, que se utiliza para encontrar relaciones ocultas en los datos.")
		st.write("Consiste en identificar un conjunto de patrones secuenciales en forma de reglas de tipo (Si/Entonces). Estos patrones tienen cierta frecuencia (ocurrencia) en los datos.")
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			#archivo = pd.read_csv(file, header=None)
			transacciones = archivo.values.reshape(-1).tolist()
			ListaM = pd.DataFrame(transacciones)
			ListaM['Frecuencia'] = 0
			ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
			ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
			ListaM = ListaM.rename(columns={0 : 'Item'})

			plt.figure(figsize=(16,20), dpi=300)
			plt.ylabel('Item')
			plt.xlabel('Frecuencia')
			plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
			st.markdown("Frecuncia de cada elemento")
			st.pyplot(plt.show())

			st.write(archivo.shape)

			ListaDatos= archivo.stack().groupby(level=0).apply(list).tolist()

			st.header("Ingrese los siguientes datos: ")
			ocurrencia = st.number_input(label="Ocurrencia", step=1)
			confianza = st.slider("Confianza mínima", 0, 100) 
			elevacion = st.slider(label="Elevación", min_value=0.01, max_value=5.0, step=0.01)
			if st.button('Enviar'):
				reglasConf = apriori(ListaDatos, min_support=ocurrencia/archivo.shape[0], min_confidence=confianza/100, min_lift=elevacion)
				Resultados = list(reglasConf)
				st.markdown("Reglas de Asociación encontradas: ")
				#st.dataframe(pd.DataFrame(Resultados))
				for item in Resultados:
					Emparejar = item[0]
					items = [x for x in Emparejar]
					st.write("Regla: ", str(item[0]))
					st.write("Soporte: ", str(item[1]))
					st.write("Confianza: ", str(item[2][0][2]))
					st.write("Elevación: ", str(item[2][0][3]))
					st.write("==============================================")

def arbolesDecision(archivo):
	with arboles:
		st.title("**Árboles de decisión**")
		st.write("El objetivo de este algoritmo es construir una estructura jerárquica eficiente y escalable que divide los datos en función de determinadas condiciones.")
		if not file:
			show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
		else:
			variables = archivo._get_numeric_data().columns.values
			varClase = st.selectbox("Seleccione la variable de clase", variables)
			a=archivo._get_numeric_data().drop([varClase], axis=1)
			X = np.array(archivo[a.columns.values]) #drop(varClase)
			Y = np.array(archivo[varClase])
			X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                  shuffle = True)
			pd.DataFrame(X_train)
			pd.DataFrame(Y_train)
			PronosticoAD = DecisionTreeRegressor()
			PronosticoAD.fit(X_train, Y_train)
			Y_Pronostico = PronosticoAD.predict(X_test)
			pd.DataFrame(Y_Pronostico)
			Valores = pd.DataFrame(Y_test, Y_Pronostico)
			plt.figure(figsize=(20, 5))
			plt.plot(Y_test, color='green', marker='o', label='Y_test')
			plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
			plt.title('Pronostico')
			plt.grid(True)
			plt.legend()
			st.pyplot(plt.show())
			st.subheader("Obtención de los parámetros del modelo")
			st.write('Criterio: \n', PronosticoAD.criterion)
			st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
			st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
			st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
			st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))
			Importancia = pd.DataFrame({'Variable': list(a.columns.values),
                            			'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
			st.dataframe(Importancia)
			st.subheader("Conformación del modelo de pronóstico")
			plt.figure(figsize=(16,16))  
			plot_tree(PronosticoAD, feature_names = a.columns.values)
			st.pyplot(plt.show())

def main():
	if(eleccion=="Seleccione un algorimo"):
		with header:
			st.title("**Mineria de datos**")
			st.text("Cadena Campos Luis")
			st.text("\t Correo de contacto: luis14oriente@gmail.com")
			
			st.header("**Objetivo.**")
			st.write("El objetivo de este proyecto es implementar algunos de los algortimos",
					 "mas usados para mineria de datos y mostrarlos como una herramienta para",
					 "el análisis de grandes conjuntos de datos")
			st.header("**Introducción.**")
			st.write("La minería de datos es el proceso computacional para la exploración ",
				     "y análisis inteligente de datos como apoyo para el proceso de la toma", 
				     "de decisiones.")
			st.write("La minería de datos puede ser aplicada en diversos ambitos como: Finanzas, Análisis de mercado, Procesos industriales, Medicina, Biología, Química, Telecomunicaciones, Análisis científico, Biometría, Hidrología")
			st.header("**Instrucciones.**")
			st.write("Para comenzar, ingrese un archivo de datos con extensión .csv en el menú lateral y seleccione el algoritmo que desea ejecutar")
	elif(eleccion == "EDA - Análisis Exploratorio de Datos"):
		algoritmoEDA(archivo)
	elif(eleccion == "Selección de Características"):
		with seleccionCaract:
			st.title("**Selección de Características**")
			st.write("Es el proceso de ordenar las variables por el valor de alguna función de puntuación.")
			st.write("De esta forma podemos reducir la dimensionalidad, es decir, el número de variables en nuestro conjunto de datos, eliminando aquellas que no nos proporcionen información relevante")
			sc=st.radio("Elija el algoritmo de Selección", ["Análisis Correlacional de Datos (ACD)",
															"Análisis de Componentes Principales (ACP)"])
			if(sc == "Análisis Correlacional de Datos (ACD)"):
				ACD(archivo)
			else:
				ACP(archivo)
	elif(eleccion == "Clusterización"):
		with clustering:
			st.title("**Clusterización**")
			st.write("El análisis clústeres consiste en la segmentación y delimitación de grupos de objetos (elementos), que son unidos por características comunes que éstos comparten")
			st.write("El objetivo es dividir una población heterogénea de elementos en un número de grupos naturales (regiones o segmentos homogéneos), de acuerdo a sus similitudes.")
			clus=st.radio("Elija el algoritmo de clusterización", ["Jerárquico","Particional"])
			if(clus == "Jerárquico"):
				cJerarquico(archivo)
			else:
				cParticional(archivo)
	elif(eleccion == "Reglas de Asociación"):
		reglasAsociacion(archivo)
	elif(eleccion == "Árboles de decisión"):
		arbolesDecision(archivo)

eleccion="Seleccione un algorimo"
header = st.container()
EDA = st.container()
seleccionCaract = st.container()
analisisCD = st.container()
analisisCP = st.container()
clustering = st.container()
jerarquico = st.container()
particional = st.container()
reglas = st.container()
arboles = st.container()

file = st.sidebar.file_uploader("Ingrese un archivo", type=["csv"])
eleccion = st.sidebar.selectbox("Algoritmo",options=["Seleccione un algorimo",
								"EDA - Análisis Exploratorio de Datos",
								"Selección de Características",
								"Clusterización",
								"Reglas de Asociación",
								"Árboles de decisión"])
show_file = st.empty()
if not file:
	show_file.info("Por favor, ingrese un archivo ".format(' '.join(["csv"])))
else:
	archivo = pd.read_csv(file)
st.set_option('deprecation.showPyplotGlobalUse', False)
if  __name__ ==  '__main__':
 	main()