# RAMON GUI
GUI para el manejo de los datos obtenidos por la estación de monitoreo.

**RAMON** RaspberryPi Acoustic Monitor:
Estación de monitoreo acústico urbano. Capaz de proveer monitoreo continuo, remoto y autónomo. Desarrollada a partir de un micrófono MEMS digital. La misma realiza básicamente las siguientes funciones:

- Grabación de ruido urbano de a ciclos de 10 s.
- Aplicación de filtro de ponderación "A".
- Ajuste compensatorio de la respuesta del micrófono.
- Análisis del audio por segundo.
- Cáculo de niveles equivalentes globales y por octava.

## GUI

Tiene como finalidad: 

- El procesamiento del archivo de volcado en formato *.npy* de la estación, generando y guardando en un archivo *.hd5* la información en formato *pandas dataframe*.
- Generación de índice del tipo *datetime* para los *dataframe*.
- Elección del período a visualizar según día/s y hora/s.
- Integración temporal de acuerdo al intervalo elegido (en horas, minutos o segundos).
- Cálculo de percentiles, valores máximo y mínimo. 
- Cálculo del niveles sonoro continuo equivalente día-tarde-noche $L_{den}$.
- Generación de gráficos según el análisis elegido.
- Guardado de la información en un archivo *.xlsx*.

## Información de prueba

Se deja adjunto en la carpeta */2020_1*, con la finalidad de probar la GUI desarrollada, un archivo (ya procesados por la GUI) en formato *.hd5* con los resultados de mediciones en distintos períodos para tres días.

## Uso

Para utilizar la GUI deben descargarse o clonarse los arhivos del repositorio. Posteriormente se debe ejecutar con *Jupyter notebook* <code>GUI.ipynb</code>. 
Los siguiente paquetes deben estar instalados:

- numpy
- scipy
- pandas
- ipython
- jupyter
- tkinter
- matplotlib
