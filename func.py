# -*- coding: utf-8 -*-
""" 
Copia de las funciones originales utilizadas en la ejecución del dispositivo.
Se incluyen funciones necesarias para el manejo de los datos por la GUI.
"""

import os
import numpy as np
import datetime
import time
import pandas as pd

root = '/home/pi/Documents/'

"""
Funciones para el cálculo de niveles.
"""

def rms(x):
    """Cálculo de nivel RMS para la señal de entrada."""
    return np.sqrt(np.sum(x**2)/len(x))

def rms_t(x, t=1.0):
    N = int(np.floor(t*fs))
    if x.ndim == 1 :
        p_inic = np.arange(0, len(x), N)
        p_fin = np.zeros(len(p_inic), dtype='int32')
        p_fin[0:-1] = p_inic[1:]
        p_fin[-1] = len(x)
        y= np.empty(len(p_inic))
        for i in np.arange(len(p_inic)):
            y[i] = rms(x[p_inic[i]:p_fin[i]])
    else:
        p_inic = np.arange(0, x.shape[1], N)
        p_fin = np.zeros(len(p_inic), dtype='int32')
        p_fin[0:-1] = p_inic[1:]
        p_fin[-1] = x.shape[1]
        y= np.empty([x.shape[0], len(p_inic)])
        for i in np.arange(x.shape[0]):
            for j in np.arange(len(p_inic)):
                y[i, j] = rms(x[i, p_inic[j]:p_fin[j]])
    return y

def niveles(x, cal, ncal):
    return 20*np.log10(x/cal)+ncal

def sum_db(x):
    return 10*np.log10(np.sum(10**(x/10)))

def mean_db(x):
    return 10*np.log10(np.mean(10**(x/10)))

def mean_t(x):
    """Promedio energético de todo el tiempo analizado para
    cada banda fraccional de octava o para el nivel global."""
    return np.apply_along_axis(mean_db,1,x)

def sum_f(x):
    """Suma energética de todos los niveles en frecuencia,
    para cada instante temporal."""
    return np.apply_along_axis(sum_db,0,x)

"""
Escritura de datos de salida.
"""

def guardado(yy, mn, root=root):
    """ Busca todos los arrays de .npy creados para el mes ingresado y los 
    guarda en dataframes sobre archivos .h5.
    Los datos ingresados deben estar en formato str."""
    path = root + str(yy) + '_' + str(mn) + '/'    
    dias_npy = []
    for items in os.listdir(path):
        if items[-4:] == '.npy':
            if items[1] == '.':
                dias_npy.append(items[0])
            else:
                dias_npy.append(items[:2])
    dias_npy.sort(key=int)            
    guardar_h5(yy, mn, dias_npy, root)       
    return

def guardar_h5(yy, mn, dd, root=root, datos=0):
    path = root + str(yy) + '_' + str(mn) + '/'    
    path_datos = path + 'datos.h5'
    if type(datos) == int:
        datos = npy_a_df(yy, mn, dd, root)
    if type(datos) == pd.core.frame.DataFrame:
        datos = {dd: datos}
    if type(datos) == dict:
        store = pd.HDFStore(path_datos)
        if not os.path.isfile(path_datos):
            for keys in datos:
                store.put(keys, datos[keys])  
                os.remove(path + keys + '.npy')
        else:
            dias_datos = sorted(list(datos.keys()), key=int)
            dias_store = []
            for keys in list(store.keys()):
                if keys[-2] == '/':
                    dias_store.append(keys[-1])
                else:
                    dias_store.append(keys[-2:])
            dias_store.sort(key=int)
            for keys in dias_datos:
                if keys in dias_store:
                    aux = store[keys]
                    data_n = pd.concat((aux, datos[keys]))
                    store.put(keys, data_n)
                    os.remove(path + keys + '.npy')
                else:
                    store.put(keys, datos[keys])
                    os.remove(path + keys + '.npy')
        store.close()
    return

def npy_a_df_dias(yy, mn, dd, root=root):
    dds = sorted(dd, key=int)
    datos_dict = {}
    for day in dds:
        try:
            datos_dict[day] = npy_a_df(yy, mn, day, root)
        except:
            continue
    return datos_dict   

def npy_a_df(yy, mn, dd, root=root):
    """ Busca datos de manera recursiva para cada uno de los días indicados, 
    devolviendo un dataframe con el período de medición y sus resultados.
    En caso de ingresar varios días, la función devuelve un diccionario
    con un dataframe para cada día.
    Los datos ingresados deben estar en formato str."""
    
    Headers = ['16 Hz', '31.5 Hz', '63 Hz', '125 Hz', '250 Hz', '500 Hz',
               '1 kHz', '2 kHz', '4 kHz', '8 kHz', '16 kHz', 'Global']
    path = root + str(yy) + '_' + str(mn) + '/'
    if not type(dd) == str:
        datos_dict = npy_a_df_dias(yy, mn, dd, root)
        return datos_dict
    else:
        try:
            data = np.load(path + str(dd) + '.npy')
        except:
            print('No hay datos disponibles para el día ' + str(dd) + '.')
            return
        niv = data[:, 3:]
        tts = data[:, :3]
        # Check for time breaks:
        dif_t = tts-np.vstack((tts[0,:],tts[:-1,:]))
        aux = np.where(np.abs(dif_t[:,2])>3)[0]
        cortes = []
        ti = tts[0, :]
        for i in np.arange(len(aux)):
            if dif_t[aux[i],0] == 0 and dif_t[aux[i],1] == 1 and ((dif_t[aux[i],2]+60) <= 2):
                continue
            cortes.append(aux[i])
        cortes.append(dif_t.shape[0])
        didx = pd.date_range(start=datetime.datetime(year=int(yy), month=int(mn), 
                day=int(dd), hour=int(ti[0]), minute=int(ti[1]), second=int(ti[2])),
                freq='1S', periods=cortes[0])
        datos_df = pd.DataFrame(niv[0:cortes[0],:], columns=Headers, index=didx)
        for i in np.arange(len(cortes)-1):
            didx = pd.date_range(start=datetime.datetime(year=int(yy), month=int(mn), 
                    day=int(dd), hour=int(tts[cortes[i],0]), minute=int(tts[cortes[i],1]), second=int(tts[cortes[i],2])),
                    freq='1S', periods=cortes[i+1]-cortes[i])
            aux_df = pd.DataFrame(niv[cortes[i]:cortes[i+1],:], columns=Headers, index=didx)
            datos_df = pd.concat([datos_df, aux_df]).round(2)
        return datos_df

def escr_arr(yy, mn, dd, mat):
    """ Guardado de datos obtenidos e información horaria en un array
    de numpy '.npy' """
    path = root + str(yy) + "_" + str(mn) + "/"
    if not os.path.exists(str(path)):
        # Creación de carpeta
        os.makedirs(str(path))
        print("\nSe creó la carpeta : '" + str(path) + "'")    
    if os.path.isfile(str(path) + str(dd) + ".npy"):
        file = np.load(path + str(dd) + ".npy")
        if file.size == 0:
            file = mat
        else:
            file = np.vstack((file, mat))
        np.save(str(path) + str(dd) + ".npy", file)
    else:
        np.save(str(path) + str(dd) + ".npy", mat)
        # Guardar array
        # Guardar en tabla, al salir de RAMon    
    return
