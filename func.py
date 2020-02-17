# -*- coding: utf-8 -*-
""" 
Implentación de funciones útiles al procesado del audio, dentro de este
script se incluyen:
- Funciones de aplicación y generación de los filtros de octavas y tercios de
octavas, definidos según la Norma IEC 61620-1.
- Implementación de funciones para la realización de ponderaciones, temporales
y de frecuencia, a partir de la Norma IEC 61672-1.
"""

import os
import numpy as np
import sounddevice as sd
from scipy.signal import butter, zpk2sos, sosfilt, convolve
import datetime
import time
import pandas as pd
import warnings
import tables

# Parámetros por defecto
fs = 48000 # Frecuencia de Muestreo [Hz]
fr = 1000.0 # Frecuencia de Referencia [Hz]

root = '/home/pi/Documents/'

""" 
Filtrado de frecuencia por octavas y tercios de octavas.
"""

fto_nom = np.array([ 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0,
    80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
    800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
    6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
])
foct_nom = fto_nom[1::3]

oct_ratio = np.around(10.0**(3.0/10.0), 5) # Según ecuación 1 en IEC 61672-1

def frec_a_num(frecs = fto_nom, frec_ref = fr, n_oct = 3.0):
    """Devuelve el número de banda para las frecuencias centrales ingresadas,
    correspondiendo la banda 0 a la frecuencia de referencia.
    n_oct representa el número de bandas por octava, siendo 1.0 para octavas
    y 3.0 para tercios de octava."""
    return np.round(n_oct*np.log2(frecs/frec_ref))

def frec_cen(nband = np.arange(-6, 5), frec_ref = fr, n_oct = 3.0):
    """Calcula las frecuencias centrales exactas para bandas de octava y
    tercios de octava. Recibe los números de banda a calcular, la
    frecuencia de referencia, y el número de bandas por octava."""
    return np.around(frec_ref*oct_ratio**(nband/n_oct), 5)

def frec_lim(frec_cen, n_oct = 3.0):
    """Devuelve las frecuencias lí­mites (inferior y superior), para
    bandas de tercio de octava y octavas, según los valores medios exactos
    de las bandas y el número de bandas por octava."""
    return np.around(frec_cen*oct_ratio**(-1/(2*n_oct)), 5), np.around(frec_cen*oct_ratio**(1/(2*n_oct)), 5)

nb_to = frec_a_num(fto_nom, fr, 3.0)
fto_cen = frec_cen(nb_to, fr, 3.0)
fto_inf, fto_sup = frec_lim(fto_cen, 3.0)

nb_oct = frec_a_num(foct_nom, fr, 1.0)
foct_cen = frec_cen(nb_oct, fr, 1.0)
foct_inf, foct_sup = frec_lim(foct_cen, 1.0)

def but_pb(inf, sup, fs=fs, order=4):
    """Obtención de los coeficientes para el diseño del filtro.
    Siendo estos filtros Butterworth pasa banda."""
    nyq = 0.5*fs
    low = inf/nyq
    high = sup/nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def but_pb_filt(x, inf, sup, fs=fs, order=4):
    """Filtrado de la señal."""
    sos = but_pb(inf, sup, fs=fs, order=order)
    return sosfilt(sos, x)

sos_oct = np.zeros([len(nb_oct), 4, 6])
sos_to = np.zeros([len(nb_to), 4, 6])

for i in range(len(nb_oct)):
    sos_oct[i] = but_pb(foct_inf[i], foct_sup[i])
    
for i in range(len(nb_to)):
    sos_to[i] = but_pb(fto_inf[i], fto_sup[i])
    
""" 
Funciones para ponderación temporal y de frecuencia.
"""

# Coeficientes calculados a partir de las ecuaciones en IEC 61672-1
z_c = np.array([0, 0])
p_c = np.array([-2*np.pi*20.598997057568145, -2*np.pi*20.598997057568145,
    -2*np.pi*12194.21714799801, -2*np.pi*12194.21714799801])
k_c = (10**(0.062/20))*p_c[3]**2

z_a = np.append(z_c, [0,0])
p_a = np.array([-2*np.pi*20.598997057568145, -2*np.pi*20.598997057568145,
     -2*np.pi*107.65264864304628, -2*np.pi*737.8622307362899,
     -2*np.pi*12194.21714799801, -2*np.pi*12194.21714799801])
k_a = (10**(2/20))*p_a[4]**2

def zpk_bil(z, p, k, fs=fs):
    """Devuelve los parametros para un filtro digital a partir de un analógico,
    a partir de la transformada bilineal. Transforma los polos y ceros del
    plano 's' al plano 'z'."""
    deg = len(p) - len(z)
    fs2 = 2.0*fs
    # Transformación bilineal de polos y ceros:
    z_b = (fs2 + z)/(fs2 - z)
    p_b = (fs2 + p)/(fs2 - p)
    z_b = np.append(z_b, -np.ones(deg))
    k_b = k*np.real(np.prod(fs2 - z)/np.prod(fs2 - p))
    return z_b, p_b, k_b

zbc, pbc, kbc = zpk_bil(z_c, p_c, k_c)
sos_C = zpk2sos(zbc, pbc, kbc)

zba, pba, kba = zpk_bil(z_a, p_a, k_a)
sos_A = zpk2sos(zba, pba, kba)
                   
def filt_A(x):
    """Devuelve la señal posterior al proceso de filtrado según
    ponderación "A". Recibe la señal de entrada (en dimensión temporal),
    y la frecuencia de sampleo."""
    return sosfilt(sos_A, x)

def filt_C(x):
    """Devuelve la señal posterior al proceso de filtrado según
    ponderación "C". Recibe la señal de entrada (en dimensión temporal)."""    
    return sosfilt(sos_C, x)

"""
Filtro inverso.
"""
def filt_inv(x, b):
    return convolve(x, b, mode='same')

    
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
Calibrado del dispositivo.
"""

def cal():
    print("\nComenzando la calibración en 5 segundos.")
    time.sleep(5)
    print("\nGrabando calibración ... ")
    sd.default.device = 'snd_rpi_simple_card'
    x = sd.rec(int(5*fs), 48000, 1, blocking=True)
    vcal = rms(x)
    print("\nGrabación finalizada.")
    ncal = float(input("\nIngrese valor de la calibración: "))
    np.savetxt(root + 'cal.csv', (vcal, ncal))
    print("\nSe escribió el archivo 'cal.csv'")
    return
    return vcal, ncal

def busca_cal(root=root):
    vcal, ncal = np.loadtxt(root + 'cal.csv', unpack=True)
    return vcal, ncal

def busca_ajuste(root=root):
    ajuste = np.loadtxt(root + 'ajuste.csv', unpack=False)
    return ajuste

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
"""     
def ver(yy, mn, dd=-1, hh=-1, mnn=-1):
""" 
#Busca los datos guardados en .h5 y entrega un dataframe con los datos.
"""
    datos_df = 0
    return datos_df
"""
def guardar_h5(yy, mn, dd, root=root, datos=0):
    warnings.simplefilter('ignore', tables.NaturalNameWarning)
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