from random import sample

import matplotlib.pyplot as plt     # Para graficar
import numpy as np                  # Para manejo de arreglos
import pandas as pd                 # Para manejo dee tablas
import statistics as st             # Para estadísticas (librería estándar)
import scipy.stats as sp            # Para estadísticas (librería científica)
import wfdb                         # Para manejo de bases de datos fisiológicas
import os                           # Para manejo de archivos e interacción con sistema operativo

from scipy.signal import find_peaks                         # Para encontrar picos en señales
from skimage.restoration import denoise_wavelet             # Para filtrado de señales
from scipy.fft import rfft, rfftfreq                        # Para transformada de Fourier
from keras.models import Sequential                         # Para creación de modelos de redes neuronales
from keras.layers import Dense                              # Para creación de capas densas
from keras.optimizers import SGD                            # Para optimización de modelos
import hashlib                    # Para verificación de integridad de archivos

mit_bih_db_path = './data/mitdb'

class RawDataWrapper:
    def __init__(self, record_name, data, headers, annotations):
        self.record_name = record_name
        self.data = data
        self.headers = headers
        self.annotations = annotations

class RawDataHandler:
    def __init__(self, db_path='mitdb'):
        self.db_path = db_path
        self.is_local_db = os.path.isdir(self.db_path)
        self.records_names = self.get_record_list()

    # Obtiene la lista de registros, si la base de datos es local lee el archivo RECORDS, si no descarga desde PhysioNet
    def get_record_list(self):
        if self.is_local_db:
            # Lee la lista de registros desde el archivo RECORDS
            with open(self.db_path + '/RECORDS') as f:
                records = f.readlines()
                records = [x.strip() for x in records]
            return records
        else:
            return wfdb.get_record_list(self.db_path, records='all')

    # Verifica que la base de datos local esté completa, requiere el archivo sha256sums
    def is_local_dataset_integrity_ok(self, sha256sums_file_path, ignore_extensions=['.xws']):
        if not os.path.isfile(sha256sums_file_path):
            print('Cannot verify integrity, sha256sums file not found')
            return False
        with open(sha256sums_file_path) as f:
            file_line = f.readlines()
            split_lines = [x.strip().split(" ") for x in file_line]
            for line in split_lines:
                file = self.db_path + '/' + line[1]
                if any(ext in file for ext in ignore_extensions):
                    continue
                if not os.path.isfile(file):
                    print('File ' + line[1] + ' not found in database')
                    return False
                # Verifica la suma sha256 del archivo
                with open(file, 'rb') as h:
                    file_hash = hashlib.sha256(h.read()).hexdigest()
                    if file_hash != line[0]:
                        print('File ' + line[1] + ' has an invalid hash')
                        return False
        return True

    def get_pn_dir(self):
        if self.is_local_db:
            return None
        else:
            return self.db_path

    def get_record_name_path(self, record_name):
        if self.is_local_db and (record_name in self.records_names):
            return self.db_path + '/' + record_name
        return record_name

    def get_data(self, name, start=0, end=None, channels=None):
        return wfdb.io.rdrecord(self.get_record_name_path(name), pn_dir=self.get_pn_dir(), sampfrom=start, sampto=end,
                                channels=channels)

    # Obtiene los encabezados de un archivo de la base de datos
    def get_headers(self, name):
        return wfdb.io.rdheader(self.get_record_name_path(name), pn_dir=self.get_pn_dir())

    # Obtiene las anotaciones de un archivo de la base de datos
    def get_annotations(self, name):
        return wfdb.io.rdann(self.get_record_name_path(name), 'atr', pn_dir=self.get_pn_dir())

    def get(self, name):
        return RawDataWrapper(name, self.get_data(name), self.get_headers(name), self.get_annotations(name))

    # Obtiene todas las muestras de datos
    def get_all_data(self):
        data = []
        for record in self.records_names:
            data.append(self.get_data(record))
        return data

    # Obtiene todos los encabezados
    def get_all_annotations(self):
        annotations = []
        for record in self.records_names:
            annotations.append(self.get_annotations(record))
        return annotations

    def get_all_in_wrapper(self):
        data = []
        for record in self.records_names:
            data.append(self.get(record))
        return data

def normalize_signal(data, lb=0, ub=1):
    mid = ub - (ub - lb) / 2
    min_v = np.min(data)
    max_v = np.max(data)
    mid_v = max_v - (max_v - min_v) / 2
    coe = (ub - lb) / (max_v - min_v)
    return data * coe - (mid_v * coe) + mid

def wavelet_transform(signal):
    return denoise_wavelet(signal,
                           method='VisuShrink',
                           mode='soft',
                           wavelet_levels=6,
                           wavelet='sym8',
                           rescale_sigma='True')

def denoise_signal(signal):
    denoised_ch1 = wavelet_transform(np.array(signal[:, 0]))
    signal[:, 0] = denoised_ch1

def get_time_vector(signal, freq_sampling):
    return np.linspace(0, np.size(signal), np.size(signal)) * (1 / freq_sampling)

def umbral_peaks(signal):
    absolute_ch= np.abs(signal)
    umbral_y_ch = np.mean(absolute_ch)
    # Los picos son encontrados en donde el valor de la señal supera a la media
    peaks_ch1, _ = find_peaks(signal, height=umbral_y_ch)
    return peaks_ch1

def heart_rate_variability(peaks):
    # Realiza la diferencia entre cada una de las muestras, el resultado lo almacena en otro arreglo
    hrv = np.diff(peaks)
    return hrv

def get_bpm(rr_interval):
    return (1 / rr_interval) * 60

def rmssd(hrv):
    suma = 0
    # Realiza la diferencia entre cada una de las muestras, el resultado lo almacena en otro arreglo
    resta_rr = np.diff(hrv)
    if np.size(resta_rr) == 0:
        return 0
    # Eleva al cuadrado las diferencias y las suma
    squared_numbers = [number ** 2 for number in resta_rr]
    for num in squared_numbers:
        suma += num
    normalize = suma / np.size(resta_rr)
    rmssd_value = np.sqrt(normalize)
    return rmssd_value*1000

def pnnx(hrv, x):
    # Valor absoluto de las diferencias entre intervalo RR
    resta_rr = np.abs(np.diff(hrv))
    if np.size(resta_rr) == 0:
        return 0, 0
    filtering_condition = resta_rr > (x / 1000)
    # Obtiene el conteo de todos los elementos cuya diferencia es mayor a x
    cumulative_sum = np.size(resta_rr[filtering_condition])
    return (cumulative_sum / np.size(resta_rr)) * 100, cumulative_sum

def get_shannon(signal):
    y = np.power(signal, 2)
    y1 = np.sum(y)
    pe = y / y1
    se = -np.sum(pe * np.log(np.power(pe, 2)))
    return se

def get_logenergy(signal):
    y = np.power(signal, 2)
    y1 = np.sum(y)
    pe = y / y1
    lee = np.sum(pe * np.log(pe))
    return lee

def statistics(signal):
    min_value = np.min(signal)
    max_value = np.max(signal)
    mean_value = np.mean(signal)
    median_value = np.median(signal)
    sum_value = np.sum(signal)
    std_value = st.stdev(signal)
    var_value = st.variance(signal)
    skw_value = sp.skew(signal)
    kurtosis_value = sp.kurtosis(signal)
    shannon_entropy = get_shannon(signal)
    energy_entropy = get_logenergy(signal)
    return min_value, max_value, mean_value, median_value, sum_value, std_value, var_value, skw_value, kurtosis_value, shannon_entropy, energy_entropy

raw_data_manager = RawDataHandler(mit_bih_db_path)

symbols_table = pd.DataFrame([
    ['N', 'Normal Sinus Rhythm'],
    ['·', 'Normal Sinus Rhythm'],
    ['L', 'Left Branch Block Beat'],
    # ['R', 'Right Branch Block Beat'],
    # ['A', 'Atrial Premature Contraction'],
    # ['a', 'Aberrated Atrial Premature Contraction'],
    # ['J', 'Nodal (junctional) Premature Contraction'],
    # ['S', 'Supraventricular Premature or Ectopic Beat'],
    ['V', 'Premature Ventricular Contraction'],
    # ['F', 'Fusion of ventricular and normal beat'],
    # ['[', 'Start of Ventricular Flutter'],
    # ['!', 'Start of Ventricular Flutter'],
    # [']', 'Start of Ventricular Flutter'],
    # ['e', 'Atrial Escape Beat'],
    # ['j', 'Nodal (junctional) Escape Beat'],
    # ['E', 'Ventricular Escape Beat'],
    # ['/', 'Paced Beat'],
    # ['f', 'Fusion of paced and normal beat'],
    # ['n', 'Supraventricular Escape Beat'],
    # ['Q', 'Unclassifiable Beat'],
    # ['?', 'Beat not classified during learning'],
    # ['|', 'Isolated QRS-like artifact']
], columns=['symbol', 'Name'])

# Tabla que contiene las clases de interés
subtype_table = pd.DataFrame([
    # ['(AB\x00', 'Atrial bigeminy'],
    ['(AFIB\x00', 'Atrial fibrillation'],
    # ['(AFL\x00', 'Atrial flutter'],
    # ['(B\x00', 'Ventricular bigeminy'],
    # ['(BII\x00', '2° heart block'],
    # ['(IVR\x00', 'Idioventricular rhythm'],
    # ['(N\x00', 'Normal sinus rhythm'],
    # ['(NOD\x00', 'Nodal (A-V junctional) rhythm'],
    # ['(P\x00', 'Paced rhythm'],
    # ['(PREX\x00', 'Pre-excitation (WPW)'],
    # ['(SBR\x00', 'Sinus bradycardia'],
    # ['(SVTA\x00', 'Supraventricular tachyarrhythmia'],
    # ['(T\x00', 'Ventricular trigeminy'],
    # ['(VFL\x00', 'Ventricular flutter'],
    # ['(VT\x00', 'Ventricular tachycardia']
], columns=['string', 'Name'])


# Construye una tabla de intervalos con sus respectivas clases y símbolos
def get_annotations_table(data_container):
    annotation_data = {
        'name': [],
        'symbol': [],
        'aux': [],
        'start_sample': [],
        'length': []
    }
    for container in data_container:
        size = np.size(container.annotations.symbol)
        for i in range(0, size - 1):
            symbol = container.annotations.symbol[i]
            annotation_data['name'].append(container.annotations.record_name)
            annotation_data['symbol'].append(symbol)
            annotation_data['aux'].append(container.annotations.aux_note[i])
            annotation_data['start_sample'].append(container.annotations.sample[i])
            annotation_data['length'].append(container.annotations.sample[i + 1] - container.annotations.sample[i])
        symbol = container.annotations.symbol[-1]
        annotation_data['name'].append(container.annotations.record_name)
        annotation_data['symbol'].append(symbol)
        annotation_data['aux'].append(container.annotations.aux_note[-1])
        annotation_data['start_sample'].append(container.annotations.sample[-1])
        annotation_data['length'].append(np.size(container.data.p_signal[:, 0]) - container.annotations.sample[-1])
    return pd.DataFrame(annotation_data)


class ChunkManager:
    def __init__(self, table, column='symbol'):
        self.table = table
        self.selected_column = column
        self.chunks = {
            'name': [],
            'symbol': [],
            'start_sample': [],
            'length': []
        }
        self.current_symbol = table[self.selected_column].iat[0]
        self.current_name = table.name.iat[0]
        self.start = 0
        self.length = 0
        index = self.table.columns.get_loc(self.selected_column) + 1
        if self.selected_column == 'aux':
            self.get_aux_chunks(index)
        elif self.selected_column == 'symbol':
            self.get_symbol_chunks(index)

    def get_symbol_chunks(self, index):
        for row in self.table.itertuples():
            self.read_row(row, index)
            self.verify_length(row, index)
            self.length += row.length

    def get_aux_chunks(self, index):
        for row in self.table.itertuples():
            if row[index] != '':
                self.read_row(row, index)
            self.verify_length(row, index)
            self.length += row.length

    def read_row(self, row, index):
        if row[index] != self.current_symbol or self.current_name != row.name:
            self.submit_chunk()
            self.set_aux(row[index], row.name, row.start_sample)

    def verify_length(self, row, index):
        if self.length > 3600:
            self.submit_chunk()
            self.set_aux(name=row.name, start=row.start_sample)

    def submit_chunk(self):
        self.chunks['name'].append(self.current_name)
        self.chunks['symbol'].append(self.current_symbol)
        self.chunks['start_sample'].append(self.start)
        self.chunks['length'].append(self.length)

    def set_aux(self, symbol=None, name='', start=0, length=0):
        if symbol is not None:
            self.current_symbol = symbol
        self.current_name = name
        self.start = start
        self.length = length

def split_chunks(table):
    train, validate, test = np.split(table.sample(frac=1), [int(.6*len(table)), int(.8*len(table))])
    return train, validate, test

def get_records_symbol(table):
    records = []
    for row in table.itertuples():
        records.append(raw_data_manager.get_data(row.name, row.start_sample, row.start_sample + row.length, channels=[0]))
    return records


class FeatureExtractor:
    def __init__(self, data, label=None):
        self.data_containers = data
        self.label = label
        self.features_table = pd.DataFrame(columns=[
            "label",
            'min',
            'max',
            'mean',
            'median',
            'sum',
            'std',
            'var',
            'skew',
            'kurtosis',
            'shannon ent',
            'log energy',
            'pnn25',
            'nn25',
            'pnn50',
            'nn50',
            'pnn75',
            'nn75',
            'rmssd'
        ])

    def start_characterization(self):
        self.normalize_signals()
        self.denoise_signals()
        self.extract_features()
        return self.features_table

    def extract_features(self):
        rows = []
        for item in self.data_containers:
            for i in range(np.size(item.p_signal, 1)):
                min_value, max_value, mean_value, median_value, sum_value, std_value, var_value, skw_value, kurtosis_value, shannon_entropy, energy_entropy = statistics(
                    item.p_signal[:, i])
                time = get_time_vector(item.p_signal[:, i], item.fs)
                peaks = umbral_peaks(item.p_signal[:, i])
                hrv = heart_rate_variability(time[peaks])
                rmssd_value = rmssd(hrv)
                pnn25, nn25 = pnnx(hrv, 25)
                pnn50, nn50 = pnnx(hrv, 50)
                pnn75, nn75 = pnnx(hrv, 75)
                rows.append({
                    "label": self.label,
                    "min": min_value,
                    "max": max_value,
                    "mean": mean_value,
                    "median": median_value,
                    "sum": sum_value,
                    "std": std_value,
                    "var": var_value,
                    "skew": skw_value,
                    "kurtosis": kurtosis_value,
                    "shannon ent": shannon_entropy,
                    "log energy": energy_entropy,
                    "pnn25": pnn25,
                    "nn25": nn25,
                    "pnn50": pnn50,
                    "nn50": nn50,
                    "pnn75": pnn75,
                    "nn75": nn75,
                    "rmssd": rmssd_value
                })
        self.features_table = pd.concat([self.features_table, pd.DataFrame(rows)], ignore_index=True)

    def normalize_signals(self):
        for data in self.data_containers:
            data.p_signal = normalize_signal(data.p_signal, lb=-1)

    def denoise_signals(self):
        for data in self.data_containers:
            denoise_signal(data.p_signal)



# Inicio del entrenamiento
if raw_data_manager.is_local_db:
    if raw_data_manager.is_local_dataset_integrity_ok(mit_bih_db_path + '/SHA256SUMS.txt',
                                                      ignore_extensions=['.xws', '.htm, .at_']):
        print('Local database integrity OK')
    else:
        print('Local database integrity NOT OK')
        exit(1)
data_wrappers = raw_data_manager.get_all_in_wrapper()
ann = get_annotations_table(data_wrappers)
filtered_types = ann[ann['symbol'].isin(symbols_table['symbol'])]
symbol_chunk_manager = ChunkManager(filtered_types)
symbol_chunks_table = pd.DataFrame(symbol_chunk_manager.chunks)
filtered_aux = ann[ann['aux'].isin(subtype_table['string'])]
records_with_afib = filtered_aux['name'].unique()
whole_with_afib = ann[ann['name'].isin(records_with_afib)]
aux_chunk_manager = ChunkManager(whole_with_afib, column='aux')
aux_chunks_table = pd.DataFrame(aux_chunk_manager.chunks)
normal_chunks = symbol_chunks_table[symbol_chunks_table['symbol'].isin(['N'])].sort_values(by='length',
                                                                                           ascending=False)
lbbb_chunks = symbol_chunks_table[symbol_chunks_table['symbol'].isin(['L'])].sort_values(by='length',
                                                                                         ascending=False)
pvc_chunks = symbol_chunks_table[symbol_chunks_table['symbol'].isin(['V'])].sort_values(by='length',
                                                                                        ascending=False)
afib_chunks = aux_chunks_table[aux_chunks_table['symbol'].isin(subtype_table['string'])].sort_values(by='length',
                                                                                                     ascending=False)
normal_chunks = normal_chunks[normal_chunks['length'] > 3000]
lbbb_chunks = lbbb_chunks[lbbb_chunks['length'] > 3000]
pvc_chunks = pvc_chunks[pvc_chunks['length'] > 3000]
afib_chunks = afib_chunks[afib_chunks['length'] > 3000]
normal_train, normal_validate, normal_test = split_chunks(normal_chunks)
lbbb_train, lbbb_validate, lbbb_test = split_chunks(lbbb_chunks)
pvc_train, pvc_validate, pvc_test = split_chunks(pvc_chunks)
afib_train, afib_validate, afib_test = split_chunks(afib_chunks)
normal_records_train = get_records_symbol(normal_train)
lbbb_records_train = get_records_symbol(lbbb_train)
pvc_records_train = get_records_symbol(pvc_train)
afib_records_train = get_records_symbol(afib_train)
normal_records_validate = get_records_symbol(normal_validate)
lbbb_records_validate = get_records_symbol(lbbb_validate)
pvc_records_validate = get_records_symbol(pvc_validate)
afib_records_validate = get_records_symbol(afib_validate)
normal_records_test = get_records_symbol(normal_test)
lbbb_records_test = get_records_symbol(lbbb_test)
pvc_records_test = get_records_symbol(pvc_test)
afib_records_test = get_records_symbol(afib_test)

feature_extractor_afib_train = FeatureExtractor(afib_records_train, label='afib')
feature_extractor_normal_train = FeatureExtractor(normal_records_train, label='normal')
feature_extractor_lbb_train = FeatureExtractor(lbbb_records_train, label='lbbb')
feature_extractor_pvc_train = FeatureExtractor(pvc_records_train, label='pvc')

feature_extractor_afib_test = FeatureExtractor(afib_records_test, label='afib')
feature_extractor_normal_test = FeatureExtractor(normal_records_test, label='normal')
feature_extractor_lbb_test = FeatureExtractor(lbbb_records_test, label='lbbb')
feature_extractor_pvc_test = FeatureExtractor(pvc_records_test, label='pvc')

feature_extractor_afib_validate = FeatureExtractor(afib_records_validate, label='afib')
feature_extractor_normal_validate = FeatureExtractor(normal_records_validate, label='normal')
feature_extractor_lbb_validate = FeatureExtractor(lbbb_records_validate, label='lbbb')
feature_extractor_pvc_validate = FeatureExtractor(pvc_records_validate, label='pvc')
afib_feat_train = feature_extractor_afib_train.start_characterization()
normal_feat_train = feature_extractor_normal_train.start_characterization()
lbb_feat_train = feature_extractor_lbb_train.start_characterization()
pvc_feat_train = feature_extractor_pvc_train.start_characterization()

afib_feat_test = feature_extractor_afib_test.start_characterization()
normal_feat_test = feature_extractor_normal_test.start_characterization()
lbb_feat_test = feature_extractor_lbb_test.start_characterization()
pvc_feat_test = feature_extractor_pvc_test.start_characterization()

afib_feat_validate = feature_extractor_afib_validate.start_characterization()
normal_feat_validate = feature_extractor_normal_validate.start_characterization()
lbb_feat_validate = feature_extractor_lbb_validate.start_characterization()
pvc_feat_validate = feature_extractor_pvc_validate.start_characterization()

input_table = pd.concat([afib_feat_train, normal_feat_train, lbb_feat_train, pvc_feat_train], ignore_index=True)
input_table_test = pd.concat([afib_feat_test, normal_feat_test, lbb_feat_test, pvc_feat_test], ignore_index=True)
input_table_validate = pd.concat([afib_feat_validate, normal_feat_validate, lbb_feat_validate, pvc_feat_validate],
                                 ignore_index=True)
input_size = 18
output_size = 4

model = Sequential()
# Capas ocultas
model.add(Dense(90, activation='relu', input_dim=input_size))
model.add(Dense(180, activation='relu'))
model.add(Dense(360, activation='relu'))
model.add(Dense(180, activation='relu'))
model.add(Dense(90, activation='relu'))
# Capa de salida
model.add(Dense(output_size, activation='softmax'))

# Compilacion
model.compile(loss='poisson', optimizer=SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
X = input_table.drop('label', axis=1).fillna(0)
Y = input_table['label']
Y = pd.get_dummies(Y)
X_test = input_table_test.drop('label', axis=1).fillna(0)
Y_test = input_table_test['label']
Y_test = pd.get_dummies(Y_test)

model.fit(X, Y, epochs=100, batch_size=10, validation_data=(X_test, Y_test))
model.save('./models/ecg_model.h5')