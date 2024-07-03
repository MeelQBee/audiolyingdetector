import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from moviepy.editor import VideoFileClip
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Función para extraer el audio de un vídeo
def extraer_audio(ruta_video, output_audio_path):
    # Moviepy para extraer el audio del video y guardarlo como un archivo de audio
    clip = VideoFileClip(ruta_video)
    clip.audio.write_audiofile(output_audio_path)

# Función para convertir el audio en espectrograma
def audio_a_espectrograma(ruta_audio, output_img_path):
    # Aquí cargo el archivo de audio y normalizo el volumen
    y, sr = librosa.load(ruta_audio)
    y = librosa.util.normalize(y)
    
    # Se crea un espectrograma utilizando el método Mel-scaled spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=256)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Guardo el espectrograma como imagen
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.savefig(output_img_path)
    plt.close()

# Función para crear un modelo simplificado con Inception V3 con Dropout y regularización L2
def crear_modelo():
    # Cargo el modelo base Inception V3 preentrenado
    modelo_base = InceptionV3(weights='imagenet', include_top=False)
    
    # Añado capas adicionales al modelo
    x = modelo_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predicciones = Dense(2, activation='softmax')(x)
    
    # Aqui se define el modelo final
    modelo = Model(inputs=modelo_base.input, outputs=predicciones)
    modelo.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

# Función para entrenar el modelo con Early Stopping, aumento de datos y registro de métricas
def entrenar_modelo(modelo, datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion):
    # Configuración del Early Stopping para detener el entrenamiento si no hay mejora
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    csv_logger = CSVLogger('training_log.csv', append=False)
    
    # Configuro el generador de datos para el aumento de datos
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    datagen.fit(datos_entrenamiento)
    
    # Entrena el modelo con el generador de datos
    modelo.fit(datagen.flow(datos_entrenamiento, etiquetas_entrenamiento, batch_size=32),
               validation_data=(datos_validacion, etiquetas_validacion),
               epochs=50, callbacks=[early_stopping, csv_logger])

# Función para predecir la clase de una imagen
def predecir_clase(modelo, ruta_imagen):
    # Se carga la imagen y la procesa para que el modelo pueda hacer predicciones
    img = image.load_img(ruta_imagen, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = modelo.predict(x)
    return np.argmax(preds[0])

# Función para procesar los videos y entrenar el modelo
def procesar_videos_y_entrenar_modelo(videos_T_txt, videos_F_txt, deceptive_dir, truthful_dir):
    # Se crean directorios para almacenar audios y espectrogramas
    audio_dir = 'audios'
    spectrogram_dir = 'spectrograms'
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    
    datos_entrenamiento = []
    etiquetas_entrenamiento = []

    # Procesa los videos en la carpeta Deceptive (Mentira)
    for root, _, files in os.walk(deceptive_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                audio_path = os.path.join(audio_dir, f"{os.path.splitext(file)[0]}.wav")
                extraer_audio(video_path, audio_path)
                output_img_path = os.path.join(spectrogram_dir, f"{os.path.splitext(file)[0]}.png")
                audio_a_espectrograma(audio_path, output_img_path)
                img = image.load_img(output_img_path, target_size=(299, 299))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                datos_entrenamiento.append(x)
                etiquetas_entrenamiento.append(1)  # Marca como 1 para Deceptive (Mentira).

    # Procesa los videos en la carpeta Truthful (Verdad)
    for root, _, files in os.walk(truthful_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                audio_path = os.path.join(audio_dir, f"{os.path.splitext(file)[0]}.wav")
                extraer_audio(video_path, audio_path)
                output_img_path = os.path.join(spectrogram_dir, f"{os.path.splitext(file)[0]}.png")
                audio_a_espectrograma(audio_path, output_img_path)
                img = image.load_img(output_img_path, target_size=(299, 299))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                datos_entrenamiento.append(x)
                etiquetas_entrenamiento.append(0)  # Marca como 0 para Truthful (Verdad).

    # Procesa los videos desde el archivo videos_T.txt (Verdad)
    with open(videos_T_txt, 'r') as file_T:
        for video_path in file_T:
            video_path = video_path.strip()
            audio_path = os.path.join(audio_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")
            extraer_audio(video_path, audio_path)
            output_img_path = os.path.join(spectrogram_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.png")
            audio_a_espectrograma(audio_path, output_img_path)
            img = image.load_img(output_img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            datos_entrenamiento.append(x)
            etiquetas_entrenamiento.append(0)  # Marca como 0 para Truthful (Verdad)

    # Procesa los videos desde el archivo videos_F.txt (Mentira)
    with open(videos_F_txt, 'r') as file_F:
        for video_path in file_F:
            video_path = video_path.strip()
            audio_path = os.path.join(audio_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")
            extraer_audio(video_path, audio_path)
            output_img_path = os.path.join(spectrogram_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.png")
            audio_a_espectrograma(audio_path, output_img_path)
            img = image.load_img(output_img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            datos_entrenamiento.append(x)
            etiquetas_entrenamiento.append(1)  # Marca como 1 para Deceptive (Mentira)

    datos_entrenamiento = np.vstack(datos_entrenamiento)
    etiquetas_entrenamiento = to_categorical(np.array(etiquetas_entrenamiento))

    # División de los datos en conjuntos de entrenamiento y validación
    datos_entrenamiento, datos_validacion, etiquetas_entrenamiento, etiquetas_validacion = train_test_split(
        datos_entrenamiento, etiquetas_entrenamiento, test_size=0.2, random_state=42)

    modelo = crear_modelo()
    entrenar_modelo(modelo, datos_entrenamiento, etiquetas_entrenamiento, datos_validacion, etiquetas_validacion)
    
    return modelo, spectrogram_dir

# Función que me saca las gráficas de las métricas de entrenamiento y validación
def graficar_métricas(log_file):
    df
