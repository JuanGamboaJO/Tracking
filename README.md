# EYEIA

## Descripción

Este repositorio contiene la base de datos y los archivos necesarios para correr el script que se encarga de predecir la dirección a la cual moveras el cursor

## Estructura del Proyecto

- **Ojos/**
  - Contiene las imágenes correspondientes a los ojos de cada sujeto, separadas por ojo derecho e izquierdo.

- **Models/**
  - Contiene los modelos entrenados.

## Instrucciones de Uso

1. **Clonar el repositorio**
    ```sh
    git clone https://github.com/JuanGamboaJO/Tracking.git
    ```

2. **Estructura de las Carpetas**
    - Asegúrate de que las carpetas mencionadas anteriormente contengan las imágenes necesarias antes de ejecutar el programa.

3. **Entrenamiento y Test de la Red Neuronal**
    - El código divide el 80% de las fotos para entrenamiento, 10% para validación y 10% para testeo.
    - Los modelos entrenados se guardarán en la carpeta `Models`.

4. **Ejecutable**
    - "Para usar el programa sin la necesidad de instalar librerías o Python, puedes descargar el ejecutable desde este link.. 
    (https://drive.google.com/file/d/1S-62edhrlk324KH3VX_n_vplx8FksnJ4/view?usp=drive_link) Red entrenada ResNet18
    (https://drive.google.com/file/d/1Hq10zRCp_oWyDkkw13nHQbrYrrAJnQiN/view?usp=drive_link) Red Propuesta