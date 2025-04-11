# Análisis de Datos de PeliPlat

Proyecto de análisis de datos para una plataforma de streaming de películas y series. Este proyecto utiliza Python para analizar métricas de usuarios, contenido, engagement y campañas de marketing.

## Estructura del Proyecto

```
.
├── data/                   # Datos del proyecto
│   ├── raw/                # Datos sin procesar
│   └── processed/          # Datos procesados listos para análisis
├── notebooks/              # Jupyter notebooks para análisis exploratorio
├── reports/                # Resultados generados
│   ├── figures/            # Visualizaciones guardadas
│   └── tables/             # Tablas generadas por el análisis
├── src/                    # Código fuente del proyecto
│   ├── analysis/           # Módulos para análisis específicos
│   └── data/               # Código para procesamiento de datos
├── utils/                  # Utilidades compartidas
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

## Datos

El proyecto analiza los siguientes conjuntos de datos:

- **Usuarios**: Información demográfica y de actividad de usuarios
- **Contenido**: Catálogo de películas y series disponibles
- **Engagement**: Interacciones de los usuarios con el contenido (likes, comentarios, etc.)
- **Actividad**: Patrones de uso de la plataforma
- **Campañas**: Métricas de campañas de marketing

## Funcionalidades Principales

El proyecto incluye análisis de:

- **Retención de usuarios**: Análisis de permanencia de usuarios por segmentos y tiempo
- **Engagement**: Métricas de interacción con el contenido por tipo y categoría
- **Segmentación**: Clasificación de usuarios por patrones de consumo
- **Campañas de Marketing**: Evaluación de efectividad y conversión
- **Patrones de Actividad**: Análisis de uso de la plataforma por día y hora

## Requisitos

Las dependencias principales del proyecto son:

- **Análisis de Datos:**
  - pandas (2.2.3)
  - numpy (2.2.4)
  - scipy (1.15.2)

- **Visualización:**
  - matplotlib (3.10.1)
  - seaborn (0.13.2)
  - plotly (6.0.1)

- **Procesamiento de Texto:**
  - ftfy (6.0.2)

- **Entorno Jupyter:**
  - ipykernel (6.29.5)
  - notebook (7.0.0)
  - jupyter (1.0.0)
  - IPython (9.1.0)

- **Utilidades:**
  - pathlib (1.0.1)
  - typing_extensions (4.13.1)

Para instalar todas las dependencias:

```bash
pip install -r requirements.txt
```

## Notebooks

El proyecto incluye los siguientes notebooks de análisis:

- `00_data_dictionary.ipynb`: Descripción de los conjuntos de datos
- `01_eda_cleaning.ipynb`: Exploración y limpieza de datos
- `02_retention.ipynb`: Análisis de retención de usuarios
- `03_engangement.ipynb`: Análisis de métricas de engagement
- `04_active_vs_old_users.ipynb`: Comparación entre usuarios activos y antiguos
- `05_campaigns.ipynb`: Análisis de campañas de marketing
- `06_activity.ipynb`: Patrones de actividad y uso
- `07_age_and_gender.ipynb`: Análisis demográfico de usuarios

## Módulos de Análisis

El proyecto está organizado en módulos específicos:

- `src/data/cleaning.py`: Funciones para limpieza y preprocesamiento de datos
- `src/analysis/campaigns.py`: Análisis de campañas de marketing
- `src/analysis/engagement.py`: Análisis de métricas de engagement
- `src/analysis/retention.py`: Análisis de retención de usuarios
- `src/analysis/content.py`: Análisis de contenido
- `src/analysis/segmentation.py`: Segmentación de usuarios

## Visualizaciones

Las visualizaciones generadas incluyen:

- Curvas de retención por país y segmento
- Heatmaps de patrones de actividad
- Análisis comparativo de campañas de marketing
- Distribución demográfica de usuarios
- Métricas de engagement por tipo de contenido y categoría

## Informes

Los resultados del análisis se almacenan en el directorio `reports/`:

- `figures/`: Gráficos y visualizaciones en formato PNG
- `tables/`: Tablas de resultados en formatos HTML y PNG

## Rutas y Compatibilidad

El proyecto utiliza rutas relativas para garantizar la compatibilidad entre diferentes entornos:

- Los notebooks acceden a los módulos y datos mediante rutas relativas (`../data/`, `../src/`, etc.)
- Las funciones de guardado en `utils/utils_peliplat.py` usan rutas relativas predeterminadas para almacenar resultados
- El módulo `cleaning.py` utiliza `pathlib` para manejar rutas de forma compatible con diferentes sistemas operativos

### Uso en Google Colab

Para usar este proyecto en Google Colab, se recomienda:

1. Clonar el repositorio en el entorno de Colab:
   ```python
   !git clone https://github.com/usuario/peliplat-analysis.git
   %cd peliplat-analysis
   ```

2. Instalar las dependencias:
   ```python
   !pip install -r requirements.txt
   ```

3. Si es necesario, ajustar las rutas en los notebooks de Colab:
   ```python
   import os
   import sys
   
   # Añadir el directorio raíz al path para poder importar los módulos
   if os.getcwd() not in sys.path:
       sys.path.append(os.getcwd())
   ```

4. Verificar que las rutas a los archivos de datos sean correctas según la estructura del repositorio clonado.
