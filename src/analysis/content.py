import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_peliplat import PALETA_MARCA, guardar_figura

def obtener_categoria_preferida(content_df: pd.DataFrame, engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la categoría preferida por usuario basada en su historial.
    
    Args:
        content_df (pd.DataFrame): DataFrame de contenido
        engagement_df (pd.DataFrame): DataFrame de engagement
    
    Returns:
        pd.DataFrame: DataFrame con user_id y su categoría preferida
    """
    content_user = content_df.merge(engagement_df, on='content_id', how='right')
    content_user = content_user.groupby('user_id')['category'].agg(
        lambda x: x.value_counts().index[0]  # Obtener la categoría más frecuente
    ).reset_index()
    content_user.columns = ['user_id', 'categoria_preferida']
    return content_user

def calcular_distribucion_contenido(df: pd.DataFrame, 
                                  columna_contenido: str,
                                  columna_segmento: str = 'segmento_visitas') -> pd.DataFrame:
    """
    Calcula la distribución de tipos de contenido por segmento.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de usuarios
        columna_contenido (str): Nombre de la columna de contenido a analizar
        columna_segmento (str): Nombre de la columna de segmentación
    
    Returns:
        pd.DataFrame: DataFrame con distribución porcentual
    """
    return pd.crosstab(
        df[columna_segmento],
        df[columna_contenido],
        normalize='index'
    ).round(3) * 100

def graficar_preferencias_contenido(contenido_tipo: pd.DataFrame, 
                                  categoria_preferida: pd.DataFrame,
                                  nombre_archivo: str = None):
    """
    Crea un gráfico comparativo de preferencias de contenido por segmento.
    
    Args:
        contenido_tipo (pd.DataFrame): DataFrame con distribución de tipos de contenido
        categoria_preferida (pd.DataFrame): DataFrame con distribución de categorías
        nombre_archivo (str): Nombre para guardar la figura
    """
    # Crear una figura con título general
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Análisis de preferencias de contenido por tipo de usuario', 
                 fontsize=16, y=1.05)

    # Gráfico 1: Tipos de contenido por segmento
    contenido_tipo.plot(
        kind='bar',
        ax=ax1,
        color=PALETA_MARCA[:3],
        width=0.8
    )
    ax1.set_title('Formato de contenido más visto', pad=15)
    ax1.set_xlabel('Tipo de usuario')
    ax1.set_ylabel('Porcentaje de usuarios (%)')
    ax1.legend(title='Formato de contenido')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_ylim(0, 60)

    # Gráfico 2: Categorías preferidas por segmento
    categoria_preferida.plot(
        kind='bar',
        ax=ax2,
        color=PALETA_MARCA,
        width=0.8
    )
    ax2.set_title('Categoría de contenido preferida', pad=15)
    ax2.set_xlabel('Tipo de usuario')
    ax2.set_ylabel('Porcentaje de usuarios (%)')
    ax2.legend(title='Categoría de contenido')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_ylim(0, 60)

    # Ajustar layout y mostrar porcentajes en las barras
    for ax in [ax1, ax2]:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if nombre_archivo:
        guardar_figura(fig, nombre_archivo)