import pandas as pd
from typing import List

def preparar_datos_usuarios(users_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina datos de usuarios con actividad.
    
    Args:
        users_df (pd.DataFrame): DataFrame de usuarios
        activity_df (pd.DataFrame): DataFrame de actividad
    
    Returns:
        pd.DataFrame: DataFrame combinado de usuarios y actividad
    """
    return users_df.merge(activity_df, on='user_id', how='left')

def segmentar_usuarios_por_visitas(df: pd.DataFrame, umbral: float = 1) -> pd.DataFrame:
    """
    Segmenta usuarios en 'Nuevo' y 'Recurrente' basado en promedio de visitas diarias.
    
    Args:
        df (pd.DataFrame): DataFrame con columna 'avg_daily_visits'
        umbral (float): Umbral para considerar usuario recurrente (default: 1)
    
    Returns:
        pd.DataFrame: DataFrame con nueva columna 'segmento_visitas'
    """
    df = df.copy()
    df['segmento_visitas'] = df['avg_daily_visits'].apply(
        lambda x: 'Nuevo' if x <= umbral else 'Recurrente'
    )
    return df

def calcular_distribucion_segmentos(df: pd.DataFrame, columna_segmento: str = 'segmento_visitas') -> pd.DataFrame:
    """
    Calcula la distribución de usuarios por segmento.
    
    Args:
        df (pd.DataFrame): DataFrame con columna de segmentación
        columna_segmento (str): Nombre de la columna de segmentación
    
    Returns:
        pd.DataFrame: DataFrame con conteo por segmento
    """
    return pd.DataFrame(
        df[columna_segmento].value_counts()
    ).reset_index().rename(
        columns={'index': 'Segmento', columna_segmento: 'Cantidad de usuarios'}
    )

def calcular_estadisticas_segmento(df: pd.DataFrame, 
                                 columnas_metricas: List[str],
                                 columna_segmento: str = 'segmento_visitas') -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas por segmento.
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        columnas_metricas (List[str]): Lista de columnas para calcular estadísticas
        columna_segmento (str): Nombre de la columna de segmentación
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas por segmento
    """
    return df.groupby(columna_segmento)[columnas_metricas].agg(['mean', 'median', 'std']).round(2)

def analizar_segmentos(df: pd.DataFrame, 
                      columnas_metricas: List[str],
                      columna_segmento: str = 'segmento_visitas') -> tuple:
    """
    Realiza análisis completo de segmentos incluyendo distribución y estadísticas.
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        columnas_metricas (List[str]): Lista de columnas para calcular estadísticas
        columna_segmento (str): Nombre de la columna de segmentación
    
    Returns:
        tuple: (DataFrame de distribución, DataFrame de estadísticas)
    """
    distribucion = calcular_distribucion_segmentos(df, columna_segmento)
    estadisticas = calcular_estadisticas_segmento(df, columnas_metricas, columna_segmento)
    return distribucion, estadisticas

