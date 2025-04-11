import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List
from utils.utils_peliplat import PALETA_MARCA, COLORES_PELIPLAT, guardar_figura

def calcular_metricas_engagement(df_content: pd.DataFrame, 
                               df_engagement: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las métricas de engagement combinando datos de contenido y engagement.
    
    Args:
        df_content (pd.DataFrame): DataFrame con información de contenido
        df_engagement (pd.DataFrame): DataFrame con información de engagement
    
    Returns:
        pd.DataFrame: DataFrame con métricas de engagement calculadas
    """
    # Agrupar engagement por content_id
    engagement_agrupado = df_engagement.groupby('content_id').agg({
        'likes': 'sum',
        'shares': 'sum',
        'comments': 'sum',
        'user_id': 'count'
    }).reset_index()
    
    # Renombrar columna de conteo de usuarios
    engagement_agrupado.rename(columns={'user_id': 'interacciones'}, inplace=True)
    
    # Merge con contenido
    df = pd.merge(df_content, engagement_agrupado, on='content_id', how='left')
    
    # Rellenar NaNs con ceros
    columnas_engagement = ['likes', 'shares', 'comments', 'interacciones']
    df[columnas_engagement] = df[columnas_engagement].fillna(0)
    
    # Calcular métricas agregadas
    df['total_engagement'] = df['likes'] + df['shares'] + df['comments']
    df['engagement_rate'] = df['total_engagement'] / df['visits'] * 100
    
    return df

def analizar_engagement_por_grupo(df: pd.DataFrame, 
                                grupo_columna: str) -> pd.DataFrame:
    """
    Analiza las métricas de engagement agrupadas por una columna específica.
    
    Args:
        df (pd.DataFrame): DataFrame con métricas de engagement
        grupo_columna (str): Nombre de la columna por la que agrupar
    
    Returns:
        pd.DataFrame: DataFrame con análisis por grupo
    """
    return df.groupby(grupo_columna).agg({
        'content_id': 'count',
        'visits': 'mean',
        'likes': 'mean',
        'shares': 'mean',
        'comments': 'mean',
        'total_engagement': 'mean',
        'engagement_rate': 'mean'
    }).reset_index().rename(columns={
        'content_id': 'cantidad',
        'visits': 'p_visitas',
        'likes': 'p_likes',
        'shares': 'p_shares',
        'comments': 'p_comments',
        'total_engagement': 'p_engagement_total',
        'engagement_rate': 'p_tasa_engagement(%)'
    }).sort_values('p_engagement_total', ascending=False)

def obtener_top_contenidos(df: pd.DataFrame, 
                         n: int = 10) -> pd.DataFrame:
    """
    Obtiene los N contenidos con mayor engagement.
    
    Args:
        df (pd.DataFrame): DataFrame con métricas de engagement
        n (int): Número de contenidos a retornar
    
    Returns:
        pd.DataFrame: Top N contenidos por engagement
    """
    columnas = ['content_id', 'type', 'category', 'visits', 
               'likes', 'shares', 'comments', 
               'total_engagement', 'engagement_rate']
    
    return df.sort_values('total_engagement', ascending=False).head(n)[columnas]

def graficar_engagement_cruzado(df: pd.DataFrame,
                              x_col: str = 'type',
                              hue_col: str = 'category',
                              y_col: str = 'total_engagement',
                              titulo: str = None,
                              subtitulo: str = None) -> Tuple[plt.Figure, sns.FacetGrid]:
    """
    Crea un gráfico de barras para visualizar el engagement cruzado.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de engagement
        x_col (str): Columna para el eje X
        hue_col (str): Columna para el color de las barras
        y_col (str): Columna para el eje Y
        titulo (str): Título principal del gráfico
        subtitulo (str): Subtítulo del gráfico
    
    Returns:
        Tuple[plt.Figure, sns.FacetGrid]: Figura y gráfico generados
    """
    # Preparar datos
    engagement_cruzado = df.groupby([x_col, hue_col])[y_col].mean().reset_index()
    engagement_promedio = df.groupby(x_col)[y_col].mean().reset_index()
    
    # Crear figura
    fig = plt.figure(figsize=(12, 8))
    
    # Crear gráfico
    chart = sns.catplot(
        data=engagement_cruzado,
        kind="bar",
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=PALETA_MARCA,  # Usar paleta de Peliplat
        height=6,
        aspect=1.5
    )
    
    # Configurar etiquetas y títulos
    chart.set_xlabels(x_col.capitalize(), fontsize=12)
    chart.set_ylabels(y_col.replace('_', ' ').capitalize(), fontsize=12)
    
    # Configurar leyenda
    chart._legend.remove()
    chart.figure.legend(title=hue_col.capitalize(), 
                       loc='upper right', 
                       bbox_to_anchor=(0.99, 0.90))
    
    # Títulos
    if titulo:
        plt.suptitle(titulo, fontsize=14, y=0.98)
    if subtitulo:
        chart.figure.text(0.5, 0.92, subtitulo,
                         ha='center', fontsize=10, style='italic',
                         color=COLORES_PELIPLAT['texto'])  # Usar color de texto Peliplat
    
    # Añadir valores en barras
    ax = chart.axes[0,0]
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f', fontsize=9)
    
    # Añadir promedios
    for i, tipo in enumerate(engagement_promedio[x_col].unique()):
        promedio = engagement_promedio.loc[engagement_promedio[x_col] == tipo, y_col].values[0]
        ax.text(i, -0.5, f'Promedio: {promedio:.1f}',
                ha='center', va='top', fontsize=11,
                color=COLORES_PELIPLAT['gris_oscuro'])  # Usar color Peliplat
    
    # Ajustes finales
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    ax.set_ylim(bottom=-1)
    
    return fig, chart

def calcular_engagement_por_periodo(df: pd.DataFrame,
                                  fecha_columna: str = 'publish_date',
                                  periodo: str = 'M') -> pd.DataFrame:
    """
    Calcula las métricas de engagement agrupadas por período temporal.
    
    Args:
        df (pd.DataFrame): DataFrame con métricas de engagement
        fecha_columna (str): Nombre de la columna de fecha
        periodo (str): Período de agrupación ('D' para día, 'W' para semana, 'M' para mes)
    
    Returns:
        pd.DataFrame: DataFrame con métricas por período
    """
    return df.set_index(fecha_columna).resample(periodo).agg({
        'content_id': 'count',
        'visits': 'sum',
        'likes': 'sum',
        'shares': 'sum',
        'comments': 'sum',
        'total_engagement': 'sum'
    }).reset_index().rename(columns={
        'content_id': 'cantidad_contenidos',
        'visits': 'total_visitas'
    })

def calcular_nivel_engagement(df, columna_score='engagement_score', n_grupos=3):
    """
    Categoriza usuarios por nivel de engagement.
    
    Args:
        df (pd.DataFrame): DataFrame con columna de engagement score
        columna_score (str): Nombre de la columna con el score
        n_grupos (int): Número de grupos para la categorización
    
    Returns:
        pd.DataFrame: DataFrame con nueva columna 'nivel_engagement'
    """
    df = df.copy()
    df['nivel_engagement'] = pd.qcut(
        df[columna_score],
        q=n_grupos,
        labels=['Bajo', 'Medio', 'Alto']
    )
    return df

def analizar_engagement_retenidos(df, dias_retencion=30, n_grupos=3):
    """Analiza engagement para usuarios retenidos"""
    usuarios_retenidos = df[df['days_active'] >= dias_retencion].copy()
    usuarios_retenidos['nivel_engagement'] = pd.qcut(
        usuarios_retenidos['engagement_score'],
        q=n_grupos,
        labels=['Bajo', 'Medio', 'Alto']
    )
    return usuarios_retenidos

def graficar_distribucion_engagement_segmento(df, segmento_col, nivel_col, 
                                            titulo='Distribución de Niveles de Engagement',
                                            nombre_archivo=None):
    """
    Crea gráfico de barras para distribución de engagement por segmento.
    
    Args:
        df (pd.DataFrame): DataFrame con datos
        segmento_col (str): Columna de segmentación
        nivel_col (str): Columna de nivel de engagement
        titulo (str): Título del gráfico
        nombre_archivo (str): Nombre para guardar el gráfico
    """
    distribucion = pd.crosstab(
        df[segmento_col],
        df[nivel_col],
        normalize='index'
    ) * 100

    plt.figure(figsize=(12, 8))
    ax = distribucion.plot(
        kind='bar',
        color=PALETA_MARCA[:3],
        width=0.7
    )

    plt.title(titulo, pad=20, fontsize=14)
    plt.xlabel('Segmento', labelpad=10, fontsize=12)
    plt.ylabel('Porcentaje de Usuarios (%)', labelpad=10, fontsize=12)
    plt.legend(title='Nivel de Engagement', fontsize=10, title_fontsize=10)
    plt.grid(True, alpha=0.3)

    for container in ax.containers:
        plt.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

    plt.margins(x=0.1)
    plt.ylim(0, 100)
    plt.tight_layout()

    if nombre_archivo:
        guardar_figura(plt.gcf(), nombre_archivo)

def preparar_datos_engagement(engagement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara datos de engagement por usuario sumando interacciones.
    
    Args:
        engagement_df (pd.DataFrame): DataFrame con datos de engagement
    
    Returns:
        pd.DataFrame: DataFrame con métricas de engagement por usuario
    """
    engagement_user = engagement_df.groupby('user_id')[['likes', 'shares', 'comments']].sum().reset_index()
    engagement_user['engagement_score'] = engagement_user[['likes', 'shares', 'comments']].sum(axis=1)
    return engagement_user

def calcular_metricas_engagement(df: pd.DataFrame, 
                               columnas_engagement: List[str] = None) -> pd.DataFrame:
    """
    Calcula estadísticas de engagement por segmento.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de usuarios y engagement
        columnas_engagement (List[str]): Lista de columnas de engagement a analizar
        
    Returns:
        pd.DataFrame: DataFrame con métricas promedio por segmento
    """
    if columnas_engagement is None:
        columnas_engagement = ['likes', 'shares', 'comments', 'engagement_score']
    
    return df.groupby('segmento_visitas')[columnas_engagement].mean().round(2)

def calcular_distribucion_engagement(df):
    """Calcula distribución de niveles de engagement por segmento"""
    return pd.crosstab(
        df['segmento_visitas'],
        df['nivel_engagement'],
        normalize='index'
    ) * 100

def analizar_engagement_usuarios_retenidos(df: pd.DataFrame, 
                                         dias_retencion: int = 30,
                                         n_grupos: int = 3) -> tuple:
    """
    Analiza el engagement para usuarios retenidos por un período específico.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de usuarios
        dias_retencion (int): Días mínimos de retención
        n_grupos (int): Número de grupos para categorizar engagement
    
    Returns:
        tuple: (DataFrame con distribución, DataFrame con estadísticas)
    """
    # Filtrar usuarios retenidos
    usuarios_retenidos = df[df['days_active'] >= dias_retencion].copy()
    
    # Categorizar engagement
    usuarios_retenidos['nivel_engagement'] = pd.qcut(
        usuarios_retenidos['engagement_score'],
        q=n_grupos,
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # Calcular distribución
    distribucion = pd.crosstab(
        usuarios_retenidos['segmento_visitas'],
        usuarios_retenidos['nivel_engagement'],
        normalize='index'
    ) * 100
    
    # Calcular estadísticas
    stats = usuarios_retenidos.groupby('segmento_visitas')['engagement_score'].describe().round(2)
    
    return distribucion, stats

def graficar_distribucion_engagement_retenidos(distribucion: pd.DataFrame,
                                             titulo: str = None,
                                             nombre_archivo: str = None):
    """
    Grafica la distribución de engagement para usuarios retenidos.
    
    Args:
        distribucion (pd.DataFrame): DataFrame con distribución de engagement
        titulo (str): Título del gráfico
        nombre_archivo (str): Nombre para guardar la figura
    """
    plt.figure(figsize=(12, 8))
    ax = distribucion.plot(
        kind='bar',
        color=PALETA_MARCA[:3],
        width=0.7
    )

    plt.title(titulo or 'Distribución de Niveles de Engagement en Usuarios Retenidos (30 días)', 
             pad=20, fontsize=14)
    plt.xlabel('Segmento', labelpad=10, fontsize=12)
    plt.ylabel('Porcentaje de Usuarios (%)', labelpad=10, fontsize=12)
    plt.legend(title='Nivel de Engagement', fontsize=10, title_fontsize=10)
    plt.grid(True, alpha=0.3)

    # Añadir etiquetas de porcentaje
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

    # Ajustes de formato
    plt.margins(x=0.1)
    plt.ylim(0, 60)  # Topear en 60%
    plt.xticks(rotation=0)  # Labels horizontales
    plt.tight_layout()

    if nombre_archivo:
        guardar_figura(plt.gcf(), nombre_archivo)
