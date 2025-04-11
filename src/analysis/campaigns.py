import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_peliplat import PALETA_MARCA, guardar_figura

def preparar_datos_campanas(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos de campañas calculando métricas adicionales.
    
    Args:
        campaigns_df (pd.DataFrame): DataFrame con datos de campañas
    
    Returns:
        pd.DataFrame: DataFrame con métricas calculadas
    """
    df = campaigns_df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
    df['conversion_rate'] = (df['signups'] / df['clicks'] * 100).round(2)
    df['daily_signups'] = (df['signups'] / df['duration_days']).round(2)
    return df

def calcular_eficiencia_campanas(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de eficiencia por campaña.
    
    Args:
        campaigns_df (pd.DataFrame): DataFrame con datos de campañas
    
    Returns:
        pd.DataFrame: DataFrame con métricas de eficiencia
    """
    return pd.DataFrame({
        'Campaña': campaigns_df['name'],
        'Duración (días)': campaigns_df['duration_days'],
        'Total Signups': campaigns_df['signups'],
        'Registros/día': campaigns_df['daily_signups']
    }).sort_values('Registros/día', ascending=False)

def graficar_ctr_conversion(df: pd.DataFrame, nombre_archivo: str = None):
    """
    Grafica CTR vs Tasa de Conversión por campaña.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de campañas
        nombre_archivo (str): Nombre para guardar la figura
    """
    plt.figure(figsize=(12, 7))
    x = np.arange(len(df['name']))
    width = 0.35

    plt.bar(x - width/2, df['CTR'] * 100, width, label='CTR', color=PALETA_MARCA[0])
    plt.bar(x + width/2, df['conversion_rate'], width, label='Tasa de Conversión', color=PALETA_MARCA[1])

    plt.title('CTR vs Tasa de Conversión por Campaña', pad=20)
    plt.xlabel('Campaña')
    plt.ylabel('Porcentaje (%)')
    plt.xticks(x, df['name'])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Añadir etiquetas en las barras
    for i, v in enumerate(df['CTR'] * 100):
        plt.text(i - width/2, v, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(df['conversion_rate']):
        plt.text(i + width/2, v, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    if nombre_archivo:
        guardar_figura(plt.gcf(), nombre_archivo)

def graficar_registros_diarios(df: pd.DataFrame, nombre_archivo: str = None):
    """
    Grafica promedio de registros diarios por campaña.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de campañas
        nombre_archivo (str): Nombre para guardar la figura
    """
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df['name'], df['daily_signups'], color=PALETA_MARCA[2], width=0.7)

    plt.title('Promedio de Registros Diarios por Campaña', pad=20)
    plt.xlabel('Campaña')
    plt.ylabel('Registros por Día')
    plt.grid(True, alpha=0.3)

    # Añadir etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()
    if nombre_archivo:
        guardar_figura(plt.gcf(), nombre_archivo)

def calcular_calidad_usuarios(users_df: pd.DataFrame, 
                            activity_df: pd.DataFrame,
                            engagement_df: pd.DataFrame,
                            campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de calidad de usuarios por campaña.
    
    Args:
        users_df (pd.DataFrame): DataFrame de usuarios
        activity_df (pd.DataFrame): DataFrame de actividad
        engagement_df (pd.DataFrame): DataFrame de engagement
        campaigns_df (pd.DataFrame): DataFrame de campañas
    
    Returns:
        pd.DataFrame: DataFrame con métricas de calidad
    """
    # Unir datos de actividad
    users_complete = users_df.merge(activity_df, on='user_id', how='left')
    
    # Calcular engagement
    engagement_user = engagement_df.groupby('user_id').agg({
        'likes': 'sum',
        'shares': 'sum',
        'comments': 'sum'
    }).reset_index()
    engagement_user['engagement_score'] = engagement_user[['likes', 'shares', 'comments']].sum(axis=1)
    
    # Unir datos de engagement
    users_complete = users_complete.merge(engagement_user, on='user_id', how='left')
    
    # Rellenar NaN con 0 en métricas de engagement
    engagement_columns = ['likes', 'shares', 'comments', 'engagement_score']
    users_complete[engagement_columns] = users_complete[engagement_columns].fillna(0)
    
    # Convertir fechas para asignar usuarios a campañas
    users_complete['signup_date'] = pd.to_datetime(users_complete['signup_date'])
    campaigns_df['start_date'] = pd.to_datetime(campaigns_df['start_date'])
    campaigns_df['end_date'] = pd.to_datetime(campaigns_df['end_date'])
    
    # Asignar usuarios a campañas según fecha de registro
    def asignar_campana(fecha_registro):
        for _, campaign in campaigns_df.iterrows():
            if campaign['start_date'] <= fecha_registro <= campaign['end_date']:
                return campaign['name']
        return None
    
    users_complete['campana'] = users_complete['signup_date'].apply(asignar_campana)
    
    # Calcular métricas por campaña
    return users_complete.groupby('campana').agg({
        'days_active': 'mean',
        'avg_daily_visits': 'mean',
        'engagement_score': 'mean',
        'user_id': 'count'
    }).rename(columns={
        'days_active': 'Días Activos',
        'avg_daily_visits': 'Visitas Diarias',
        'engagement_score': 'Engagement Score',
        'user_id': 'Total Usuarios'
    }).round(2)
