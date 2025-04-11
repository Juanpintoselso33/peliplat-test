import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_peliplat import PALETA_MARCA, guardar_figura


# Crear función para calcular retención
def calcular_retencion_periodo(df, fecha_registro, fecha_ultima_actividad, dias_activos, dias_periodo):
    """
    Calcula la tasa de retención para un período específico.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de usuarios
        fecha_registro (str): Nombre de la columna con fecha de registro
        fecha_ultima_actividad (str): Nombre de la columna con fecha de última actividad
        dias_activos (str): Nombre de la columna con días activos
        dias_periodo (int): Período para calcular la retención (7, 14, 30, etc.)
    
    Returns:
        float: Tasa de retención como porcentaje
    """
    fecha_max = df[fecha_ultima_actividad].max()
    umbral = pd.Timedelta(days=dias_periodo)
    usuarios_elegibles = df[df[fecha_registro] <= (fecha_max - umbral)]
    
    if len(usuarios_elegibles) == 0:
        return 0.0
        
    return (usuarios_elegibles[usuarios_elegibles[dias_activos] >= dias_periodo].shape[0] / 
            usuarios_elegibles.shape[0] * 100)

def calcular_retencion_por_grupo(df, grupo, fecha_registro, fecha_ultima_actividad, 
                                dias_activos, periodos=[7, 14, 30]):
    """
    Calcula la retención por grupo (campaña, país, etc.) para diferentes períodos.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de usuarios
        grupo (str): Nombre de la columna para agrupar (ej: 'campana', 'country')
        fecha_registro (str): Nombre de la columna con fecha de registro
        fecha_ultima_actividad (str): Nombre de la columna con fecha de última actividad
        dias_activos (str): Nombre de la columna con días activos
        periodos (list): Lista de períodos para calcular retención
    
    Returns:
        pd.DataFrame: DataFrame con tasas de retención por grupo y período
    """
    retencion_grupos = {}
    
    for nombre_grupo in df[grupo].unique():
        if pd.isna(nombre_grupo):
            continue
            
        usuarios_grupo = df[df[grupo] == nombre_grupo]
        retencion_grupos[nombre_grupo] = [
            calcular_retencion_periodo(
                usuarios_grupo, 
                fecha_registro, 
                fecha_ultima_actividad, 
                dias_activos, 
                dias
            ) for dias in periodos
        ]
    
    return pd.DataFrame(
        retencion_grupos,
        index=[f'{dias} días' for dias in periodos]
    ).round(2)

def calcular_retencion_por_pais(df, fecha_registro='signup_date', 
                               fecha_ultima_actividad='last_active', 
                               dias_activos='days_active', 
                               periodos=[7, 14, 30, 60, 89]):
    """
    Calcula la retención por país considerando la fecha máxima específica de cada país.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de usuarios
        fecha_registro (str): Nombre de la columna con fecha de registro
        fecha_ultima_actividad (str): Nombre de la columna con fecha de última actividad
        dias_activos (str): Nombre de la columna con días activos
        periodos (list): Lista de períodos para calcular retención
    
    Returns:
        pd.DataFrame: DataFrame con tasas de retención por país y período
    """
    retencion_paises = {}
    
    for pais in df['country'].unique():
        usuarios_pais = df[df['country'] == pais]
        fecha_max_pais = usuarios_pais[fecha_ultima_actividad].max()
        
        retencion_pais = []
        for dias in periodos:
            umbral = pd.Timedelta(days=dias)
            usuarios_elegibles = usuarios_pais[usuarios_pais[fecha_registro] <= (fecha_max_pais - umbral)]
            
            if len(usuarios_elegibles) == 0:
                retencion = None  # Usamos None en lugar de 0 para indicar ausencia de datos
            else:
                retencion = (usuarios_elegibles[usuarios_elegibles[dias_activos] >= dias].shape[0] / 
                           usuarios_elegibles.shape[0] * 100)
            
            retencion_pais.append(retencion)
            
        retencion_paises[pais] = retencion_pais
    
    df_retencion = pd.DataFrame(
        retencion_paises,
        index=[f'{dias} días' for dias in periodos]
    )
    
    return df_retencion.round(2)

def calcular_retencion_pais_serie(usuarios_pais, dias, fecha_registro='signup_date', 
                                 fecha_ultima_actividad='last_active', 
                                 dias_activos='days_active'):
    """
    Calcula la serie completa de retención para un país.
    
    Args:
        usuarios_pais (pd.DataFrame): DataFrame filtrado para un país
        dias (list): Lista de días para calcular la retención
        fecha_registro (str): Nombre columna fecha registro
        fecha_ultima_actividad (str): Nombre columna última actividad
        dias_activos (str): Nombre columna días activos
    
    Returns:
        tuple: (lista de días válidos, lista de retenciones correspondientes)
    """
    fecha_max_pais = usuarios_pais[fecha_ultima_actividad].max()
    retencion_pais = []
    dias_validos = []
    
    for d in dias:
        umbral = pd.Timedelta(days=d)
        usuarios_elegibles = usuarios_pais[usuarios_pais[fecha_registro] <= (fecha_max_pais - umbral)]
        
        if len(usuarios_elegibles) > 0:
            retencion = usuarios_elegibles[usuarios_elegibles[dias_activos] >= d].shape[0] / usuarios_elegibles.shape[0]
            if retencion > 0:  # Solo agregar si hay retención
                retencion_pais.append(retencion)
                dias_validos.append(d)
    
    return dias_validos, retencion_pais

def graficar_retencion_por_grupo(df_retencion: pd.DataFrame,
                                titulo: str = 'Retención por Segmento y Período',
                                nombre_archivo: str = None):
    """
    Grafica las tasas de retención por grupo.
    
    Args:
        df_retencion (pd.DataFrame): DataFrame con tasas de retención por grupo
        titulo (str): Título del gráfico
        nombre_archivo (str): Nombre para guardar la figura
    """
    plt.figure(figsize=(14, 20))
    ax = df_retencion.plot(
        kind='bar',
        color=PALETA_MARCA[:2],
        width=0.7
    )

    plt.title(titulo, pad=20, fontsize=14)
    plt.xlabel('Período', labelpad=10, fontsize=12)
    plt.ylabel('Tasa de Retención (%)', labelpad=10, fontsize=12)
    plt.legend(
        title='Segmento', 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        fontsize=10,
        title_fontsize=10
    )
    plt.grid(True, alpha=0.3)

    # Añadir etiquetas de porcentaje
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=8, fontsize=10)

    # Ajustar formato y rotar labels del eje X
    plt.xticks(range(len(df_retencion.index)), 
               [f'D{dias.split()[0]}' for dias in df_retencion.index],  # Cambiar formato a D7, D14, etc
               fontsize=10,
               rotation=0)  # Sin rotación
    plt.yticks(fontsize=10)
    plt.margins(x=0.1, y=0.2)
    plt.ylim(0, 110)
    plt.tight_layout()

    if nombre_archivo:
        guardar_figura(plt.gcf(), nombre_archivo)