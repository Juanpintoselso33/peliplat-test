import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import pandas as pd
from IPython.display import HTML, display

# Paleta de colores de Peliplat
COLORES_PELIPLAT = {
    'principal': '#333333',     # Color oscuro del header
    'secundario': '#f5a676',    # Color durazno/naranja claro
    'azul': '#3b7cc5',          # Azul para enlaces/botones
    'blanco': '#ffffff',        # Blanco para fondos
    'gris_claro': '#f5f5f5',    # Gris muy claro para fondos alternativos
    'gris_medio': '#cccccc',    # Gris para bordes y separadores
    'texto': '#333333',         # Color de texto principal
    'accent': '#e74c3c',        # Color de acento (rojo) para destacados
    'turquesa': '#5CCDC9',      # Color turquesa/verde azulado de la marca
    'gris_oscuro': '#3c3c3c',   # Gris oscuro del header
    'amarillo': '#D9BE4A'       # Amarillo/dorado de la marca
}

# Nuevas paletas categóricas basadas en los colores de la marca
PALETA_MARCA = [COLORES_PELIPLAT['turquesa'], COLORES_PELIPLAT['gris_oscuro'], COLORES_PELIPLAT['amarillo'], COLORES_PELIPLAT['secundario']]

# Paletas categóricas inspiradas en la web
PALETA_CATEGORIAS = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']  # Paleta base para categorías
PALETA_VIRIDIS = sns.color_palette("viridis", 10)  # Alternativa para datos continuos

# Configuración global de estilos
def configurar_estilo_peliplat():
    """Configura el estilo global para gráficos de Peliplat"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Parámetros de estilo personalizados
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.labelcolor'] = COLORES_PELIPLAT['texto']
    plt.rcParams['axes.edgecolor'] = COLORES_PELIPLAT['gris_medio']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = COLORES_PELIPLAT['gris_medio']
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['xtick.color'] = COLORES_PELIPLAT['texto']
    plt.rcParams['ytick.color'] = COLORES_PELIPLAT['texto']
    plt.rcParams['figure.facecolor'] = COLORES_PELIPLAT['blanco']
    plt.rcParams['axes.facecolor'] = COLORES_PELIPLAT['blanco']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Configura seaborn para complementar
    sns.set_context("notebook", font_scale=1.1)

def guardar_figura(fig, nombre_archivo, ruta_guardado='../reports/figures'):
    """
    Guarda una figura en la ruta especificada
    """
    if not os.path.exists(ruta_guardado):
        os.makedirs(ruta_guardado)
        
    ruta_completa = os.path.join(ruta_guardado, nombre_archivo)
    fig.savefig(ruta_completa, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {ruta_completa}")

def estilizar_tabla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el estilo de Peliplat a un DataFrame para su visualización.
    
    Args:
        df (pd.DataFrame): DataFrame a estilizar
    
    Returns:
        pd.DataFrame: DataFrame estilizado
    """
    return df.style.set_properties(**{
        'background-color': COLORES_PELIPLAT['blanco'],
        'color': COLORES_PELIPLAT['texto'],
        'border-color': COLORES_PELIPLAT['gris_medio']
    }).set_table_styles([
        {'selector': 'th',
         'props': [('background-color', COLORES_PELIPLAT['gris_oscuro']),
                  ('color', COLORES_PELIPLAT['blanco']),
                  ('font-weight', 'bold'),
                  ('text-align', 'center')]},
        {'selector': 'td',
         'props': [('text-align', 'center')]},
        {'selector': '',
         'props': [('border', f'1px solid {COLORES_PELIPLAT["gris_medio"]}'),
                  ('font-family', 'Arial, Helvetica, sans-serif')]}
    ]).format(precision=2)  # Formatea números con 2 decimales

def guardar_tabla(df: pd.DataFrame, 
                 nombre_archivo: str, 
                 ruta_guardado: str = '../reports/tables',
                 formato: str = 'both') -> None:
    """
    Guarda una tabla estilizada en formato HTML y/o PNG.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        nombre_archivo (str): Nombre del archivo (sin extensión)
        ruta_guardado (str): Ruta donde guardar el archivo
        formato (str): Formato de guardado ('html', 'png' o 'both')
    """
    if not os.path.exists(ruta_guardado):
        os.makedirs(ruta_guardado)
    
    nombre_base = nombre_archivo.split('.')[0]
    
    if formato in ['html', 'both']:
        # Guardar en HTML
        ruta_html = os.path.join(ruta_guardado, f"{nombre_base}.html")
        tabla_html = estilizar_tabla(df).to_html()
        with open(ruta_html, 'w', encoding='utf-8') as f:
            f.write(tabla_html)
        print(f"Tabla guardada en HTML: {ruta_html}")
    
    if formato in ['png', 'both']:
        # Guardar en PNG
        ruta_png = os.path.join(ruta_guardado, f"{nombre_base}.png")
        
        # Crear una figura con la tabla estilizada
        fig, ax = plt.subplots(figsize=(12, len(df)*0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        tabla = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=[COLORES_PELIPLAT['gris_oscuro']]*len(df.columns))
        
        # Ajustar estilo de la tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.2, 1.5)
        
        # Guardar figura
        plt.savefig(ruta_png, 
                   bbox_inches='tight',
                   dpi=300,
                   facecolor=COLORES_PELIPLAT['blanco'])
        plt.close()
        print(f"Tabla guardada en PNG: {ruta_png}")

def mostrar_tabla(df: pd.DataFrame, 
                 titulo: str = None, 
                 guardar: bool = False, 
                 nombre_archivo: str = None,
                 formato: str = 'both') -> None:
    """
    Muestra un DataFrame con el estilo de Peliplat y opcionalmente lo guarda.
    
    Args:
        df (pd.DataFrame): DataFrame a mostrar
        titulo (str, optional): Título para la tabla
        guardar (bool): Si se debe guardar la tabla
        nombre_archivo (str): Nombre del archivo si se guarda
        formato (str): Formato de guardado ('html', 'png' o 'both')
    """
    if titulo:
        print(f"\n{titulo}")
    
    # Crear tabla estilizada
    tabla_estilizada = estilizar_tabla(df)
    
    try:
        # Intentar usar display de IPython
        from IPython.display import display
        display(tabla_estilizada)
    except ImportError:
        # Si no está disponible, usar el método to_string de pandas
        print(df.to_string())
    
    # Guardar si se solicita
    if guardar and nombre_archivo:
        guardar_tabla(df, nombre_archivo, formato=formato)

