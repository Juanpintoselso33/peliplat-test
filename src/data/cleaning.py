import pandas as pd
import ftfy
from pathlib import Path

def cast_dates(df, cols):
    """
    Convierte columnas de texto a formato de fecha.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar.
        cols (list): Lista de nombres de columnas a convertir.
        
    Returns:
        pd.DataFrame: DataFrame con las columnas convertidas a fechas.
    """
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def fix_encoding(df):
    """
    Corrige problemas de codificación en columnas de texto.
    
    Args:
        df (pd.DataFrame): DataFrame con problemas de codificación.
        
    Returns:
        pd.DataFrame: DataFrame con la codificación corregida.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)
    return df


def remove_duplicates(
    df, 
    subset=None, 
    keep='first', 
    consolidate=False, 
    agg_functions=None
):
    """
    Elimina o consolida filas duplicadas de un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar.
        subset (list, optional): Columnas a considerar para identificar duplicados.
        keep (str, optional): 'first' mantiene el primer duplicado, 'last' el último,
                             False elimina todos los duplicados.
        consolidate (bool, optional): Si es True, consolida los duplicados.
        agg_functions (dict, optional): Funciones de agregación para consolidación.
              
    Returns:
        tuple: (DataFrame procesado, DataFrame con duplicados encontrados).
    """
    # Identificar todas las filas duplicadas (originales y copias)
    all_duplicates = df.duplicated(subset=subset, keep=False)
    df_all_duplicates = df[all_duplicates].copy() if any(all_duplicates) else pd.DataFrame()
    
    # Almacenar duplicados que serán eliminados o procesados
    df_duplicados = df[df.duplicated(subset=subset, keep=keep)].copy() if not consolidate else df_all_duplicates.copy()
    
    # Mostrar información sobre duplicados
    if len(df_all_duplicates) > 0:
        n_grupos = df_all_duplicates.groupby(subset).ngroups
        total_duplicados = len(df_all_duplicates) - n_grupos
        
        print(f"🔍 Se encontraron {total_duplicados} registros duplicados en {n_grupos} grupos.")
        
        # Mostrar cada grupo de duplicados
        for key, group in df_all_duplicates.groupby(subset):
            if len(group) > 1:
                print("\n⚠️ Duplicados encontrados:")
                print(group.to_string())
                print("-" * 50)
    
    # Procesar según la opción elegida
    if consolidate and len(df_all_duplicates) > 0:
        # Configurar funciones de agregación predeterminadas
        if agg_functions is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            agg_functions = {col: 'sum' for col in numeric_cols if col not in subset}
        
        print(f"🔄 Consolidando duplicados usando: {agg_functions}")
        
        try:
            # Procesar registros no duplicados
            non_duplicates = df[~all_duplicates].copy()
            
            # Agregar duplicados consolidados
            consolidated = df.groupby(subset).agg(agg_functions).reset_index()
            
            # Combinar ambos conjuntos
            df_procesado = pd.concat([non_duplicates, consolidated])
            
            print(f"✅ Registros después de consolidación: {len(df_procesado)}")
        except Exception as e:
            print(f"⚠️ Error durante consolidación: {e}")
            # Fallback: eliminar duplicados
            df_procesado = df.drop_duplicates(subset=subset, keep=keep)
            print(f"⚠️ Se aplicó eliminación de duplicados como alternativa")
    else:
        # Eliminar duplicados
        df_procesado = df.drop_duplicates(subset=subset, keep=keep)
    
    return df_procesado, df_duplicados


def remove_outliers(df, col, lower_quantile=0.01, upper_quantile=0.99):
    """
    Elimina valores atípicos (outliers) de una columna numérica.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar.
        col (str): Nombre de la columna a procesar.
        lower_quantile (float): Cuantil inferior para definir outliers.
        upper_quantile (float): Cuantil superior para definir outliers.
        
    Returns:
        pd.DataFrame: DataFrame sin outliers en la columna especificada.
    """
    try:
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        filtered_df = df[(df[col] >= lower) & (df[col] <= upper)]
        return filtered_df
    except Exception as e:
        print(f"⚠️ Error removiendo outliers de {col}: {e}")
        return df


def winsorize_outliers(df, column, iqr_factor=3.0, verbose=True):
    """
    Identifica outliers y los winsoriza (recorta a los límites).
    
    Args:
        df (pd.DataFrame): DataFrame a procesar.
        column (str): Nombre de la columna a procesar.
        iqr_factor (float): Factor multiplicador del IQR para definir outliers.
        verbose (bool): Muestra estadísticas detalladas si es True.
        
    Returns:
        pd.DataFrame: DataFrame con outliers winsorizados.
    """
    try:
        df_result = df.copy()
        
        # Calcular límites
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        # Identificar outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        if verbose:
            print(f"\n📊 Estadísticas de {column} antes de winsorización:")
            print(df[column].describe())
            print(f"🔍 Límites para outliers (IQR x {iqr_factor}): {lower_bound:.2f} - {upper_bound:.2f}")
            print(f"🔢 Número de outliers detectados: {len(outliers)}")
            
            if len(outliers) > 0:
                print("⚠️ Ejemplos de outliers:")
                print(outliers.head())
        
        if len(outliers) > 0:
            # Aplicar winsorización
            df_result.loc[df_result[column] > upper_bound, column] = upper_bound
            df_result.loc[df_result[column] < lower_bound, column] = lower_bound
            
            if verbose:
                print(f"\n📈 Estadísticas después de winsorización:")
                print(df_result[column].describe())
        elif verbose:
            print("✅ No se detectaron outliers.")
        
        return df_result
        
    except Exception as e:
        print(f"⚠️ Error winsorizando {column}: {e}")
        return df


def process_outliers(dfs_dict, column_factors=None, verbose=True):
    """
    Procesa outliers en múltiples DataFrames usando winsorización.
    
    Args:
        dfs_dict (dict): Diccionario {nombre_dataframe: dataframe}.
        column_factors (dict): Diccionario anidado con factores IQR por columna.
                              Ejemplo: {'activity': {'avg_daily_visits': 3.0}}
        verbose (bool): Muestra estadísticas detalladas si es True.
    
    Returns:
        dict: Diccionario con DataFrames procesados.
    """
    if column_factors is None:
        column_factors = {}
    
    results = {}
    
    if verbose:
        print("=== 🔍 TRATAMIENTO DE OUTLIERS (MEJORES PRÁCTICAS) ===")
    
    try:
        # Procesar cada DataFrame
        for df_name, df in dfs_dict.items():
            if verbose:
                print(f"\n--- 📋 {df_name.upper()} ---")
            
            # Procesar columnas específicas si existen
            if df_name in column_factors:
                results[df_name] = df.copy()
                for column, factor in column_factors[df_name].items():
                    if verbose:
                        print(f"\n--- 📊 {df_name.upper()} ({column.upper()}) ---")
                    results[df_name] = winsorize_outliers(results[df_name], column, factor, verbose)
            else:
                results[df_name] = df.copy()
        
        # Mostrar recuento final
        if verbose:
            print("\n=== ✅ RECUENTO DE REGISTROS FINAL ===")
            for df_name, df in results.items():
                print(f"📋 {df_name}: {len(df)} registros")
        
        return results
        
    except Exception as e:
        print(f"⚠️ Error procesando outliers: {e}")
        # Devolver los DataFrames originales en caso de error
        return dfs_dict
    
def fix_temporal_inconsistencies(df, signup_col='signup_date', activity_col='last_active', verbose=True):
    """
    Corrige inconsistencias temporales donde la fecha de última actividad es anterior
    a la fecha de registro, ajustando la fecha de registro.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar.
        signup_col (str): Nombre de la columna de fecha de registro.
        activity_col (str): Nombre de la columna de fecha de última actividad.
        verbose (bool): Muestra estadísticas detalladas si es True.
        
    Returns:
        pd.DataFrame: DataFrame con inconsistencias temporales corregidas.
    """
    df_result = df.copy()
    
    # Calcular días activos
    df_result['days_active'] = (df_result[activity_col] - df_result[signup_col]).dt.days
    
    # Identificar inconsistencias temporales
    inconsistencias = df_result[df_result['days_active'] < 0].copy()
    
    if len(inconsistencias) > 0:
        if verbose:
            print(f"🔍 Se encontraron {len(inconsistencias)} registros con inconsistencias temporales.")
            print("\n⚠️ Ejemplos de inconsistencias:")
            cols_display = ['user_id', signup_col, activity_col, 'days_active'] if 'user_id' in df_result.columns else [signup_col, activity_col, 'days_active']
            print(inconsistencias[cols_display].head())
        
        # Corregir inconsistencias (ajustar fecha de registro a fecha de actividad)
        for idx, row in inconsistencias.iterrows():
            df_result.loc[idx, signup_col] = row[activity_col]
        
        # Recalcular días activos
        df_result['days_active'] = (df_result[activity_col] - df_result[signup_col]).dt.days
        
        if verbose:
            print(f"\n✅ Inconsistencias temporales corregidas. Registros afectados: {len(inconsistencias)}")
    elif verbose:
        print("✅ No se detectaron inconsistencias temporales.")
    
    return df_result


def save_processed_data(
    dfs_dict, 
    directory='../data/processed/', 
    prefix='peliplat_', 
    suffix='_clean', 
    encoding='utf-8', 
    verbose=True
):
    """
    Guarda múltiples DataFrames procesados en archivos CSV.
    
    Args:
        dfs_dict (dict): Diccionario {nombre_dataframe: dataframe}.
                        Ejemplo: {'users': users_df, 'activity': activity_df}
        directory (str): Ruta donde guardar los archivos.
        prefix (str): Prefijo para los nombres de archivo (ej: 'peliplat_').
        suffix (str): Sufijo para los nombres de archivo (ej: '_clean').
        encoding (str): Codificación para los archivos CSV.
        verbose (bool): Muestra progreso en consola si True.
    
    Returns:
        list: Rutas de los archivos guardados.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for df_name, df in dfs_dict.items():
        filename = f"{prefix}{df_name}{suffix}.csv"
        filepath = directory / filename
        
        try:
            df.to_csv(filepath, index=False, encoding=encoding)
            saved_files.append(str(filepath))
            if verbose:
                print(f"✅ Guardado exitoso: {filename} ({len(df)} registros)")
        except Exception as e:
            if verbose:
                print(f"⚠️ Error guardando {filename}: {e}")
    
    if verbose:
        print(f"\n📂 Archivos guardados en: {directory.resolve()}")
    
    return saved_files
