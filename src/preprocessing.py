import os

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import OneHotEncoder
import osmnx as ox


###def load_accidents(filepath):
###    """Carica i dati degli incidenti da un file Excel."""
###    if not os.path.exists(filepath):
###        print(f"File dati incidenti non trovato: {filepath}")
###        return None
###    try:
###        df = pd.read_excel(filepath)
###        print(f"Dati incidenti caricati da {filepath}: {len(df)} righe.")
###        if df.empty:
###            print("Il file dei dati incidenti è vuoto.")
###            return None
###        return df
###    except Exception as e:
###        print(f"Errore durante il caricamento del file dati incidenti {filepath}: {e}")
###        return None


def preprocess_accidents(df_accidents):
    """
    Pre-processa il DataFrame degli incidenti stradali:
    - Rimuove righe con gravità non valida.
    - Gestisce valori mancanti (NaN) per la gravità.
    - Crea la colonna 'geometry' per la georeferenziazione.
    """
    print("Pre-elaborazione dei dati incidenti stradali...")
    if df_accidents is None or df_accidents.empty:
        print("DataFrame incidenti vuoto o None. Restituendo un GeoDataFrame vuoto.")
        # Ritorna un GeoDataFrame vuoto con le colonne attese
        return gpd.GeoDataFrame(columns=['Latitudine', 'Longitudine', 'Gravita', 'geometry'], crs="EPSG:4326")

    # Assicura che le colonne Latitudine e Longitudine siano numeriche
    df_accidents['Latitudine'] = pd.to_numeric(df_accidents['Latitudine'], errors='coerce')
    df_accidents['Longitudine'] = pd.to_numeric(df_accidents['Longitudine'], errors='coerce')

    # Rimuove righe dove Latitudine o Longitudine non sono valide
    df_accidents.dropna(subset=['Latitudine', 'Longitudine'], inplace=True)

    # Rimuove righe dove 'Gravita' non è un numero valido
    df_accidents = df_accidents[pd.to_numeric(df_accidents['Gravita'], errors='coerce').notna()]
    df_accidents['Gravita'] = df_accidents['Gravita'].astype(int)

    # Gestisce i valori mancanti nella colonna 'Gravita' (es. con la media)
    if df_accidents['Gravita'].isnull().any():
        df_accidents['Gravita'] = df_accidents['Gravita'].fillna(df_accidents['Gravita'].mean())
        print("Valori NaN nella colonna 'Gravita' gestiti con la media.")

    # Crea la colonna 'geometry' da Latitudine e Longitudine
    df_accidents['geometry'] = df_accidents.apply(
        lambda row: Point(row['Longitudine'], row['Latitudine']),
        axis=1
    )

    # Rimuove righe con geometria NaN (potrebbero esserci se Lat/Lon erano NaN prima della dropna)
    df_accidents.dropna(subset=['geometry'], inplace=True)

    # Converte in GeoDataFrame
    gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry='geometry', crs="EPSG:4326")

    print("Dati incidenti pre-elaborati con successo.")
    return gdf_accidents


def preprocess_traffic_data(df_traffic):
    """
    Pre-processa il DataFrame dei dati di traffico.
    - Converte le colonne di data e ora nel formato datetime.
    - Crea una colonna 'geometry' per la georeferenziazione.
    """
    print("Pre-elaborazione dei dati di traffico...")
    if df_traffic is None or df_traffic.empty:
        print("DataFrame traffico vuoto o None. Restituendo un GeoDataFrame vuoto.")
        # Ritorna un GeoDataFrame vuoto con le colonne attese
        return gpd.GeoDataFrame(columns=['Latitudine', 'Longitudine', 'DataRilevamento',
                                         'OraRilevamento', 'ConteggioVeicoli', 'VelocitaMedia',
                                         'IndiceCongestione', 'geometry'], crs="EPSG:4326")

    # Conversione a numerico per sicurezza
    for col in ['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']:
        if col in df_traffic.columns:
            df_traffic[col] = pd.to_numeric(df_traffic[col], errors='coerce')
            if col == 'ConteggioVeicoli':
                df_traffic[col] = df_traffic[col].astype('Int64') # 'Int64' supporta NaN

    # Corregge la scala di VelocitaMedia
    if 'VelocitaMedia' in df_traffic.columns and df_traffic['VelocitaMedia'].max() > 100: # Soglia ragionevole per identificare il problema
        df_traffic['VelocitaMedia'] = df_traffic['VelocitaMedia'] / 100.0
        df_traffic['VelocitaMedia'] = df_traffic['VelocitaMedia'].round(2) # Arrotonda per pulizia

    # Corregge la scala di IndiceCongestione
    # Se il max è circa 99 e dovrebbe essere circa 1, divide per 100.
    if 'IndiceCongestione' in df_traffic.columns and df_traffic['IndiceCongestione'].max() > 10: # Soglia ragionevole
        df_traffic['IndiceCongestione'] = df_traffic['IndiceCongestione'] / 100.0
        df_traffic['IndiceCongestione'] = df_traffic['IndiceCongestione'].round(2) # Arrotonda per pulizia
        df_traffic['IndiceCongestione'] = df_traffic['IndiceCongestione'].clip(lower=0.0, upper=1.0)

    # Combina Data e Ora in un'unica colonna datetime
    df_traffic['Timestamp'] = pd.to_datetime(
        df_traffic['DataRilevamento'] + ' ' + df_traffic['OraRilevamento'],
        dayfirst=True, errors='coerce'
    )

    # Rimuove righe con Timestamp NaN se la conversione fallisce
    df_traffic.dropna(subset=['Timestamp'], inplace=True)

    # Latitudine e Longitudine devono essere numerici prima di creare il punto
    # Aggiunto 'errors='coerce'' per gestire eventuali valori non numerici
    df_traffic['Latitudine'] = pd.to_numeric(df_traffic['Latitudine'], errors='coerce')
    df_traffic['Longitudine'] = pd.to_numeric(df_traffic['Longitudine'], errors='coerce')
    df_traffic.dropna(subset=['Latitudine', 'Longitudine'], inplace=True)

    df_traffic['geometry'] = df_traffic.apply(
        lambda row: Point(row['Longitudine'], row['Latitudine']),
        axis=1
    )

    # Rimuove righe con geometria NaN
    df_traffic.dropna(subset=['geometry'], inplace=True)

    # Converte in GeoDataFrame
    gdf_traffic = gpd.GeoDataFrame(df_traffic, geometry='geometry', crs="EPSG:4326")

    print("Dati traffico pre-elaborati con successo.")
    return gdf_traffic


def load_traffic_data(filepath):
    """Carica i dati di traffico da un file Excel."""
    if not os.path.exists(filepath):
        print(f"File dati traffico non trovato: {filepath}")
        return None
    try:
        df = pd.read_excel(filepath)
        print(f"Dati traffico caricati da {filepath}: {len(df)} righe.")
        if df.empty:
            print("Il file dei dati traffico è vuoto.")
            return None
        return df
    except Exception as e:
        print(f"Errore durante il caricamento del file dati traffico {filepath}: {e}")
        return None


def get_node_edge_features_from_osm(G, pois_gdf):
    """
    Estrae features aggiuntive dai dati OSM (nodi e edge).
    - Features sui nodi (incroci): grado, presenza di POI vicini.
    - Features sugli edge (segmenti stradali): lunghezza, tipo di strada, limiti di velocità, presenza di ciclabili/marciapiedi.
    """
    print("Estrazione features dal grafo OSM...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Funzione per pulire i valori di 'lanes'
    def clean_lanes(lanes_value):
        # Converte 'lanes_value' in un formato scalare gestibile
        if isinstance(lanes_value, (np.ndarray, pd.Series, list)):
            if len(lanes_value) > 0:
                # Prende il primo elemento. Potrebbe essere un altro array/list
                return clean_lanes(lanes_value[0])
            else:
                return 1 # Array/list vuota

        # Caso 1: Valore mancante (NaN)
        if pd.isna(lanes_value):
            return 1

        # Caso 2: È già un numero (int o float)
        if isinstance(lanes_value, (int, float)):
            return int(lanes_value)

        # Caso 3: È una stringa
        if isinstance(lanes_value, str):
            if ';' in lanes_value:
                try:
                    return int(lanes_value.split(';')[0].strip())
                except ValueError:
                    return 1
            try:
                return int(lanes_value.strip())
            except ValueError:
                return 1

        # Caso 4: È una lista Python (non np.ndarray o pd.Series)
        if isinstance(lanes_value, list):
            if len(lanes_value) > 0:
                first_element = lanes_value[0]
                try:
                    # Converte a stringa prima di int per massima robustezza
                    return int(str(first_element).strip())
                except (ValueError, TypeError):
                    pass # Fallimento, prosegue al default
            return 1 # Lista vuota

        # Caso 5: Qualsiasi altro tipo inaspettato non gestito sopra
        return 1

    # Applica la funzione di pulizia alla colonna 'lanes'
    edges_gdf['corsie'] = edges_gdf['lanes'].apply(clean_lanes)

    # Caratteristiche dei nodi (incroci)
    nodes_gdf['grado_incrocio'] = nodes_gdf.index.map(G.degree) # Numero di strade che si connettono all'incrocio

    # Identifica semafori e strisce pedonali esistenti dai POI
    ###if pois_gdf is not None and not pois_gdf.empty:
    ###    semafori_esistenti = pois_gdf[pois_gdf['highway'] == 'traffic_signals']
    ###    strisce_esistenti = pois_gdf[pois_gdf['highway'] == 'crossing']

    # Caratteristiche degli edge (segmenti stradali)
    edges_gdf['lunghezza_m'] = edges_gdf['length']
    # Gestisce il caso in cui 'highway' è una lista (es. ['primary', 'motorway'])
    edges_gdf['tipo_strada'] = edges_gdf['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
    # Imputa 1 corsia se non specificato (assumendo una corsia per direzione se non specificato)

    # One-Hot Encode per 'tipo_strada'
    encoder_ht = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_highway = encoder_ht.fit_transform(edges_gdf[['tipo_strada']])
    encoded_highway_df = pd.DataFrame(encoded_highway, columns=encoder_ht.get_feature_names_out(['tipo_strada']), index=edges_gdf.index)
    edges_gdf = pd.concat([edges_gdf, encoded_highway_df], axis=1)

    if 'sidewalk' in edges_gdf.columns:
        edges_gdf['ha_marciapiede'] = edges_gdf['sidewalk'].notna().astype(int)
    else:
        # Se la colonna 'sidewalk' non esiste, assumiamo che non ci siano marciapiedi
        edges_gdf['ha_marciapiede'] = 0

    if 'cycleway' in edges_gdf.columns:
        edges_gdf['ha_pista_ciclabile'] = edges_gdf['cycleway'].notna().astype(int)
    else:
        # Se la colonna 'cycleway' non esiste, assumiamo che non ci siano piste ciclabili
        edges_gdf['ha_pista_ciclabile'] = 0

    # Imputa limite di velocità medio (es. 50 km/h in area urbana) se mancante
    def parse_maxspeed(speed_val):
        if isinstance(speed_val, list):
            speed_val = speed_val[0] # Prende il primo se è una lista
        try:
            return float(str(speed_val).replace(' km/h', '').split(' ')[0])
        except (ValueError, TypeError):
            return np.nan # Ritorna NaN se non può essere convertito

    edges_gdf['limite_velocita'] = edges_gdf['maxspeed'].apply(parse_maxspeed)
    edges_gdf['limite_velocita'] = edges_gdf['limite_velocita'].fillna(50) # Imputa 50 se NaN dopo la conversione

    # Filtra e seleziona colonne pertinenti per il modello
    nodes_features = nodes_gdf[['grado_incrocio', 'geometry']].copy()
    edges_features = edges_gdf[['lunghezza_m', 'corsie', 'ha_marciapiede', 'ha_pista_ciclabile', 'limite_velocita'] + encoded_highway_df.columns.tolist() + ['geometry']].copy()

    print("Features da OSM estratte con successo.")
    return nodes_features, edges_features
