# src/preprocessing.py
import os

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import osmnx as ox
from sklearn.impute import SimpleImputer

def load_accidents(filepath):
    """Carica i dati degli incidenti da un file Excel."""
    if not os.path.exists(filepath):
        print(f"File dati incidenti non trovato: {filepath}")
        return None
    try:
        df = pd.read_excel(filepath)
        print(f"Dati incidenti caricati da {filepath}: {len(df)} righe.")
        if df.empty:
            print("Il file dei dati incidenti è vuoto.")
            return None
        return df
    except Exception as e:
        print(f"Errore durante il caricamento del file dati incidenti {filepath}: {e}")
        return None

# def preprocess_accidents(df_accidents):
#     """
#     Pre-processa il DataFrame degli incidenti stradali:
#     - Rimuove righe con gravità non valida.
#     - Gestisce valori mancanti (NaN) per la gravità.
#     - Crea la colonna 'geometry' per la georeferenziazione.
#     """
#     print("Pre-elaborazione dei dati incidenti stradali...")
#     # --- Adatta questi nomi di colonna ai tuoi dati ISTAT ---
#     # Esempio: combinare colonne 'Data' e 'Ora' in un unico timestamp
#     # Se il tuo file ha una colonna unica 'Timestamp_Incidente', usala direttamente
#     if 'Data' in df_accidents.columns and 'Ora' in df_accidents.columns:
#         df_accidents['timestamp'] = pd.to_datetime(df_accidents['Data'] + ' ' + df_accidents['Ora'], errors='coerce', format='%d/%m/%Y %H:%M')
#         df_accidents.drop(columns=['Data', 'Ora'], inplace=True)
#     elif 'Data_Ora_Incidente' in df_accidents.columns: # Esempio di colonna unica
#         df_accidents['timestamp'] = pd.to_datetime(df_accidents['Data_Ora_Incidente'], errors='coerce')
#     else:
#         print("Attenzione: Colonna timestamp non trovata o riconosciuta nei dati incidenti. Aggiustare i nomi delle colonne o la logica.")
#
#     # Rimuovi righe con timestamp mancanti dopo la conversione
#     df_accidents.dropna(subset=['timestamp'], inplace=True)
#
#     # Estrai features temporali
#     df_accidents['ora_del_giorno'] = df_accidents['timestamp'].dt.hour
#     df_accidents['giorno_della_settimana'] = df_accidents['timestamp'].dt.dayofweek # Lunedì=0, Domenica=6
#     df_accidents['is_weekend'] = df_accidents['giorno_della_settimana'].isin([5, 6]).astype(int)
#
#     # Gestione dati mancanti per colonne numeriche rilevanti (es. 'Gravità' se è un numero)
#     # df_accidents['Gravita'].fillna(df_accidents['Gravita'].mode()[0], inplace=True) # Esempio imputazione con la moda
#
#     # --- Adatta questi nomi di colonna Latitudine/Longitudine ---
#     # Crea GeoDataFrame se hai Latitudine e Longitudine
#     if 'Latitudine' in df_accidents.columns and 'Longitudine' in df_accidents.columns:
#         df_accidents = df_accidents.dropna(subset=['Latitudine', 'Longitudine'])
#         # Assicurati che latitudine e longitudine siano numeri
#         df_accidents['Latitudine'] = pd.to_numeric(df_accidents['Latitudine'], errors='coerce')
#         df_accidents['Longitudine'] = pd.to_numeric(df_accidents['Longitudine'], errors='coerce')
#         df_accidents.dropna(subset=['Latitudine', 'Longitudine'], inplace=True)
#
#         geometry = [Point(xy) for xy in zip(df_accidents['Longitudine'], df_accidents['Latitudine'])]
#         gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry=geometry, crs="EPSG:4326") # WGS84
#         print("Dati incidenti convertiti in GeoDataFrame.")
#     else:
#         print("Attenzione: Colonne Lat/Lon non trovate nei dati incidenti. Impossibile creare GeoDataFrame.")
#         gdf_accidents = df_accidents # Restituisce DataFrame normale se non georeferenziabile
#
#     return gdf_accidents # O df_accidents se non GeoDataFrame

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

    # Assicurati che le colonne Latitudine e Longitudine siano numeriche
    df_accidents['Latitudine'] = pd.to_numeric(df_accidents['Latitudine'], errors='coerce')
    df_accidents['Longitudine'] = pd.to_numeric(df_accidents['Longitudine'], errors='coerce')

    # Rimuovi righe dove Latitudine o Longitudine non sono valide
    df_accidents.dropna(subset=['Latitudine', 'Longitudine'], inplace=True)

    # Rimuovi righe dove 'Gravita' non è un numero valido
    df_accidents = df_accidents[pd.to_numeric(df_accidents['Gravita'], errors='coerce').notna()]
    df_accidents['Gravita'] = df_accidents['Gravita'].astype(int)

    # Gestisci i valori mancanti nella colonna 'Gravita' (es. con la media)
    if df_accidents['Gravita'].isnull().any():
        # Correzione del FutureWarning
        df_accidents['Gravita'] = df_accidents['Gravita'].fillna(df_accidents['Gravita'].mean())
        print("Valori NaN nella colonna 'Gravita' gestiti con la media.")

    # Crea la colonna 'geometry' da Latitudine e Longitudine
    df_accidents['geometry'] = df_accidents.apply(
        lambda row: Point(row['Longitudine'], row['Latitudine']),
        axis=1
    )

    # Rimuovi righe con geometria NaN (potrebbero esserci se Lat/Lon erano NaN prima della dropna)
    df_accidents.dropna(subset=['geometry'], inplace=True)

    # Converti in GeoDataFrame
    gdf_accidents = gpd.GeoDataFrame(df_accidents, geometry='geometry', crs="EPSG:4326")

    print("Dati incidenti pre-elaborati con successo.")
    return gdf_accidents

#def preprocess_traffic(df_traffic):
#    """
#    Pulizia e feature engineering per i dati di traffico.
#    - Converti timestamp
#    - Gestione mancanti (es. 'vehicle_count', 'avg_speed')
#    - Estrazione features temporali
#    - Calcolo indici di congestione
#    """
#    print("Pre-elaborazione dati traffico...")
#    # --- Adatta questi nomi di colonna al tuo dataset di traffico ---
#    if 'timestamp' in df_traffic.columns:
#        df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'], errors='coerce')
#    else:
#        print("Attenzione: Colonna timestamp non trovata o riconosciuta nei dati traffico. Aggiustare i nomi delle colonne o la logica.")
#        return None # Oppure gestisci diversamente
#
#    df_traffic.dropna(subset=['timestamp'], inplace=True)
#
#    # Estrai features temporali
#    df_traffic['ora_del_giorno'] = df_traffic['timestamp'].dt.hour
#    df_traffic['giorno_della_settimana'] = df_traffic['timestamp'].dt.dayofweek
#    df_traffic['is_weekend'] = df_traffic['giorno_della_settimana'].isin([5, 6]).astype(int)
#    df_traffic['is_ora_di_punta'] = ((df_traffic['ora_del_giorno'] >= 7) & (df_traffic['ora_del_giorno'] <= 9)) | \
#                                    ((df_traffic['ora_del_giorno'] >= 17) & (df_traffic['ora_del_giorno'] <= 19)).astype(int)
#
#    # Gestione mancanti per vehicle_count e avg_speed
#    # Assicurati che i nomi delle colonne siano corretti per il tuo dataset
#    if 'conteggio_veicoli' in df_traffic.columns:
#        imputer_vc = SimpleImputer(strategy='mean')
#        df_traffic['conteggio_veicoli'] = imputer_vc.fit_transform(df_traffic[['conteggio_veicoli']])
#    if 'velocita_media' in df_traffic.columns:
#        imputer_as = SimpleImputer(strategy='mean')
#        df_traffic['velocita_media'] = imputer_as.fit_transform(df_traffic[['velocita_media']])
#
#    # Calcolo indici di congestione (se hai entrambe le colonne)
#    if 'conteggio_veicoli' in df_traffic.columns and 'velocita_media' in df_traffic.columns:
#        # Evita divisione per zero
#        df_traffic['indice_congestione'] = df_traffic['conteggio_veicoli'] / (df_traffic['velocita_media'] + 1e-6)
#    else:
#        print("Attenzione: 'conteggio_veicoli' o 'velocita_media' non trovati. Impossibile calcolare l'indice di congestione.")
#
#    return df_traffic

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

    # Conversione a numerico per sicurezza (utile se load_traffic_data non l'ha fatto perfettamente o per altri file)
    for col in ['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']:
        if col in df_traffic.columns:
            df_traffic[col] = pd.to_numeric(df_traffic[col], errors='coerce')
            # Per ConteggioVeicoli, assicurati che sia un intero (potrebbe avere NaN se errors='coerce' li ha creati)
            if col == 'ConteggioVeicoli':
                df_traffic[col] = df_traffic[col].astype('Int64') # 'Int64' supporta NaN

    # --- INIZIO CORREZIONE SCALA QUI ---
    # Correggi la scala di VelocitaMedia
    # Se il max è circa 6000 e dovrebbe essere circa 60, dividi per 100.
    # Applica questa correzione solo se la colonna esiste e sembra "sballata"
    if 'VelocitaMedia' in df_traffic.columns and df_traffic['VelocitaMedia'].max() > 100: # Soglia ragionevole per identificare il problema
        #print("Correggo la scala di 'VelocitaMedia' (divido per 100).")
        df_traffic['VelocitaMedia'] = df_traffic['VelocitaMedia'] / 100.0
        df_traffic['VelocitaMedia'] = df_traffic['VelocitaMedia'].round(2) # Arrotonda per pulizia

    # Correggi la scala di IndiceCongestione
    # Se il max è circa 99 e dovrebbe essere circa 1, dividi per 100.
    if 'IndiceCongestione' in df_traffic.columns and df_traffic['IndiceCongestione'].max() > 10: # Soglia ragionevole
        #print("Correggo la scala di 'IndiceCongestione' (divido per 100).")
        df_traffic['IndiceCongestione'] = df_traffic['IndiceCongestione'] / 100.0
        df_traffic['IndiceCongestione'] = df_traffic['IndiceCongestione'].round(2) # Arrotonda per pulizia
        # Assicurati che l'indice rimanga tra 0 e 1
        df_traffic['IndiceCongestione'] = df_traffic['IndiceCongestione'].clip(lower=0.0, upper=1.0)
    # --- FINE CORREZIONE SCALA ---

    # Combina Data e Ora in un'unica colonna datetime
    df_traffic['Timestamp'] = pd.to_datetime(
        df_traffic['DataRilevamento'] + ' ' + df_traffic['OraRilevamento'],
        dayfirst=True, errors='coerce'
    )

    # Rimuovi righe con Timestamp NaN se la conversione fallisce
    df_traffic.dropna(subset=['Timestamp'], inplace=True)

    # Assicurati che Latitudine e Longitudine siano numerici prima di creare il punto
    # Anche se il tuo df_traffic.dtypes le mostrava già come float64, è una buona pratica
    # Aggiungi 'errors='coerce'' per gestire eventuali valori non numerici che potrebbero esserci sfuggiti
    df_traffic['Latitudine'] = pd.to_numeric(df_traffic['Latitudine'], errors='coerce')
    df_traffic['Longitudine'] = pd.to_numeric(df_traffic['Longitudine'], errors='coerce')
    df_traffic.dropna(subset=['Latitudine', 'Longitudine'], inplace=True)

    # Crea la colonna 'geometry' da Latitudine e Longitudine
    #df_traffic['geometry'] = df_traffic.apply(
    #    lambda row: Point(row['Longitudine'], row['Latitudine']) if pd.notnull(row['Longitudine']) and pd.notnull(row['Latitudine']) else None,
    #    axis=1
    #)

    df_traffic['geometry'] = df_traffic.apply(
        lambda row: Point(row['Longitudine'], row['Latitudine']), # Qui non serve controllare isnull se hai fatto dropna sopra
        axis=1
    )

    # Rimuovi righe con geometria NaN
    df_traffic.dropna(subset=['geometry'], inplace=True)

    # Converti in GeoDataFrame
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

#def preprocess_demographics(df_demographics):
#    """
#    Pulizia e feature engineering per i dati demografici.
#    - Filtra per Bari
#    - Rinomina colonne (se necessario)
#    - Normalizzazione o creazione di densità
#    """
#    print("Pre-elaborazione dati demografici...")
#    # --- Adatta il filtraggio e i nomi delle colonne al tuo file ISTAT ---
#    # Esempio: filtra per il codice comune di Bari. Il codice ISTAT di Bari è 072006.
#    # Trova la colonna giusta per il codice comune nel tuo file ISTAT (es. 'COD_COMUNE', 'Codice_Comune_ISTAT')
#    if 'Codice_Comune' in df_demographics.columns:
#        df_demographics_bari = df_demographics[df_demographics['Codice_Comune'].astype(str).str.contains('072006', na=False)].copy()
#    elif 'Nome_Comune' in df_demographics.columns: # Se hai il nome del comune
#        df_demographics_bari = df_demographics[df_demographics['Nome_Comune'].str.contains('Bari', case=False, na=False)].copy()
#    else:
#        print("Attenzione: Colonna comune non identificata. Assicurarsi di filtrare manualmente i dati per Bari o aggiustare la logica.")
#        df_demographics_bari = df_demographics.copy() # Continua con tutti i dati se non puoi filtrare
#
#    # Esempio di selezione e rinominazione di colonne rilevanti (adatta ai tuoi dati)
#    # Assumi che 'Popolazione_Totale' sia la colonna con la popolazione
#    if 'Popolazione_Totale' in df_demographics_bari.columns:
#        df_demographics_bari.rename(columns={'Popolazione_Totale': 'popolazione_totale'}, inplace=True)
#    else:
#        print("Attenzione: Colonna 'Popolazione_Totale' non trovata. Rinominare la colonna della popolazione se ha un nome diverso.")
#
#    # Se hai dati a livello di sezione di censimento, puoi calcolare densità
#    # e poi mapparle spazialmente ai quartieri o a griglie.
#
#    return df_demographics_bari

def get_node_edge_features_from_osm(G, pois_gdf):
    """
    Estrae features aggiuntive dai dati OSM (nodi e edge).
    - Features sui nodi (incroci): grado, presenza di POI vicini.
    - Features sugli edge (segmenti stradali): lunghezza, tipo di strada, limiti di velocità, presenza di ciclabili/marciapiedi.
    """
    print("Estrazione features dal grafo OSM...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # --- INIZIO MODIFICA PER GESTIRE 'lanes' ---

    # Funzione per pulire i valori di 'lanes'
    def clean_lanes(lanes_value):
        # Debug print - lascialo attivo per vedere cosa arriva
        # print(f"Processing value: {lanes_value}, Type: {type(lanes_value)}")

        # --- INIZIO NUOVA GESTIONE ROBUSTA DEL TIPO DI INGRESSO ---
        # Converti 'lanes_value' in un formato scalare gestibile, se è un array/list-like
        if isinstance(lanes_value, (np.ndarray, pd.Series, list)):
            if len(lanes_value) > 0:
                # Prendi il primo elemento. Potrebbe essere un altro array/list, quindi ricursivo.
                return clean_lanes(lanes_value[0])
            else:
                return 1 # Array/list vuota
        # --- FINE NUOVA GESTIONE ROBUSTA ---

        # Caso 1: Valore mancante (NaN) - Ora dovrebbe essere sicuro, poiché abbiamo gestito array multi-elemento
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
                    pass # Fallimento, prosegui al default
            return 1 # Lista vuota

        # Caso 5: Qualsiasi altro tipo inaspettato non gestito sopra
        return 1

    # Applica la funzione di pulizia alla colonna 'lanes'
    edges_gdf['corsie'] = edges_gdf['lanes'].apply(clean_lanes)

    # --- FINE MODIFICA PER GESTIRE 'lanes' ---

    # --- Caratteristiche dei nodi (incroci) ---
    nodes_gdf['grado_incrocio'] = nodes_gdf.index.map(G.degree) # Numero di strade che si connettono all'incrocio

    # Identifica semafori e strisce pedonali esistenti dai POI
    if pois_gdf is not None and not pois_gdf.empty:
        semafori_esistenti = pois_gdf[pois_gdf['highway'] == 'traffic_signals']
        strisce_esistenti = pois_gdf[pois_gdf['highway'] == 'crossing']

        # Assegna il semaforo/strisce al nodo OSM più vicino
        # Questo è un processo che potrebbe essere più accurato in data_integration.py
        # Per ora, facciamo un'assegnazione semplice basata sulla vicinanza.
        # nodes_gdf['has_existing_traffic_signal'] = ...
        # nodes_gdf['has_existing_crossing'] = ...
        # Questo verrà gestito meglio nella fase di integrazione per mappare i POI ai nodi.

    # --- Caratteristiche degli edge (segmenti stradali) ---
    edges_gdf['lunghezza_m'] = edges_gdf['length']
    # Gestisce il caso in cui 'highway' è una lista (es. ['primary', 'motorway'])
    edges_gdf['tipo_strada'] = edges_gdf['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
    # Imputa 1 corsia se non specificato (assumendo una corsia per direzione se non specificato)
    #edges_gdf['corsie'] = edges_gdf['lanes'].fillna(1).astype(int)

    # One-Hot Encode per 'tipo_strada'
    encoder_ht = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_highway = encoder_ht.fit_transform(edges_gdf[['tipo_strada']])
    encoded_highway_df = pd.DataFrame(encoded_highway, columns=encoder_ht.get_feature_names_out(['tipo_strada']), index=edges_gdf.index)
    edges_gdf = pd.concat([edges_gdf, encoded_highway_df], axis=1)

    # Esistenza di marciapiede/pista ciclabile
    #edges_gdf['ha_marciapiede'] = edges_gdf['sidewalk'].notna().astype(int)
    #edges_gdf['ha_pista_ciclabile'] = edges_gdf['cycleway'].notna().astype(int)

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
    # Assicurati che 'maxspeed' sia gestito per tipi diversi (stringa con 'km/h', lista, ecc.)
    def parse_maxspeed(speed_val):
        if isinstance(speed_val, list):
            speed_val = speed_val[0] # Prendi il primo se è una lista
        try:
            return float(str(speed_val).replace(' km/h', '').split(' ')[0])
        except (ValueError, TypeError):
            return np.nan # Ritorna NaN se non può essere convertito
    #edges_gdf['limite_velocita'] = pd.to_numeric(edges_gdf['maxspeed'], errors='coerce').fillna(50)
    edges_gdf['limite_velocita'] = edges_gdf['maxspeed'].apply(parse_maxspeed)
    edges_gdf['limite_velocita'] = edges_gdf['limite_velocita'].fillna(50) # Imputa 50 se NaN dopo la conversione

    # Filtra e seleziona colonne pertinenti per i modelli
    nodes_features = nodes_gdf[['grado_incrocio', 'geometry']].copy() # Aggiungeremo POI counts dopo
    edges_features = edges_gdf[['lunghezza_m', 'corsie', 'ha_marciapiede', 'ha_pista_ciclabile', 'limite_velocita'] + encoded_highway_df.columns.tolist() + ['geometry']].copy()

    print("Features da OSM estratte con successo.")
    return nodes_features, edges_features
