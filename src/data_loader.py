# src/data_loader.py
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx

def load_road_network(filepath="data/rete_stradale.graphml"):
    """Carica la rete stradale da un file GraphML."""
    try:
        G = ox.load_graphml(filepath)
        print(f"Caricata la rete stradale da {filepath}")
        return G
    except FileNotFoundError:
        print(f"Errore: File della rete stradale non trovato in {filepath}. Eseguire lo script di acquisizione OSMnx.")
        return None

def load_pois(filepath="data/punti_interesse.geojson"):
    """Carica i Punti di Interesse (POI) da un file GeoJSON."""
    try:
        pois_gdf = gpd.read_file(filepath)
        print(f"Caricati i POI da {filepath}")
        return pois_gdf
    except FileNotFoundError:
        print(f"Errore: File POI non trovato in {filepath}. Eseguire lo script di acquisizione OSMnx.")
        return None

def load_accidents(filepath="data/incidenti_stradali.xlsx"):
    """Carica i dati di incidenti stradali"""
    try:
        if filepath.endswith('.xlsx'):
            # Potrebbe essere necessario specificare il nome del foglio (sheet_name) se ce ne sono più
            df_accidents = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df_accidents = pd.read_csv(filepath)
        else:
            raise ValueError("Formato del file incidenti non supportato. Usare .xlsx o .csv.")

        #print(f"Caricati i dati degli incidenti da {filepath}")
        return df_accidents
    except FileNotFoundError:
        #print(f"Errore: File dati incidenti non trovato in {filepath}.")
        #print("Assicurarsi che il file sia nella cartella 'data'.")
        return None

def load_traffic_data(filepath="data/dati_traffico.csv"):
    """Carica i dati di traffico."""
    try:
        if filepath.endswith('.csv'):
            # Per CSV, usa 'decimal' per il separatore decimale e 'thousands' per il separatore delle migliaia
            df_traffic = pd.read_csv(filepath, decimal=',', thousands='.')
            #df_traffic = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df_traffic = pd.read_json(filepath)
        elif filepath.endswith('.xlsx'):
            df_traffic = pd.read_excel(filepath)
            # Applica la conversione per i formati numerici italiani
            for col in ['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']:
                if col in df_traffic.columns:
                    # Sostituisci il punto delle migliaia con niente e la virgola decimale con un punto
                    # Poi converti a numerico. errors='coerce' trasforma errori in NaN.
                    df_traffic[col] = df_traffic[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    df_traffic[col] = pd.to_numeric(df_traffic[col], errors='coerce')
                    # Potresti voler ripristinare il tipo int64 per ConteggioVeicoli dopo la conversione se appropriato
                    if col == 'ConteggioVeicoli':
                        df_traffic[col] = df_traffic[col].astype('Int64') # Usa Int64 per supportare NaN
        else:
            raise ValueError("Formato del file traffico non supportato. Usare .csv, .json o .xlsx.")
        #print(f"Caricati i dati di traffico da {filepath}")
        return df_traffic
    except FileNotFoundError:
        #print(f"Errore: File dati traffico non trovato in {filepath}.")
        #print("Assicurarsi che il file sia nella cartella 'data'.")
        return None

#def load_demographics(filepath="data/dati_demografici.xlsx"):
#    """Carica i dati demografici (da ISTAT)."""
#    try:
#        if filepath.endswith('.xlsx'):
#            # Potrebbe essere necessario specificare il nome del foglio (sheet_name)
#            df_demographics = pd.read_excel(filepath)
#        elif filepath.endswith('.csv'):
#            df_demographics = pd.read_csv(filepath)
#        else:
#            raise ValueError("Formato del file demografico non supportato. Usare .xlsx o .csv.")
#        print(f"Caricati i dati demografici da {filepath}")
#        return df_demographics
#    except FileNotFoundError:
#        print(f"Errore: File dati demografici non trovato in {filepath}.")
#        print("Assicurarsi che il file sia nella cartella 'data'.")
#        return None