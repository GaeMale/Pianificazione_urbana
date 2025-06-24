import pandas as pd
import geopandas as gpd
import osmnx as ox


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
            df_accidents = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df_accidents = pd.read_csv(filepath)
        else:
            raise ValueError("Formato del file incidenti non supportato. Usare .xlsx o .csv.")

        return df_accidents
    except FileNotFoundError:
        return None


def load_traffic_data(filepath="data/dati_traffico.csv"):
    """Carica i dati di traffico."""
    try:
        if filepath.endswith('.csv'):
            # Per CSV, uso 'decimal' per il separatore decimale e 'thousands' per il separatore delle migliaia
            df_traffic = pd.read_csv(filepath, decimal=',', thousands='.')
        elif filepath.endswith('.json'):
            df_traffic = pd.read_json(filepath)
        elif filepath.endswith('.xlsx'):
            df_traffic = pd.read_excel(filepath)
            # Applica la conversione per i formati numerici
            for col in ['ConteggioVeicoli', 'VelocitaMedia', 'IndiceCongestione']:
                if col in df_traffic.columns:
                    # Sostituisce il punto delle migliaia con niente e la virgola decimale con un punto
                    # Poi converte a numerico. errors='coerce' trasforma errori in NaN.
                    df_traffic[col] = df_traffic[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                    df_traffic[col] = pd.to_numeric(df_traffic[col], errors='coerce')
                    if col == 'ConteggioVeicoli':
                        df_traffic[col] = df_traffic[col].astype('Int64') # Usa Int64 per supportare NaN
        else:
            raise ValueError("Formato del file traffico non supportato. Usare .csv, .json o .xlsx.")
        return df_traffic
    except FileNotFoundError:
        return None