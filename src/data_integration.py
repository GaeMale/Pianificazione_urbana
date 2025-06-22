import os

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import Point
from scipy.spatial import cKDTree

# --- Definizioni di Soglie (possono essere spostate in config.py se preferisci) ---
# Queste soglie sono usate per definire un incrocio ad "alto rischio" nel target dei modelli.
RISK_THRESHOLD_NUM_ACCIDENTS = 3
RISK_THRESHOLD_AVG_SEVERITY = 2.5 # Su una scala di gravità, es. 1-4 o 1-5

# Soglie per la prossimità dei POI
POI_PROXIMITY_RADIUS_METERS = 300 # Raggio di ricerca per POI vicini a un nodo OSM

# --- Funzioni di Integrazione Dati ---

def assign_accidents_to_osm_elements(gdf_accidents_processed, G):
    """
    Assegna gli incidenti al nodo OSM (incrocio) più vicino.
    Aggrega il numero di incidenti e la gravità media per nodo.
    """
    print("Assegnazione incidenti ai nodi OSM più vicini...")

    if gdf_accidents_processed is None or gdf_accidents_processed.empty:
        print("GeoDataFrame incidenti elaborato è vuoto o None. Impossibile assegnare incidenti.")
        return pd.DataFrame(columns=['osmid', 'num_incidenti', 'gravita_media_incidente'])

    # Estrai i nodi del grafo come GeoDataFrame
    # nodes_gdf, _ = ox.graph_to_gdfs(G, edges=False) # Vecchio modo se tornasse 2 valori
    nodes_gdf = ox.graph_to_gdfs(G, edges=False) # Corretto per aspettarsi 1 valore
    #nodes_gdf = nodes_gdf[['geometry', 'osmid']] # Manteniamo solo geometria e osmid
    if 'osmid' not in nodes_gdf.columns:
        nodes_gdf['osmid'] = nodes_gdf.index

    if nodes_gdf.empty:
        print("Il grafo OSM non contiene nodi. Impossibile assegnare incidenti.")
        return pd.DataFrame(columns=['osmid', 'num_incidenti', 'gravita_media_incidente'])

    # Converti le coordinate dei nodi in un array NumPy per cKDTree
    nodes_coords = np.array([(p.x, p.y) for p in nodes_gdf.geometry])
    # Crea un cKDTree per una ricerca efficiente dei vicini più prossimi
    tree = cKDTree(nodes_coords)

    # Converti le coordinate degli incidenti in un array NumPy
    accidents_coords = np.array([(p.x, p.y) for p in gdf_accidents_processed.geometry])

    # Trova l'indice del nodo più vicino per ogni incidente
    distances, node_indices = tree.query(accidents_coords)

    # Assegna l'OSMID del nodo più vicino a ciascun incidente
    #gdf_accidents_processed['nearest_node_osmid'] = nodes_gdf.iloc[node_indices]['osmid'].values
    gdf_accidents_processed['nearest_node_osmid'] = nodes_gdf.index[node_indices].values

    # Aggrega gli incidenti per nodo OSM
    accidents_per_node = gdf_accidents_processed.groupby('nearest_node_osmid').agg(
        num_incidenti=('Gravita', 'count'), # Conta quanti incidenti ci sono stati
        gravita_media_incidente=('Gravita', 'mean') # Calcola la gravità media
    ).reset_index()

    accidents_per_node.rename(columns={'nearest_node_osmid': 'osmid'}, inplace=True)
    print("Incidenti assegnati ai nodi OSM e aggregati.")
    return accidents_per_node

def assign_spatial_features_to_nodes(nodes_features, pois_gdf, G):
    """
    Assegna features spaziali (es. POI vicini, densità popolazione) ai nodi OSM.
    Include un fallback per la densità di popolazione se i dati demografici granulari non sono disponibili.

    Args:
        nodes_features (pd.DataFrame): DataFrame delle features base dei nodi OSM (deve avere 'osmid' come indice).
        pois_gdf (gpd.GeoDataFrame): GeoDataFrame dei POI pre-elaborati, con colonna 'geometry'. Può essere None o vuoto.
        df_demographics_processed (pd.DataFrame o None): DataFrame dei dati demografici pre-elaborati,
                                                           o None se non disponibili/caricati.
        G (networkx.MultiDiGraph): Grafo della rete stradale OSM (usato per i nodi e le loro geometrie).
        city_avg_pop_density (float): Densità media di popolazione da usare come fallback se
                                      df_demographics_processed è None o vuoto.

    Returns:
        pd.DataFrame: DataFrame nodes_features aggiornato con le nuove features spaziali.
    """
    print("Assegnazione features spaziali ai nodi OSM...")

    # Assicurati che nodes_features abbia 'osmid' come indice
    # Se 'osmid' è una colonna, impostala come indice. Se è già l'indice, non fa nulla.
    if nodes_features.index.name != 'osmid':
        if 'osmid' in nodes_features.columns:
            nodes_features = nodes_features.set_index('osmid').copy()
        else:
            # Fallback se 'osmid' non è né indice né colonna (dovrebbe essere l'indice da get_node_edge_features_from_osm)
            # Questo caso è improbabile se la pipeline è corretta, ma è un'ulteriore safety check.
            print("ATTENZIONE: 'osmid' non trovato come indice o colonna in nodes_features. Assumendo l'indice attuale come osmid.")
            #nodes_features['osmid_temp'] = nodes_features.index # Salva l'indice corrente se necessario
            nodes_features.index.name = 'osmid' # Imposta il nome dell'indice per consistenza

    # Ottieni le geometrie dei nodi dal grafo G per le operazioni spaziali
    # Questo è necessario per le sjoin, dato che nodes_features potrebbe non essere un GeoDataFrame
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)[['geometry']]
    # Assicurati che l'indice sia 'osmid'
    nodes_gdf.index.name = 'osmid'

    # Per l'Italia, un buon CRS proiettato è EPSG:25832 (UTM Zone 32N).
    projected_crs_epsg = 25832

    if nodes_gdf.crs is None or nodes_gdf.crs.is_geographic:
        print(f"AVVISO: nodes_gdf è in un CRS geografico o senza CRS. Riproietto a EPSG:{projected_crs_epsg} per buffer accurati.")
        nodes_gdf = nodes_gdf.to_crs(epsg=projected_crs_epsg)
    elif nodes_gdf.crs.to_epsg() != projected_crs_epsg:
        print(f"AVVISO: nodes_gdf è in un CRS proiettato diverso ({nodes_gdf.crs.to_epsg()}). Riproietto a EPSG:{projected_crs_epsg} per consistenza.")
        nodes_gdf = nodes_gdf.to_crs(epsg=projected_crs_epsg)

    # --- Gestione POI (Punti di Interesse) ---
    print("Assegnazione POI (Punti di Interesse)...")
    if pois_gdf is not None and not pois_gdf.empty:
        # Assicurati che pois_gdf abbia un CRS e che sia lo stesso dei nodi
        if pois_gdf.crs is None and nodes_gdf.crs is not None:
            pois_gdf = pois_gdf.set_crs(nodes_gdf.crs)
        elif pois_gdf.crs != nodes_gdf.crs:
            pois_gdf = pois_gdf.to_crs(nodes_gdf.crs)

        # Creazione di buffer attorno ai nodi per trovare POI vicini (es. raggio di 500 metri)
        # Ora il buffer è corretto perché nodes_gdf è in un CRS proiettato.
        nodes_buffer = nodes_gdf.buffer(500)
        nodes_buffer_gdf = gpd.GeoDataFrame(geometry=nodes_buffer, index=nodes_buffer.index, crs=nodes_gdf.crs)
        nodes_buffer_gdf.index.name = 'osmid' # Assicura che l'indice sia osmid

        # Sjoin per contare i POI vicini
        pois_within_buffer = gpd.sjoin(nodes_buffer_gdf, pois_gdf, how="left", predicate="intersects")

        # Debug: Stampa le colonne di pois_within_buffer per vedere cosa c'è
        print("Colonne disponibili in pois_within_buffer:", pois_within_buffer.columns.tolist())

        # Conta i diversi tipi di POI
        # Se 'amenity' non è presente, questo potrebbe causare un errore. Assumiamo che ci sia.
        if 'amenity' in pois_within_buffer.columns:
            nodes_features['num_scuole_vicine'] = pois_within_buffer[
                pois_within_buffer['amenity'] == 'school'
                ].groupby('osmid').size().reindex(nodes_features.index, fill_value=0)

            nodes_features['num_ospedali_vicini'] = pois_within_buffer[
                pois_within_buffer['amenity'] == 'hospital'
                ].groupby('osmid').size().reindex(nodes_features.index, fill_value=0)

            #nodes_features['num_stazioni_bus_vicine'] = pois_within_buffer[
            #    pois_within_buffer['public_transport'] == 'bus_stop'
            #    ].groupby('osmid').size().reindex(nodes_features.index, fill_value=0)

            print("Conteggio POI vicini assegnato.")
        else:
            print("ATTENZIONE: Colonna 'amenity' non trovata in pois_gdf. Conteggi POI non assegnati.")
            nodes_features['num_scuole_vicine'] = 0.0
            nodes_features['num_ospedali_vicini'] = 0.0
            nodes_features['num_stazioni_bus_vicine'] = 0.0 # Assicura che la colonna esista e sia a 0

        # Correggi i FutureWarning: Assegna il risultato indietro alla colonna
        nodes_features['num_scuole_vicine'] = nodes_features['num_scuole_vicine'].fillna(0)
        nodes_features['num_ospedali_vicini'] = nodes_features['num_ospedali_vicini'].fillna(0)
        #nodes_features['num_stazioni_bus_vicine'] = nodes_features['num_stazioni_bus_vicine'].fillna(0)
    else:
        print("GeoDataFrame POI non disponibile o vuoto. Saltando l'assegnazione dei POI.")
        nodes_features['num_scuole_vicine'] = 0.0
        nodes_features['num_ospedali_vicini'] = 0.0
        nodes_features['num_stazioni_bus_vicine'] = 0.0

    # --- Gestione Semaphores (Semafori) e Crosswalks (Strisce pedonali) ---
    print("Assegnazione features semafori/strisce pedonali...")

    # Questo è l'approccio più robusto se G.graph['place'] o G.graph['bbox'] non sono presenti.

    # 1. Ottieni i nodi del grafo come GeoDataFrame
    nodes_geo_df_for_bbox_calculation = ox.graph_to_gdfs(G, edges=False)

    # Gestione di un grafo vuoto o senza geometrie valide per il bounding box
    if nodes_geo_df_for_bbox_calculation.empty or nodes_geo_df_for_bbox_calculation['geometry'].isnull().all():
        print("ATTENZIONE: Il grafo G non contiene nodi con geometria valida. Impossibile determinare l'area di query per semafori/strisce pedonali.")
        # Assicurati che le colonne siano comunque create se si esce qui
        nodes_features['has_traffic_signals'] = 0
        nodes_features['has_crosswalk'] = 0
        return nodes_features # Esci dalla funzione se non ci sono nodi validi da cui calcolare il bbox

    # 2. Calcola il bounding box combinato di tutte le geometrie dei nodi
    # bounds restituisce (minx, miny, maxx, maxy) che corrispondono a (ovest, sud, est, nord)
    minx, miny, maxx, maxy = nodes_geo_df_for_bbox_calculation.unary_union.bounds

    # ox.features_from_bbox si aspetta una tupla (north, south, east, west)
    query_area_bbox = (maxy, miny, maxx, minx) # Riassegna per l'ordine corretto

    # Recupera i nodi con tag 'highway'='traffic_signals' o 'crossing'
    try:
        #traffic_signals_gdf = ox.features_from_place(
        # Usa ox.features_from_bbox con il bounding box calcolato
        traffic_signals_gdf = ox.features_from_bbox(
            bbox=query_area_bbox, # Passa il bounding box
            tags={'highway': 'traffic_signals'}
        )
        traffic_signals_nodes = traffic_signals_gdf.index if not traffic_signals_gdf.empty else pd.Index([])
    except Exception as e:
        print(f"ATTENZIONE: Errore nel recupero dei semafori: {e}. Presumo nessun semaforo.")
        traffic_signals_nodes = pd.Index([])

    try:
        # Usa ox.features_from_bbox anche qui per gli attraversamenti pedonali
        crosswalk_gdf = ox.features_from_bbox(
            bbox=query_area_bbox, # Passa il bounding box
            tags={'highway': 'crossing'}
        )
        crosswalk_nodes = crosswalk_gdf.index if not crosswalk_gdf.empty else pd.Index([])
    except Exception as e:
        print(f"ATTENZIONE: Errore nel recupero delle strisce pedonali: {e}. Presumo nessuna striscia pedonale.")
        crosswalk_nodes = pd.Index([])

    #traffic_signals_nodes = ox.features_from_place(
    #    G.graph['place'], tags={'highway': 'traffic_signals'}
    #).index
    #crosswalk_nodes = ox.features_from_place(
    #    G.graph['place'], tags={'highway': 'crossing'}
    #).index

    nodes_features['has_traffic_signals'] = nodes_features.index.isin(traffic_signals_nodes).astype(int)
    nodes_features['has_crosswalk'] = nodes_features.index.isin(crosswalk_nodes).astype(int)
    print("Features semafori/strisce pedonali assegnate.")


    # --- Gestione Dati Demografici ---
    #print("Assegnazione densità di popolazione...")
    #if df_demographics_processed is not None and not df_demographics_processed.empty:
    #    # Questo blocco verrebbe usato se avessi un GeoDataFrame di demografia reale
    #    # con poligoni (es. sezioni censuarie) e dati di densità al loro interno.
    #    # Qui faresti una spazial join (sjoin) tra i nodi e i poligoni demografici.
    #    print("Dati demografici granulari disponibili. Assegnazione basata sui dati reali (simulazione/placeholder).")
    #    # Per ora, usiamo una simulazione o un placeholder se non è implementata la sjoin reale.
    #    # nodes_features['densita_popolazione_vicina'] = df_demographics_processed['density_column'].loc[matching_osmid_index]
    #    nodes_features['densita_popolazione_vicina'] = np.random.rand(len(nodes_features)) * 1000 # Esempio fittizio
    #else:
    #    # Questo è il blocco di FALLBACK che usa il valore medio della città
    #    print(f"Dati demografici granulari non disponibili o vuoti. Assegnando la densità media della città ({city_avg_pop_density}).")
    #    nodes_features['densita_popolazione_vicina'] = city_avg_pop_density

    print("Features spaziali assegnate ai nodi OSM.")
    return nodes_features

# def assign_spatial_features_to_nodes(nodes_features, pois_gdf, df_demographics_processed, G, city_avg_pop_density=0.0):
#     """
#     ssegna features spaziali (es. POI vicini, densità popolazione) ai nodi.
#     Include un fallback per la densità di popolazione se i dati demografici non sono disponibili.
#
#     Args:
#         nodes_features (pd.DataFrame): DataFrame delle features base dei nodi OSM.
#         pois_gdf (gpd.GeoDataFrame): GeoDataFrame dei POI pre-elaborati.
#         df_demographics_processed (pd.DataFrame o None): DataFrame dei dati demografici pre-elaborati,
#                                                            o None se non disponibili.
#         G (networkx.MultiDiGraph): Grafo della rete stradale OSM.
#         city_avg_pop_density (float): Densità media di popolazione da usare come fallback se
#                                       df_demographics_processed è None o vuoto.
#
#     Returns:
#         pd.DataFrame: DataFrame nodes_features aggiornato con le nuove features spaziali.
#     """
#     print("Assegnazione features spaziali ai nodi OSM...")
#
#     # Assicurati che nodes_features sia un DataFrame e che l'indice sia l'osmid
#     if nodes_features.index.name != 'osmid':
#         nodes_features = nodes_features.copy()
#         nodes_features['osmid'] = nodes_features.index
#         nodes_features.set_index('osmid', inplace=True)
#
#     # Inizializza le nuove colonne a zero
#     nodes_features['num_scuole_vicine'] = 0
#     nodes_features['num_ospedali_vicini'] = 0
#     nodes_features['num_stazioni_bus_vicine'] = 0
#     nodes_features['num_incroci_principali_vicini'] = 0 # Esempio di feature aggiuntiva
#
#     # Aggiungi colonne per la presenza di semafori/strisce pedonali (se non già presenti da preprocessing)
#     # Queste dovrebbero già venire da get_node_edge_features_from_osm
#     if 'is_traffic_signal_node' not in nodes_features.columns:
#         nodes_features['is_traffic_signal_node'] = nodes_features.apply(
#             lambda row: 1 if G.nodes[row.name].get('highway') == 'traffic_signals' else 0, axis=1
#         )
#     if 'is_crossing_node' not in nodes_features.columns:
#         nodes_features['is_crossing_node'] = nodes_features.apply(
#             lambda row: 1 if G.nodes[row.name].get('highway') == 'crossing' else 0, axis=1
#         )
#     # Rinomina per coerenza con l'output delle raccomandazioni
#     nodes_features.rename(columns={
#         'is_traffic_signal_node': 'semaforo_esistente',
#         'is_crossing_node': 'strisce_esistenti'
#     }, inplace=True)
#
#
#     # Gestione dei POI
#     if pois_gdf is not None and not pois_gdf.empty:
#         # Filtra solo i POI che sono punti (per calcolare la distanza)
#         pois_points_gdf = pois_gdf[pois_gdf.geometry.geom_type == 'Point'].copy()
#
#         if not pois_points_gdf.empty:
#             # Assicurati che nodes_features abbia una colonna di geometria per le query spaziali
#             # Usiamo le coordinate y (lat) e x (lon) dei nodi
#             nodes_gdf_temp = gpd.GeoDataFrame(
#                 nodes_features.index,
#                 geometry=[Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes_features.index],
#                 crs="EPSG:4326"
#             )
#             nodes_gdf_temp = nodes_gdf_temp.set_index(nodes_features.index)
#
#             # Proietta in un CRS proiettato per calcoli di distanza in metri
#             nodes_gdf_proj = nodes_gdf_temp.to_crs(epsg=3857) # Web Mercator
#             pois_gdf_proj = pois_points_gdf.to_crs(epsg=3857)
#
#             # Effettua una join spaziale per trovare i POI entro il raggio
#             sjoin_result = gpd.sjoin(nodes_gdf_proj, pois_gdf_proj, how="inner", predicate="dwithin", distance=POI_PROXIMITY_RADIUS_METERS)
#
#             # Aggrega i conteggi per tipo di POI
#             if not sjoin_result.empty:
#                 # Scuole
#                 schools_near_nodes = sjoin_result[sjoin_result['amenity'] == 'school']
#                 if not schools_near_nodes.empty:
#                     nodes_features['num_scuole_vicine'] = schools_near_nodes.groupby(level=0).size()
#
#                 # Ospedali
#                 hospitals_near_nodes = sjoin_result[sjoin_result['amenity'] == 'hospital']
#                 if not hospitals_near_nodes.empty:
#                     nodes_features['num_ospedali_vicine'] = hospitals_near_nodes.groupby(level=0).size()
#
#                 # Stazioni Bus (o fermate bus)
#                 bus_stops_near_nodes = sjoin_result[sjoin_result['highway'] == 'bus_stop'] # o amenity=='bus_station'
#                 if not bus_stops_near_nodes.empty:
#                     nodes_features['num_stazioni_bus_vicine'] = bus_stops_near_nodes.groupby(level=0).size()
#
#             # Riempi i NaN con 0 per i nodi senza POI vicini
#             # OLD: nodes_features['num_scuole_vicine'].fillna(0, inplace=True)
#             nodes_features['num_scuole_vicine'] = nodes_features['num_scuole_vicine'].fillna(0)
#             # OLD: nodes_features['num_ospedali_vicine'].fillna(0, inplace=True)
#             nodes_features['num_ospedali_vicine'] = nodes_features['num_ospedali_vicine'].fillna(0)
#             # OLD: nodes_features['num_stazioni_bus_vicine'].fillna(0, inplace=True)
#             nodes_features['num_stazioni_bus_vicine'] = nodes_features['num_stazioni_bus_vicine'].fillna(0)
#         else:
#             print("GeoDataFrame POI non contiene geometrie di tipo Point. Saltando l'assegnazione dei POI.")
#     else:
#         print("GeoDataFrame POI non disponibile o vuoto. Saltando l'assegnazione dei POI.")
#
#     # Gestione dati demografici ( placeholder, richiede un df_demographics_processed con geometry)
#     #nodes_features['densita_popolazione_vicina'] = 0.0 # Default a 0
#     if df_demographics_processed is not None and not df_demographics_processed.empty:
#         # print("Simulazione assegnazione densità di popolazione (richiede dati demografici georeferenziati).")
#         # # Questa parte è un esempio concettuale e richiederebbe un vero GeoDataFrame di demografia
#         # # Per ora, simuliamo una densità casuale se non ci sono dati reali, o la lasciamo a 0.
#         # if 'geometry' in df_demographics_processed.columns and not nodes_features.empty:
#         #     # Per una vera integrazione, faresti una sjoin tra nodes_gdf_proj e df_demographics_processed.to_crs(epsg=3857)
#         #     # e poi assegneresti la densità media/sommata
#         #     nodes_features['densita_popolazione_vicina'] = np.random.rand(len(nodes_features)) * 1000 # Esempio fittizio
#         # else:
#         #     print("GeoDataFrame demografico non valido o mancante. Densità popolazione impostata a 0.")
#
#         print("Simulazione assegnazione densità di popolazione (richiede dati demografici georeferenziati).")
#         # Questa parte è un esempio concettuale e richiederebbe un vero GeoDataFrame di demografia
#         # Per ora, simuliamo una densità casuale se non ci sono dati reali, o la lasciamo a 0.
#         # Se hai un vero GeoDataFrame demografico, qui faresti una sjoin
#         # Per ora, come fallback di test, puoi ancora usare una densità casuale o 0.
#         nodes_features['densita_popolazione_vicina'] = np.random.rand(len(nodes_features)) * 1000 # Esempio fittizio
#         print("Dati demografici granulari (simulati) assegnati.")
#     else:
#         print(f"Dati demografici elaborati non disponibili o vuoti. Assegnando la densità media della città ({city_avg_pop_density}).")
#         # --- NUOVA LOGICA DI FALLBACK PER DEMOGRAFIA ---
#         nodes_features['densita_popolazione_vicina'] = city_avg_pop_density
#         # --- FINE NUOVA LOGICA ---
#
#     print("Features spaziali assegnate ai nodi OSM.")
#     return nodes_features

# def aggregate_traffic_to_osm_elements(df_traffic_processed, G):
#     """
#     Simula l'aggregazione di dati di traffico (reali o fittizi) ai nodi OSM.
#     Nel nostro caso, generiamo dati fittizi direttamente per i nodi.
#     """
#     print("Aggregazione dati traffico ai nodi OSM...")
#
#     # Estrai i nodi del grafo
#     nodes_gdf = ox.graph_to_gdfs(G, edges=False)
#
#     # Crea un DataFrame per i dati di traffico aggregati ai nodi
#     traffic_data_for_nodes = pd.DataFrame(index=nodes_gdf.index)
#     traffic_data_for_nodes.index.name = 'osmid'
#
#     # Genera dati di traffico fittizi per ciascun nodo OSM
#     # Questi valori possono essere correlati alle caratteristiche del nodo, ad esempio:
#     # nodi con grado alto (incroci complessi) potrebbero avere più traffico.
#     # nodi vicini a scuole/ospedali potrebbero avere traffico pedonale o veicolare specifico.
#
#     # Esempio semplice: traffico basato sul grado del nodo e un po' di casualità
#     node_degrees = pd.Series({node: G.degree[node] for node in G.nodes})
#
#     traffic_data_for_nodes['avg_conteggio_veicoli'] = node_degrees.reindex(nodes_gdf.index).fillna(0) * 100 + np.random.randint(0, 500, len(nodes_gdf))
#     traffic_data_for_nodes['avg_velocita'] = np.random.uniform(20, 60, len(nodes_gdf)) # Velocità media km/h
#     traffic_data_for_nodes['indice_congestione'] = np.random.uniform(0.1, 0.8, len(nodes_gdf)) # 0-1, 1=molto congestionato
#
#     print("Dati traffico (fittizi) aggregati ai nodi OSM.")
#     return traffic_data_for_nodes.reset_index() # Resetta l'indice per avere 'osmid' come colonna


def aggregate_traffic_to_osm_elements(gdf_traffic_processed, G):
    """
    Aggrega i dati di traffico (reali o fittizi) ai nodi OSM.
    Se gdf_traffic_processed è vuoto, genera dati fittizi direttamente per i nodi.
    """
    print("Aggregazione dati traffico ai nodi OSM...")

    # Estrai i nodi del grafo come GeoDataFrame per le query spaziali
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    #nodes_gdf = nodes_gdf[['geometry', 'osmid']]
    ########if 'osmid' not in nodes_gdf.columns:
        #########nodes_gdf['osmid'] = nodes_gdf.index
    if nodes_gdf.crs != "EPSG:4326":
        nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    if gdf_traffic_processed.crs != "EPSG:4326":
        gdf_traffic_processed = gdf_traffic_processed.to_crs("EPSG:4326")

    traffic_longitudes = gdf_traffic_processed.geometry.x.tolist()
    traffic_latitudes = gdf_traffic_processed.geometry.y.tolist()
    nearest_nodes_osmid = ox.nearest_nodes(G, traffic_longitudes, traffic_latitudes)

    #traffic_points_lat_lon = gdf_traffic_processed[['geometry']].apply(lambda x: (x.y, x.x), axis=1).tolist()
    #nearest_nodes_osmid = ox.nearest_nodes(G, [p[1] for p in traffic_points_lat_lon], [p[0] for p in traffic_points_lat_lon])

    # Crea un DataFrame di mappatura tra l'indice del punto di traffico e l'osmid del nodo più vicino
    traffic_node_mapping = pd.DataFrame({
        'original_index': gdf_traffic_processed.index,
        'osmid': nearest_nodes_osmid
    })

    # Unisci i dati di traffico con gli osmid dei nodi più vicini
    gdf_traffic_with_osmid = gdf_traffic_processed.merge(traffic_node_mapping, left_index=True, right_on='original_index', how='left')

    # Calcola la distanza in metri per filtrare (facoltativo ma raccomandato per precisione)
    # Converti i nodi e i punti di traffico a un CRS che usa metri (es. UTM Zone per Bari: EPSG:32633)
    # o usa haversine distance. Per semplicità, userò una stima grezza o assumerò che ox.nearest_nodes
    # sia sufficientemente vicino. Per un controllo preciso, dovresti convertire il CRS per calcolare la distanza.

    # Potresti calcolare la distanza e filtrare qui se vuoi un controllo più rigoroso sulla vicinanza:
    # traffic_points_proj = gdf_traffic_with_osmid.to_crs(epsg=32633) # Esempio per UTM
    # nodes_proj = nodes_gdf.loc[gdf_traffic_with_osmid['osmid']].to_crs(epsg=32633)
    # distances = traffic_points_proj.geometry.distance(nodes_proj.geometry)
    # gdf_traffic_with_osmid_filtered = gdf_traffic_with_osmid[distances <= max_distance_meters]


    # 3. Aggrega le feature di traffico per ogni nodo OSM
    # Assumiamo che gdf_traffic_processed abbia colonne come 'conteggio_veicoli', 'velocita', 'indice_congestione'
    # Raggruppa per 'osmid' e calcola la media per le metriche numeriche
    aggregated_df = gdf_traffic_with_osmid.groupby('osmid').agg(
        avg_conteggio_veicoli=('ConteggioVeicoli', 'mean'),
        avg_velocita=('VelocitaMedia', 'mean'),
        avg_indice_congestione=('IndiceCongestione', 'mean')
    ).reset_index() # reset_index per rendere osmid una colonna

    # Converti osmid in indice per coerenza con altri GeoDataFrame/DataFrame
    aggregated_df = aggregated_df.set_index('osmid')

    print(f"Aggregazione traffico completata. {len(aggregated_df)} nodi OSM con dati di traffico.")
    return aggregated_df


    # Inizializza un DataFrame per i dati di traffico aggregati ai nodi
    # Verrà riempito o con dati reali aggregati o con dati fittizi
    ####traffic_data_for_nodes = pd.DataFrame(index=nodes_gdf.index)
    ####traffic_data_for_nodes.index.name = 'osmid'
    ####traffic_data_for_nodes['avg_conteggio_veicoli'] = 0.0
    ####traffic_data_for_nodes['avg_velocita'] = 0.0
    ####traffic_data_for_nodes['indice_congestione'] = 0.0
    ####
    ####if gdf_traffic_processed is not None and not gdf_traffic_processed.empty:
    ####    print("Elaborazione dati traffico caricati per l'aggregazione...")
    ####
    ####    # Converti le coordinate dei nodi in un array NumPy per cKDTree
    ####    nodes_coords = np.array([(p.x, p.y) for p in nodes_gdf.geometry])
    ####    tree = cKDTree(nodes_coords)
    ####
    ####    # Converti le coordinate dei sensori/punti di traffico in un array NumPy
    ####    traffic_coords = np.array([(p.x, p.y) for p in gdf_traffic_processed.geometry])
    ####
    ####    # Trova l'indice del nodo OSM più vicino per ogni punto di rilevamento traffico
    ####    distances, node_indices = tree.query(traffic_coords)
    ####
    ####    # Assegna l'OSMID del nodo più vicino a ciascun rilevamento traffico
    ####    #gdf_traffic_processed['nearest_node_osmid'] = nodes_gdf.iloc[node_indices]['osmid'].values
    ####    gdf_traffic_processed['nearest_node_osmid'] = nodes_gdf.index[node_indices].values
    ####
    ####    # Aggrega le metriche di traffico per nodo OSM
    ####    # Possiamo prendere la media delle rilevazioni per ConteggioVeicoli, VelocitaMedia, IndiceCongestione
    ####    traffic_agg = gdf_traffic_processed.groupby('nearest_node_osmid').agg(
    ####        avg_conteggio_veicoli=('ConteggioVeicoli', 'mean'),
    ####        avg_velocita=('VelocitaMedia', 'mean'),
    ####        indice_congestione=('IndiceCongestione', 'mean')
    ####    ).reset_index()
    ####
    ####    traffic_agg.rename(columns={'nearest_node_osmid': 'osmid'}, inplace=True)
    ####
    ####    # Unisci i dati aggregati nel DataFrame finale
    ####    traffic_data_for_nodes = traffic_data_for_nodes.reset_index().merge(
    ####        traffic_agg, on='osmid', how='left', suffixes=('_old', '')
    ####    ).set_index('osmid')
    ####
    ####    # Riempi i NaN (nodi senza dati traffico) con 0
    ####    # OLD: traffic_data_for_nodes['avg_conteggio_veicoli'].fillna(0.0, inplace=True)
    ####    traffic_data_for_nodes['avg_conteggio_veicoli'] = traffic_data_for_nodes['avg_conteggio_veicoli'].fillna(0.0)
    ####    # OLD: traffic_data_for_nodes['avg_velocita'].fillna(0.0, inplace=True)
    ####    traffic_data_for_nodes['avg_velocita'] = traffic_data_for_nodes['avg_velocita'].fillna(0.0)
    ####    # OLD: traffic_data_for_nodes['indice_congestione'].fillna(0.0, inplace=True)
    ####    traffic_data_for_nodes['indice_congestione'] = traffic_data_for_nodes['indice_congestione'].fillna(0.0)
    ####
    ####    print("Dati traffico reali/caricati aggregati ai nodi OSM.")
    ####
    ####else:
    ####    #print("ATTENZIONE: Dati traffico non disponibili o vuoti. Generazione dati traffico fittizi in tempo reale.")
    ####    # Genera dati di traffico fittizi per ciascun nodo OSM
    ####    node_degrees = pd.Series({node: G.degree[node] for node in G.nodes})
    ####
    ####    # Applica i dati fittizi direttamente al DataFrame traffic_data_for_nodes
    ####    traffic_data_for_nodes['avg_conteggio_veicoli'] = node_degrees.reindex(nodes_gdf.index).fillna(0) * 100 + np.random.randint(0, 500, len(nodes_gdf))
    ####    traffic_data_for_nodes['avg_velocita'] = np.random.uniform(20, 60, len(nodes_gdf))
    ####    traffic_data_for_nodes['indice_congestione'] = np.random.uniform(0.1, 0.8, len(nodes_gdf))
    ####    print("Dati traffico (fittizi) generati in tempo reale e aggregati ai nodi OSM.")
    ####
    ####return traffic_data_for_nodes.reset_index() # Resetta l'indice per avere 'osmid' come colonna

def aggregate_features_by_node(nodes_features_gdf, incidents_gdf, traffic_aggregated_df, pois_gdf, buffer_distance=100):
    """
    Aggrega tutte le feature (OSM, incidenti, traffico, POI) a livello di nodo OSM.

    Args:
        nodes_features_gdf (gpd.GeoDataFrame): GeoDataFrame dei nodi OSM con le feature estratte.
        incidents_gdf (gpd.GeoDataFrame): GeoDataFrame degli incidenti pre-elaborati.
        traffic_aggregated_df (pd.DataFrame): DataFrame delle feature di traffico aggregate per nodo OSM.
        pois_gdf (gpd.GeoDataFrame): GeoDataFrame dei POI pre-elaborati.
        buffer_distance (int): Distanza in metri per il buffer attorno ai nodi per aggregare incidenti e POI.

    Returns:
        pd.DataFrame: Un DataFrame finale dove ogni riga è un nodo OSM e le colonne sono tutte le feature aggregate.
    """
    print(f"Aggregazione di tutte le feature ai nodi OSM con buffer di {buffer_distance} metri...")

    #print(f"LOG --> Nodes GDF shape: {nodes_features_gdf.shape}")
    #print(f"LOG --> Incidents GDF shape: {incidents_gdf.shape if incidents_gdf is not None else 'None'}")
    #print(f"LOG --> POIs GDF shape: {pois_gdf.shape if pois_gdf is not None else 'None'}")
    #print(f"LOG --> Traffic Aggregated DF shape: {traffic_aggregated_df.shape if traffic_aggregated_df is not None else 'None'}")
    #print(f"LOG --> Radius Meters: {buffer_distance}")
    # 1. Preparazione della base: nodes_features_gdf è il nostro punto di partenza
    #final_features_df = nodes_features_gdf.copy()
    ######################final_features_df = nodes_features_gdf.reset_index() ------------------<<<<<<<<<
    final_features_df = nodes_features_gdf.copy()

    # --- Assicurati che tutti i GeoDataFrame siano nello stesso CRS proiettato ---
    # Questo è FONDAMENTALE per calcoli di distanza accurati (es. buffer, intersects)
    # Si consiglia un CRS metrico (es. EPSG:3857 Web Mercator o un UTM locale per l'Italia)
    ##########################
    #target_crs = "EPSG:4326" # (UTM Zone 33N per l'Italia centrale/sud)

    if final_features_df.crs is None:
        raise ValueError("nodes_features_gdf non ha un CRS definito. Assicurati che il GeoDataFrame abbia un CRS.")

    # Se il CRS è geografico (lat/lon), stima il CRS UTM
    if final_features_df.crs.is_geographic:
        # Usa il CRS UTM stimato dal centroide per coerenza
        # Non usare .estimate_utm_crs() su unary_union.centroid.to_crs(4326)
        # Usa direttamente sul GeoDataFrame principale.
        target_crs_proj = final_features_df.estimate_utm_crs()
        print(f"CRS target stimato: {target_crs_proj.to_string()}")
    else:
        target_crs_proj = final_features_df.crs # Se è già proiettato, usa quello

    # Converte tutti i GeoDataFrame al CRS proiettato metrico
    final_features_df = final_features_df.to_crs(target_crs_proj)
    incidents_gdf = incidents_gdf.to_crs(target_crs_proj)

    # Preparazione di pois_gdf: Salviamo l'indice in una colonna prima della proiezione
    ###################### AGGIUNGERE DOPO??
    if pois_gdf is not None and not pois_gdf.empty:
        # Aggiungiamo una colonna con l'indice originale del POI, che useremo poi
        pois_gdf['original_poi_id'] = pois_gdf.index
        pois_gdf = pois_gdf.to_crs(target_crs_proj)
    ###################### pois_gdf = pois_gdf.to_crs(target_crs_proj)
    print(f"GeoDataFrame convertiti al CRS proiettato: {target_crs_proj.to_string()}")

    #if final_features_df.crs != target_crs:
    #    final_features_df = final_features_df.to_crs(target_crs)
    #if incidents_gdf.crs != target_crs:
    #    incidents_gdf = incidents_gdf.to_crs(target_crs)
    #if pois_gdf.crs != target_crs:
    #    pois_gdf = pois_gdf.to_crs(target_crs)

    # Assicurati che l'indice sia 'osmid' per unioni future e sia una colonna
    #if 'osmid' not in final_features_df.columns:
    #if final_features_df.index.name != 'osmid':
    #    final_features_df['osmid'] = final_features_df.index
    #else:
    #    final_features_df = final_features_df.set_index('osmid')

    if final_features_df.index.name != 'osmid':
        if 'osmid' in final_features_df.columns:
            final_features_df = final_features_df.set_index('osmid')
        else:
            print("AVVISO: La colonna 'osmid' non trovata. L'indice corrente verrà usato come 'osmid'.")
            final_features_df['osmid'] = final_features_df.index.copy() # Rendi l'indice una colonna temporanea

    # 2. Aggregazione dei Dati Incidenti per Nodo
    # Inizializza la colonna degli incidenti a 0
    final_features_df['num_incidenti_vicini'] = 0

    # Controlla se ci sono incidenti da aggregare
    if not incidents_gdf.empty:
        # Esegui un join spaziale per trovare gli incidenti che intersecano i buffer
        # Crea i buffer attorno ai nodi
        nodes_buffered = final_features_df.geometry.buffer(buffer_distance)
        nodes_buffered_gdf = gpd.GeoDataFrame(
            {'geometry': nodes_buffered}, # Dizionario per specificare il nome della colonna 'geometry'
            index=final_features_df.index, # Imposta l'indice esplicitamente sull'osmid
            crs=target_crs_proj
        )
        nodes_buffered_gdf.index.name = 'osmid' # Assicurati che il nome dell'indice sia 'osmid'

        #nodes_buffered_gdf = gpd.GeoDataFrame(final_features_df.index, geometry=nodes_buffered, crs=target_crs_proj)
        #nodes_buffered_gdf.set_index(final_features_df.index, inplace=True) # Assicura che l'indice sia osmid

        # Esegui il join spaziale
        sjoin_incidents = gpd.sjoin(incidents_gdf, nodes_buffered_gdf, how="inner", predicate="intersects")

        #print(f"\n--- DEBUG: Dati di sjoin_incidents dopo il join spaziale ---")
        #print(f"sjoin_incidents è vuoto? {sjoin_incidents.empty}")
        #if not sjoin_incidents.empty:
        #    print(f"Colonne di sjoin_incidents: {sjoin_incidents.columns.tolist()}")
        #    print(f"Prime 5 righe di sjoin_incidents:\n{sjoin_incidents.head()}")
        #    print(f"Informazioni di sjoin_incidents:\n")
        #    sjoin_incidents.info()
        #print(f"--- FINE DEBUG INTENSIVO ---\n")

        if not sjoin_incidents.empty:
            incident_counts = sjoin_incidents.groupby('osmid').size()
            final_features_df['num_incidenti_vicini'] = final_features_df['num_incidenti_vicini'].add(incident_counts, fill_value=0)
            print(f"Feature incidenti aggregate. Nodi con incidenti vicini: {final_features_df['num_incidenti_vicini'].astype(bool).sum()}")
        else:
            print("Nessuna intersezione tra incidenti e nodi bufferizzati. 'num_incidenti_vicini' rimane a zero.")

        # Conta gli incidenti per ogni nodo
        #incident_counts = sjoin_incidents.groupby(sjoin_incidents.index_right).size()
        #incident_counts = sjoin_incidents.groupby('index_right').size()

        # Aggiorna la colonna nel DataFrame finale
        #final_features_df['num_incidenti_vicini'] = final_features_df['num_incidenti_vicini'].add(incident_counts, fill_value=0)

        #print(f"Feature incidenti aggregate. Nodi con incidenti vicini: {final_features_df['num_incidenti_vicini'].astype(bool).sum()}")
    else:
        print("Nessun dato incidenti da aggregare. 'num_incidenti_vicini' rimane a zero.")

    # Cicla sui nodi e conta gli incidenti nel buffer
    #for idx, node_row in final_features_df.iterrows():
    #    node_buffer = node_row.geometry.buffer(buffer_distance)
    #    incidents_in_buffer = incidents_gdf[incidents_gdf.geometry.intersects(node_buffer)]
    #    final_features_df.loc[idx, 'num_incidenti_vicini'] = len(incidents_in_buffer)

    #print(f"Feature incidenti aggregate. Nodi con incidenti vicini: {final_features_df['num_incidenti_vicini'].astype(bool).sum()}")


    # 3. Aggregazione dei Dati POI per Nodo
    # Inizializza le colonne POI a 0
    # Puoi aggiungere altre categorie di POI se ne hai estratte di specifiche
    final_features_df['num_pois_vicini'] = 0 # Conteggio generale dei POI

    # NEW
    final_features_df['num_attraversamenti_pedonali_vicini'] = 0
    final_features_df['num_scuole_vicine'] = 0
    final_features_df['num_negozi_vicini'] = 0

    # Un esempio per categorie specifiche se il tuo pois_gdf ha una colonna 'type' o 'amenity'
    # Forse hai tipi specifici che vuoi contare (es. 'num_ristoranti_vicini', 'num_scuole_vicine')
    # if 'amenity' in pois_gdf.columns:
    #     for amenity_type in pois_gdf['amenity'].unique():
    #         final_features_df[f'num_{amenity_type}_vicini'] = 0

    ########## POSSIBILE EVOLUTIVA ##########
    ##########possible_amenity_cols = []
    ##########if pois_gdf is not None and not pois_gdf.empty and 'amenity' in pois_gdf.columns:
    ##########    possible_amenity_cols = [f'poi_amenity_{amenity_type}' for amenity_type in pois_gdf['amenity'].unique()]
    ##########    for col_name in possible_amenity_cols:
    ##########        if col_name not in final_features_df.columns:
    ##########            final_features_df[col_name] = 0

    # Controlla se ci sono POI da aggregare
    if pois_gdf is not None and not pois_gdf.empty:
        ############
        # Riutilizzo dei buffer dei nodi creati in precedenza per gli incidenti
        if 'nodes_buffered_gdf' not in locals():
            nodes_buffered = final_features_df.geometry.buffer(buffer_distance)
            nodes_buffered_gdf = gpd.GeoDataFrame(
                {'geometry': nodes_buffered},
                index=final_features_df.index,
                crs=target_crs_proj
            )
            nodes_buffered_gdf.index.name = 'osmid'
        sjoin_pois = gpd.sjoin(pois_gdf, nodes_buffered_gdf, how="inner", predicate="intersects")

        # *** AGGIUNTO: Controllo se il risultato della sjoin è vuoto ***
        ###################### AGGIUNGERE DOPO??
        ###################### if not sjoin_pois.empty:
        if not sjoin_pois.empty and 'original_poi_id' in sjoin_pois.columns: # <--- Controllo sulla nuova colonna
            # Aggrega i POI presenti nel raggio specificato
            ###################### poi_counts = sjoin_pois.groupby('osmid').size()
            # In questo caso li conteggia separatamente se nel raggio considerato uno stesso POI appaia più volte
            poi_counts = sjoin_pois.groupby('osmid')['original_poi_id'].nunique()
            final_features_df['num_pois_vicini'] = final_features_df['num_pois_vicini'].add(poi_counts, fill_value=0)
            ######poi_counts = sjoin_pois.groupby('osmid')['index_left'].nunique()

            ##########if 'amenity' in pois_gdf.columns:
            ##########    amenity_dummies = pd.get_dummies(sjoin_pois['amenity'], prefix='poi_amenity')
            ##########    sjoin_pois_with_dummies = pd.concat([sjoin_pois, amenity_dummies], axis=1)

            ##########    amenity_counts_by_node = sjoin_pois_with_dummies.groupby('osmid')[amenity_dummies.columns].sum()

            ##########    # Update existing columns with actual counts, or add new ones (which were initialized to 0)
            ##########    final_features_df.update(amenity_counts_by_node)

            # Filtra i POI che sono considerati attraversamenti pedonali
            pedestrian_related_pois_sjoin = sjoin_pois[
                #(sjoin_pois.get('highway') == 'traffic_signals') | # Semaforo principale #COMMENTARE SE NON VUOI CONSIDERARLO
                (sjoin_pois.get('crossing_ref') == 'zebra') |      # Strisce pedonali zebra
                (sjoin_pois.get('crossing') == 'marked') |         # Attraversamento marcato
                (sjoin_pois.get('crossing') == 'traffic_signals') # Attraversamento semaforizzato
                #(sjoin_pois.get('highway') == 'crossing')          # Attraversamenti pedonali generici #COMMENTARE SE NON VUOI CONSIDERARLO
                ].copy() # Usiamo .copy() per prevenire SettingWithCopyWarning

            ###################### AGGIUNGERE DOPO??
            ###################### if not pedestrian_related_pois_sjoin.empty: ###################### and 'index_left' in pedestrian_related_pois_sjoin.columns: # <--- Nuovo controllo
            ######################     ###################### distinct_pedestrian_pois_per_node = pedestrian_related_pois_sjoin.groupby('osmid')['index_left'].nunique()
            ######################     distinct_pedestrian_pois_per_node = pedestrian_related_pois_sjoin.groupby('osmid').size()
            ######################     final_features_df['num_attraversamenti_pedonali_vicini'] = \
            ######################         final_features_df['num_attraversamenti_pedonali_vicini'].add(distinct_pedestrian_pois_per_node, fill_value=0)
            ###################### else:
            ######################     print("Nessun POI relativo ad attraversamenti pedonali trovato dopo il filtraggio o colonna 'index_left' mancante. 'num_attraversamenti_pedonali_vicini' rimane zero.")

            ###################### AGGIUNGERE DOPO??
            if not pedestrian_related_pois_sjoin.empty and 'original_poi_id' in pedestrian_related_pois_sjoin.columns:
                distinct_pedestrian_pois_per_node = pedestrian_related_pois_sjoin.groupby('osmid')['original_poi_id'].nunique()
                final_features_df['num_attraversamenti_pedonali_vicini'] = \
                    final_features_df['num_attraversamenti_pedonali_vicini'].add(distinct_pedestrian_pois_per_node, fill_value=0)
            else:
                print("Nessun POI relativo ad attraversamenti pedonali trovato dopo il filtraggio o colonna 'original_poi_id' mancante. 'num_attraversamenti_pedonali_vicini' rimane zero.")

            schools = sjoin_pois[(sjoin_pois.get('amenity') == 'school') | (sjoin_pois.get('amenity') == 'kindergarten')]
            if not schools.empty:
                school_counts = schools.groupby('osmid').size()
                final_features_df['num_scuole_vicine'] = final_features_df['num_scuole_vicine'].add(school_counts, fill_value=0)

            commercial_pois = sjoin_pois[
                (sjoin_pois.get('shop').notna())               # Qualsiasi tipo di negozio (es. shop=supermarket, shop=clothes)
                #(sjoin_pois.get('amenity') == 'restaurant') |
                #(sjoin_pois.get('amenity') == 'cafe') |
                #(sjoin_pois.get('amenity') == 'fast_food') |
                #(sjoin_pois.get('amenity') == 'bar')
                ].copy() # .copy() è importante per evitare SettingWithCopyWarning

            if not commercial_pois.empty and 'original_poi_id' in commercial_pois.columns:
                # Contiamo il numero di negozi/attività commerciali distinti per nodo
                commercial_counts = commercial_pois.groupby('osmid')['original_poi_id'].nunique()
                final_features_df['num_negozi_vicini'] = final_features_df['num_negozi_vicini'].add(commercial_counts, fill_value=0)
            else:
                print("Nessun POI commerciale trovato dopo il filtraggio o colonna 'original_poi_id' mancante. 'num_negozi_vicini' rimane zero.")

            print(f"Feature POI aggregate. Nodi con POI vicini: {final_features_df['num_pois_vicini'].astype(bool).sum()}")
            print(f"Nodi con attraversamenti pedonali vicini: {final_features_df['num_attraversamenti_pedonali_vicini'].astype(bool).sum()}")
            print(f"Nodi con scuole vicine: {final_features_df['num_scuole_vicine'].astype(bool).sum()}")
            print(f"Nodi con negozi/attività commerciali vicine: {final_features_df['num_negozi_vicini'].astype(bool).sum()}")
        else:
            print("Nessuna intersezione tra POI e nodi bufferizzati. 'num_pois_vicini' e le feature per amenity rimangono a zero.")
    else:
        print("Nessun dato POI da aggregare. 'num_pois_vicini' rimane a zero.")

    # Controlla se ci sono POI da aggregare
    #if not pois_gdf.empty:
    #    # Esegui un join spaziale per trovare i POI che intersecano i buffer
    #    # Riutilizzo dei buffer dei nodi creati in precedenza
    #    sjoin_pois = gpd.sjoin(pois_gdf, nodes_buffered_gdf, how="inner", predicate="intersects")
#
    #    # Conta i POI per ogni nodo
    #    # poi_counts = sjoin_pois.groupby(sjoin_pois.index_right).size()
    #    poi_counts = sjoin_pois.groupby('index_right').size() # Usa il nome della colonna
#
    #    # Aggiorna la colonna nel DataFrame finale
    #    final_features_df['num_pois_vicini'] = final_features_df['num_pois_vicini'].add(poi_counts, fill_value=0)
#
    #    # Se hai categorie specifiche di POI e vuoi aggregarle:
    #    # Assicurati che 'amenity' sia una colonna nel tuo pois_gdf
    #    if 'amenity' in pois_gdf.columns:
    #        # Crea un one-hot encoding delle amenità per i POI che sono stati "giunti" ai nodi
    #        # e poi raggruppa per nodo (index_right)
    #        amenity_dummies = pd.get_dummies(sjoin_pois['amenity'], prefix='poi_amenity')
    #        # Unisci amenity_dummies con sjoin_pois per poter raggruppare per index_right
    #        sjoin_pois_with_dummies = pd.concat([sjoin_pois, amenity_dummies], axis=1)
#
    #        # Somma le occorrenze per ogni tipo di amenity per ogni nodo
    #        amenity_counts_by_node = sjoin_pois_with_dummies.groupby(sjoin_pois_with_dummies.index_right)[amenity_dummies.columns].sum()
#
    #        # Unisci queste nuove colonne al DataFrame finale
    #        final_features_df = final_features_df.merge(
    #            amenity_counts_by_node,
    #            left_index=True,
    #            right_index=True,
    #            how='left'
    #        )
    #        # Riempi NaN con 0 per le nuove colonne di amenità
    #        for col in amenity_dummies.columns:
    #            final_features_df[col] = final_features_df[col].fillna(0).astype(int)
#
    #print(f"Feature POI aggregate. Nodi con POI vicini: {final_features_df['num_pois_vicini'].astype(bool).sum()}")

    #for idx, node_row in final_features_df.iterrows():
    #    node_buffer = node_row.geometry.buffer(buffer_distance)
    #    pois_in_buffer = pois_gdf[pois_gdf.geometry.intersects(node_buffer)]

    #    final_features_df.loc[idx, 'num_pois_vicini'] = len(pois_in_buffer)

        # Se hai categorie specifiche di POI:
        # if 'amenity' in pois_gdf.columns:
        #     for amenity_type in pois_in_buffer['amenity'].unique():
        #         final_features_df.loc[idx, f'num_{amenity_type}_vicini'] = (pois_in_buffer['amenity'] == amenity_type).sum()

    #print(f"Feature POI aggregate. Nodi con POI vicini: {final_features_df['num_pois_vicini'].astype(bool).sum()}")


    # 4. Integrazione dei Dati Traffico Aggregati
    # traffic_aggregated_df dovrebbe già essere un DataFrame con 'osmid' come colonna
    # e le feature di traffico aggregate.
    if traffic_aggregated_df is not None and not traffic_aggregated_df.empty:
        # Assicurati che 'osmid' sia l'indice o una colonna nel traffic_aggregated_df
        if 'osmid' in traffic_aggregated_df.columns:
            traffic_aggregated_df = traffic_aggregated_df.set_index('osmid')

        # Unisci le feature di traffico al DataFrame finale
        # Usa 'left' join per mantenere tutti i nodi OSM e riempire con 0.0 dove non ci sono dati traffico
        # Se traffic_aggregated_df contiene solo le colonne avg, selezionale esplicitamente
        traffic_cols = ['avg_conteggio_veicoli', 'avg_velocita', 'avg_indice_congestione']
        # Filtra le colonne che effettivamente esistono nel traffic_aggregated_df
        existing_traffic_cols = [col for col in traffic_cols if col in traffic_aggregated_df.columns]

        if existing_traffic_cols:
            final_features_df = final_features_df.merge(
                traffic_aggregated_df[existing_traffic_cols],
                left_index=True,
                right_index=True,
                how='left'
            )

            # Riempi eventuali NaN (nodi senza dati traffico) con 0
            for col in existing_traffic_cols:
                final_features_df[col] = final_features_df[col].fillna(0.0)

            print("Feature traffico integrate.")
        else:
            print("AVVISO: Nessuna colonna di traffico riconosciuta in traffic_aggregated_df.")
            # Inizializza le colonne a 0.0 se non presenti
            for col in traffic_cols:
                if col not in final_features_df.columns:
                    final_features_df[col] = 0.0

        # Unisci le feature di traffico al DataFrame finale
        # Usa 'left' join per mantenere tutti i nodi OSM e riempire con 0.0 dove non ci sono dati traffico
        #final_features_df = final_features_df.merge(
        #    traffic_aggregated_df[['avg_conteggio_veicoli', 'avg_velocita', 'avg_indice_congestione']],
        #    left_index=True,
        #    right_index=True,
        #    how='left'
        #)

        # Se ora_del_giorno e giorno_della_settimana sono nel traffic_aggregated_df
        # e vuoi aggiungerli come feature globali (assumendo siano unici per l'aggregazione)
        if 'ora_del_giorno' in traffic_aggregated_df.columns and 'ora_del_giorno' not in final_features_df.columns:
            final_features_df['ora_del_giorno'] = traffic_aggregated_df['ora_del_giorno'].iloc[0] if not traffic_aggregated_df.empty else 0
        if 'giorno_della_settimana' in traffic_aggregated_df.columns and 'giorno_della_settimana' not in final_features_df.columns:
            final_features_df['giorno_della_settimana'] = traffic_aggregated_df['giorno_della_settimana'].iloc[0] if not traffic_aggregated_df.empty else 0

        # Riempi eventuali NaN (nodi senza dati traffico) con 0
        #final_features_df['avg_conteggio_veicoli'] = final_features_df['avg_conteggio_veicoli'].fillna(0.0)
        #final_features_df['avg_velocita'] = final_features_df['avg_velocita'].fillna(0.0)
        #final_features_df['avg_indice_congestione'] = final_features_df['avg_indice_congestione'].fillna(0.0)

        ## Se ora_del_giorno e giorno_della_settimana sono nel traffic_aggregated_df
        #if 'ora_del_giorno' in traffic_aggregated_df.columns:
        #    final_features_df['ora_del_giorno'] = traffic_aggregated_df['ora_del_giorno'].iloc[0] # Prendi il primo valore (se fisso o aggregato)
        #if 'giorno_della_settimana' in traffic_aggregated_df.columns:
        #    final_features_df['giorno_della_settimana'] = traffic_aggregated_df['giorno_della_settimana'].iloc[0] # Prendi il primo valore

        #print("Feature traffico integrate.")
    else:
        print("ATTENZIONE: Dati traffico aggregati non disponibili o vuoti. Le feature traffico saranno inizializzate a zero.")
        final_features_df['avg_conteggio_veicoli'] = 0.0
        final_features_df['avg_velocita'] = 0.0
        final_features_df['avg_indice_congestione'] = 0.0
        if 'ora_del_giorno' not in final_features_df.columns:
            final_features_df['ora_del_giorno'] = 0
        if 'giorno_della_settimana' not in final_features_df.columns:
            final_features_df['giorno_della_settimana'] = 0

        #print("ATTENZIONE: Dati traffico aggregati non disponibili o vuoti. Le feature traffico saranno zero.")
        ## Se non ci sono dati di traffico, le colonne sono già a 0.0 dall'inizializzazione
        ## Potresti volerle definire esplicitamente qui se non fossero state inizializzate sopra
        #final_features_df['avg_conteggio_veicoli'] = 0.0
        #final_features_df['avg_velocita'] = 0.0
        #final_features_df['avg_indice_congestione'] = 0.0
        #final_features_df['ora_del_giorno'] = 0 # Default se non dai dati traffico
        #final_features_df['giorno_della_settimana'] = 0 # Default se non dai dati traffico


    # 5. Pulizia Finale e Selezione delle Colonne per il Modello
    # Rimuovi la colonna 'geometry' e altre colonne non numeriche non necessarie per il modello
    # La geometria serve per le operazioni spaziali, ma non per il modello ML stesso
    #if 'geometry' in final_features_df.columns:
    #    final_features_df = final_features_df.drop(columns=['geometry'])

    if 'geometry' in final_features_df.columns:
        final_features_df = final_features_df.drop(columns=['geometry'])

    # Rimuovi la colonna temporanea 'original_poi_id' prima di ritornare il DataFrame finale
    # Assicurati che non venga rimossa se è già l'indice o se non esiste (cosa improbabile)
    ###################### AGGIUNGERE DOPO??
    if 'original_poi_id' in final_features_df.columns:
        final_features_df = final_features_df.drop(columns=['original_poi_id'])

    # Riporta l'indice 'osmid' a una colonna normale se desiderato per i modelli ML
    final_features_df = final_features_df.reset_index()

    # Assicurati che l'indice sia l'osmid e sia una colonna nel DataFrame finale
    #final_features_df = final_features_df.reset_index()

    print("Tutte le feature aggregate con successo per i nodi OSM.")
    return final_features_df

def calculate_incident_severity_features(nodes_features_gdf, incidents_gdf, buffer_distance=100):
    """
    Calcola la gravità media degli incidenti per ogni nodo OSM basandosi
    sugli incidenti all'interno di un buffer specificato.

    Args:
        nodes_features_gdf (gpd.GeoDataFrame): GeoDataFrame dei nodi OSM con le feature.
        incidents_gdf (gpd.GeoDataFrame): GeoDataFrame degli incidenti pre-elaborati (con colonna 'gravità' numerica).
        buffer_distance (int): Distanza in metri per il buffer attorno ai nodi.

    Returns:
        pd.DataFrame: Un DataFrame con 'osmid' e 'gravita_media_incidente'.
    """
    print(f"Calcolo della gravità media degli incidenti per nodo (buffer: {buffer_distance}m)...")

    # Assicurati che il CRS sia proiettato per calcoli di buffer accurati
    if nodes_features_gdf.crs.is_geographic:
        target_crs_proj = nodes_features_gdf.estimate_utm_crs()
    else:
        target_crs_proj = nodes_features_gdf.crs

    nodes_proj = nodes_features_gdf.to_crs(target_crs_proj)
    incidents_proj = incidents_gdf.to_crs(target_crs_proj)

    # Crea i buffer attorno ai nodi
    nodes_buffered = nodes_proj.geometry.buffer(buffer_distance)
    nodes_buffered_gdf = gpd.GeoDataFrame(
        {'geometry': nodes_buffered},
        index=nodes_proj.index, # Importante mantenere l'osmid come indice
        crs=target_crs_proj
    )
    nodes_buffered_gdf.index.name = 'osmid'

    # Esegui il join spaziale
    # Assicurati che incidents_proj contenga la colonna 'gravità'
    if 'Gravita' not in incidents_proj.columns:
        raise ValueError("La colonna 'Gravita' non è presente in incidents_gdf. Assicurati che il pre-processing degli incidenti l'abbia creata e sia numerica.")

    sjoin_incidents = gpd.sjoin(incidents_proj, nodes_buffered_gdf, how="inner", predicate="intersects")

    if not sjoin_incidents.empty:
        # Aggrega per 'osmid' (che è index_right dopo il sjoin)
        incident_severity_agg = sjoin_incidents.groupby('osmid').agg(
            gravita_media_incidente=('Gravita', 'mean')
        ).reset_index()
    else:
        print("Nessuna intersezione tra incidenti e nodi bufferizzati per il calcolo della gravità media.")
        # Se non ci sono intersezioni, crea un DataFrame vuoto con le colonne corrette
        incident_severity_agg = pd.DataFrame(columns=['osmid', 'gravita_media_incidente'])

    return incident_severity_agg

def create_unified_features_df(nodes_features, edges_features, accidents_per_node_df, traffic_aggregated_df, G):
    """
    Crea un DataFrame unificato con tutte le features per ogni nodo/incrocio potenziale.
    Questo sarà il dataset finale per la modellazione.
    """
    print("Creazione del DataFrame unificato delle features (il tuo dataset per la modellazione)...")

    # Inizializza un DataFrame con tutti i nodi incrocio di G e le loro features iniziali
    # nodes_features proviene da get_node_edge_features_from_osm in preprocessing
    unified_df = nodes_features.copy()
    if 'osmid' not in unified_df.columns:
        unified_df['osmid'] = unified_df.index # Assicurati che l'indice sia l'osmid per le unioni

    # Aggiungiamo anche le coordinate dei nodi come features
    # Assicurati che G.nodes[n] contenga 'y' (lat) e 'x' (lon)
    unified_df['latitude'] = [G.nodes[n]['y'] for n in unified_df['osmid']]
    unified_df['longitude'] = [G.nodes[n]['x'] for n in unified_df['osmid']]

    # Unisci le features aggregate degli incidenti
    if accidents_per_node_df is not None and not accidents_per_node_df.empty:
        unified_df = unified_df.merge(accidents_per_node_df[['osmid', 'num_incidenti', 'gravita_media_incidente']], on='osmid', how='left')
        # OLD: unified_df['num_incidenti'].fillna(0, inplace=True)
        unified_df['num_incidenti'] = unified_df['num_incidenti'].fillna(0)
        # OLD: unified_df['gravita_media_incidente'].fillna(0, inplace=True)
        unified_df['gravita_media_incidente'] = unified_df['gravita_media_incidente'].fillna(0)
    else:
        print("Attenzione: Dati incidenti per nodo non disponibili o vuoti. Impostando 'num_incidenti' e 'gravita_media_incidente' a 0.")
        unified_df['num_incidenti'] = 0
        unified_df['gravita_media_incidente'] = 0

    # Unisci le features aggregate del traffico per nodo
    if traffic_aggregated_df is not None and not traffic_aggregated_df.empty:
        # traffic_aggregated_df dovrebbe avere 'osmid' e le colonne di traffico come 'avg_conteggio_veicoli', 'avg_velocita', 'indice_congestione'
        traffic_cols_to_merge = ['avg_conteggio_veicoli', 'avg_velocita', 'avg_indice_congestione']

        # Filtra solo le colonne esistenti in traffic_aggregated_df e nel nostro elenco desiderato
        existing_traffic_cols = [col for col in traffic_cols_to_merge if col in traffic_aggregated_df.columns]

        if existing_traffic_cols:
            unified_df = unified_df.merge(traffic_aggregated_df[['osmid'] + existing_traffic_cols], on='osmid', how='left')
            for col in existing_traffic_cols:
                # OLD: unified_df[col].fillna(0.0, inplace=True)
                unified_df[col] = unified_df[col].fillna(0.0)
        else:
            print("Nessuna colonna di traffico rilevante trovata in traffic_aggregated_df. Impostando colonne traffico a 0.")
            for col in traffic_cols_to_merge:
                unified_df[col] = 0.0
    else:
        print("Attenzione: Dati traffico aggregati non disponibili o vuoti. Impostando le colonne traffico a 0.")
        unified_df['avg_conteggio_veicoli'] = 0.0
        unified_df['avg_velocita'] = 0.0
        unified_df['avg_indice_congestione'] = 0.0

    # Aggiungi un target binario per il rischio di incidente alto (necessario per il classificatore)
    # Basato sulle soglie definite all'inizio del file
    unified_df['ha_rischio_incidente_alto'] = (
            (unified_df['num_incidenti'] >= RISK_THRESHOLD_NUM_ACCIDENTS) |
            (unified_df['gravita_media_incidente'] >= RISK_THRESHOLD_AVG_SEVERITY)
    ).astype(int) # Converti True/False in 1/0

    # Seleziona solo le colonne numeriche/booleane per il modello e quelle che vogliamo come features
    # Rimuovi l'osmid come feature per il modello, ma assicurati che sia l'indice

    # Imposta l'osmid come indice prima di selezionare le features finali
    final_features_df = unified_df.set_index('osmid')

    # Filtra le colonne per il modello (solo quelle numeriche/booleane che sono features)
    # Assicurati che non ci siano colonne come 'geometry' o altre non numeriche che non vuoi nel modello
    final_features_df = final_features_df.select_dtypes(include=[np.number, np.bool_]).copy()

    # Rimuovi il target dal set di feature se per errore è ancora qui
    if 'ha_rischio_incidente_alto' in final_features_df.columns:
        # Lo teniamo per ora nel dataset unificato, ma sarà rimosso prima del training
        # Potrebbe essere utile averlo nel dataset finale per l'analisi.
        pass

    print("DataFrame unificato delle features (il tuo dataset) creato.")
    return final_features_df

# import geopandas as gpd
# import pandas as pd
# import networkx as nx
# import osmnx as ox
# from scipy.spatial import cKDTree
# from shapely.geometry import Point
# import numpy as np
#
# # ... (le altre funzioni come assign_spatial_features_to_nodes, assign_accidents_to_osm_elements, aggregate_traffic_to_osm_elements) ...
#
# def create_unified_features_df(nodes_features, edges_features, accidents_per_node_df, traffic_aggregated_df, G):
#     """
#     Crea un DataFrame unificato con tutte le features per ogni nodo/incrocio potenziale.
#     Questo sarà il dataset finale per la modellazione.
#     """
#     print("Creazione del DataFrame unificato delle features (il tuo dataset per la modellazione)...")
#
#     # Inizializza un DataFrame con tutti i nodi incrocio di G e le loro features iniziali
#     # nodes_features proviene da get_node_edge_features_from_osm
#     unified_df = nodes_features.copy()
#     unified_df['osmid'] = unified_df.index # Assicurati che l'indice sia l'osmid per le unioni
#
#     # Aggiungiamo anche le coordinate dei nodi come features
#     unified_df['latitude'] = [G.nodes[n]['y'] for n in unified_df.index]
#     unified_df['longitude'] = [G.nodes[n]['x'] for n in unified_df.index]
#
#     # Unisci le features aggregate degli incidenti
#     if accidents_per_node_df is not None and not accidents_per_node_df.empty:
#         unified_df = unified_df.merge(accidents_per_node_df, on='osmid', how='left')
#         unified_df['num_incidenti'].fillna(0, inplace=True)
#         unified_df['gravita_media_incidente'].fillna(0, inplace=True)
#     else:
#         print("Attenzione: Dati incidenti per nodo non disponibili o vuoti. Impostando 'num_incidenti' e 'gravita_media_incidente' a 0.")
#         unified_df['num_incidenti'] = 0
#         unified_df['gravita_media_incidente'] = 0
#
#     # Unisci le features aggregate del traffico per nodo
#     if traffic_aggregated_df is not None and not traffic_aggregated_df.empty:
#         # Assumiamo che traffic_aggregated_df abbia una colonna 'osmid' e una 'indice_congestione' o 'avg_conteggio_veicoli'.
#         # Facciamo una media se ci sono più entries per lo stesso nodo.
#         # È importante che la colonna usata per il 'punteggio_traffico' sia consistente con il target del modello di traffico.
#
#         # Consideriamo le features di traffico che vogliamo portare nel dataset unificato
#         traffic_cols_to_merge = ['avg_conteggio_veicoli', 'avg_velocita', 'indice_congestione']
#
#         # Filtra solo le colonne esistenti in traffic_aggregated_df
#         existing_traffic_cols = [col for col in traffic_cols_to_merge if col in traffic_aggregated_df.columns]
#
#         if existing_traffic_cols:
#             traffic_node_agg = traffic_aggregated_df.groupby('osmid')[existing_traffic_cols].mean().reset_index()
#             unified_df = unified_df.merge(traffic_node_agg, on='osmid', how='left')
#             # Riempi i NaN per le colonne di traffico se il merge ha creato dei buchi
#             for col in existing_traffic_cols:
#                 unified_df[col].fillna(0, inplace=True) # O un valore logico, es. media globale
#         else:
#             print("Nessuna colonna di traffico rilevante trovata per l'aggregazione. Impostando le colonne traffico a 0.")
#             for col in traffic_cols_to_merge:
#                 unified_df[col] = 0.0
#     else:
#         print("Attenzione: Dati traffico aggregati non disponibili o vuoti. Impostando le colonne traffico a 0.")
#         # Assicurati che le colonne di traffico esistano, anche se vuote/zero
#         unified_df['avg_conteggio_veicoli'] = 0.0
#         unified_df['avg_velocita'] = 0.0
#         unified_df['indice_congestione'] = 0.0
#
#     # Aggiungi un target binario per il rischio di incidente alto (necessario per il classificatore)
#     # Questa logica dovrebbe essere in preprocess_accidents per coerenza, ma la aggiungiamo qui
#     # per assicurarci che sia presente nel df finale se non lo è già.
#     # Definisci una soglia per "alto rischio" (es. più di X incidenti o gravità media alta)
#     RISK_THRESHOLD_NUM_ACCIDENTS = 3
#     RISK_THRESHOLD_AVG_SEVERITY = 2.5 # Su scala 1-5
#
#     unified_df['ha_rischio_incidente_alto'] = (
#             (unified_df['num_incidenti'] >= RISK_THRESHOLD_NUM_ACCIDENTS) |
#             (unified_df['gravita_media_incidente'] >= RISK_THRESHOLD_AVG_SEVERITY)
#     ).astype(int) # Converti True/False in 1/0
#
#     # Seleziona solo le colonne numeriche/booleane per il modello
#     # Potresti voler specificare esplicitamente le features che vuoi includere
#     final_features_df = unified_df.select_dtypes(include=[np.number, np.bool_]).copy()
#
#     # Rimuovi colonne che non sono features ma ID o target
#     # Asegura che 'osmid' sia l'indice o una colonna, non verrà usato come feature
#     if 'osmid' in final_features_df.columns:
#         final_features_df = final_features_df.set_index('osmid')
#
#     print("DataFrame unificato delle features (il tuo dataset) creato.")
#     return final_features_df