import geopandas as gpd
import pandas as pd
import osmnx as ox

# Queste soglie sono usate per definire un incrocio ad "alto rischio" nel target dei modelli.
RISK_THRESHOLD_NUM_ACCIDENTS = 3
RISK_THRESHOLD_AVG_SEVERITY = 2.5 # Su una scala di gravità

# Soglie per la prossimità dei POI
POI_PROXIMITY_RADIUS_METERS = 300 # Raggio di ricerca per POI vicini a un nodo OSM


def aggregate_traffic_to_osm_elements(gdf_traffic_processed, G):
    """
    Aggrega i dati di traffico (reali o fittizi) ai nodi OSM.
    """
    print("Aggregazione dati traffico ai nodi OSM...")

    # Estrai i nodi del grafo come GeoDataFrame per le query spaziali
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)

    if nodes_gdf.crs != "EPSG:4326":
        nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    if gdf_traffic_processed.crs != "EPSG:4326":
        gdf_traffic_processed = gdf_traffic_processed.to_crs("EPSG:4326")

    traffic_longitudes = gdf_traffic_processed.geometry.x.tolist()
    traffic_latitudes = gdf_traffic_processed.geometry.y.tolist()
    nearest_nodes_osmid = ox.nearest_nodes(G, traffic_longitudes, traffic_latitudes)

    # Crea un DataFrame di mappatura tra l'indice del punto di traffico e l'osmid del nodo più vicino
    traffic_node_mapping = pd.DataFrame({
        'original_index': gdf_traffic_processed.index,
        'osmid': nearest_nodes_osmid
    })

    # Unisci i dati di traffico con gli osmid dei nodi più vicini
    gdf_traffic_with_osmid = gdf_traffic_processed.merge(traffic_node_mapping, left_index=True, right_on='original_index', how='left')

    # Aggrega le feature di traffico per ogni nodo OSM
    # Assumendo che gdf_traffic_processed abbia colonne come 'conteggio_veicoli', 'velocita', 'indice_congestione'
    # Raggruppa per 'osmid' e calcola la media per le metriche numeriche
    aggregated_df = gdf_traffic_with_osmid.groupby('osmid').agg(
        avg_conteggio_veicoli=('ConteggioVeicoli', 'mean'),
        avg_velocita=('VelocitaMedia', 'mean'),
        avg_indice_congestione=('IndiceCongestione', 'mean')
    ).reset_index() # reset_index per rendere osmid una colonna

    # Converti osmid in indice per coerenza con altri GeoDataFrame/DataFrame
    aggregated_df = aggregated_df.set_index('osmid')

    print(f"Aggregazione traffico completata. {len(aggregated_df)} nodi OSM con dati di traffico.\n")
    return aggregated_df


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
    final_features_df = nodes_features_gdf.copy()

    if final_features_df.crs is None:
        raise ValueError("nodes_features_gdf non ha un CRS definito. Assicurati che il GeoDataFrame abbia un CRS.")

    # Se il CRS è geografico (lat/lon), stima il CRS UTM
    if final_features_df.crs.is_geographic:
        # Usa il CRS UTM stimato dal centroide per coerenza
        target_crs_proj = final_features_df.estimate_utm_crs()
        #print(f"CRS target stimato: {target_crs_proj.to_string()}")
    else:
        target_crs_proj = final_features_df.crs # Se è già proiettato, usa quello

    # Converte tutti i GeoDataFrame al CRS proiettato metrico
    final_features_df = final_features_df.to_crs(target_crs_proj)
    incidents_gdf = incidents_gdf.to_crs(target_crs_proj)

    # Salviamo l'indice in una colonna prima della proiezione
    if pois_gdf is not None and not pois_gdf.empty:
        # Aggiungiamo una colonna con l'indice originale del POI
        pois_gdf['original_poi_id'] = pois_gdf.index
        pois_gdf = pois_gdf.to_crs(target_crs_proj)

    #print(f"GeoDataFrame convertiti al CRS proiettato: {target_crs_proj.to_string()}")

    if final_features_df.index.name != 'osmid':
        if 'osmid' in final_features_df.columns:
            final_features_df = final_features_df.set_index('osmid')
        else:
            print("AVVISO: La colonna 'osmid' non trovata. L'indice corrente verrà usato come 'osmid'.")
            final_features_df['osmid'] = final_features_df.index.copy() # Rendi l'indice una colonna temporanea

    # Aggregazione dei Dati Incidenti per Nodo
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
        nodes_buffered_gdf.index.name = 'osmid'

        # Esegui il join spaziale
        sjoin_incidents = gpd.sjoin(incidents_gdf, nodes_buffered_gdf, how="inner", predicate="intersects")

        if not sjoin_incidents.empty:
            incident_counts = sjoin_incidents.groupby('osmid').size()
            final_features_df['num_incidenti_vicini'] = final_features_df['num_incidenti_vicini'].add(incident_counts, fill_value=0)
            print(f"Feature incidenti aggregate. Nodi con incidenti vicini: {final_features_df['num_incidenti_vicini'].astype(bool).sum()}")
        else:
            print("Nessuna intersezione tra incidenti e nodi bufferizzati. 'num_incidenti_vicini' rimane a zero.")

    else:
        print("Nessun dato incidenti da aggregare. 'num_incidenti_vicini' rimane a zero.")

    # Aggregazione dei Dati POI per Nodo
    # Inizializza le colonne POI a 0
    final_features_df['num_pois_vicini'] = 0 # Conteggio generale dei POI
    final_features_df['num_attraversamenti_pedonali_vicini'] = 0
    final_features_df['num_scuole_vicine'] = 0
    final_features_df['num_negozi_vicini'] = 0

    # Controlla se ci sono POI da aggregare
    if pois_gdf is not None and not pois_gdf.empty:
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

        # Controllo se il risultato della sjoin è vuoto
        if not sjoin_pois.empty and 'original_poi_id' in sjoin_pois.columns:
            # Aggrega i POI presenti nel raggio specificato
            # In questo caso li conteggia separatamente se nel raggio considerato uno stesso POI appaia più volte
            poi_counts = sjoin_pois.groupby('osmid')['original_poi_id'].nunique()
            final_features_df['num_pois_vicini'] = final_features_df['num_pois_vicini'].add(poi_counts, fill_value=0)

            # Filtra i POI che sono considerati attraversamenti pedonali
            pedestrian_related_pois_sjoin = sjoin_pois[
                (sjoin_pois.get('crossing_ref') == 'zebra') |      # Strisce pedonali zebra
                (sjoin_pois.get('crossing') == 'marked') |         # Attraversamento marcato
                (sjoin_pois.get('crossing') == 'traffic_signals') # Attraversamento semaforizzato
                #(sjoin_pois.get('highway') == 'crossing')          # Attraversamenti pedonali generici
                ].copy() # Usiamo .copy() per prevenire SettingWithCopyWarning

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
                (sjoin_pois.get('shop').notna()) # Qualsiasi tipo di negozio (es. shop=supermarket, shop=clothes)
                ].copy()

            if not commercial_pois.empty and 'original_poi_id' in commercial_pois.columns:
                # Contiamo il numero di negozi/attività commerciali distinti per nodo
                commercial_counts = commercial_pois.groupby('osmid')['original_poi_id'].nunique()
                final_features_df['num_negozi_vicini'] = final_features_df['num_negozi_vicini'].add(commercial_counts, fill_value=0)
            else:
                print("Nessun POI commerciale trovato dopo il filtraggio o colonna 'original_poi_id' mancante. 'num_negozi_vicini' rimane zero.")

            print(f"Feature POI aggregate. Nodi con POI vicini: {final_features_df['num_pois_vicini'].astype(bool).sum()}")
            #print(f"Nodi con attraversamenti pedonali vicini: {final_features_df['num_attraversamenti_pedonali_vicini'].astype(bool).sum()}")
            #print(f"Nodi con scuole vicine: {final_features_df['num_scuole_vicine'].astype(bool).sum()}")
            #print(f"Nodi con negozi/attività commerciali vicine: {final_features_df['num_negozi_vicini'].astype(bool).sum()}")
        else:
            print("Nessuna intersezione tra POI e nodi bufferizzati. 'num_pois_vicini' e le feature per amenity rimangono a zero.")
    else:
        print("Nessun dato POI da aggregare. 'num_pois_vicini' rimane a zero.")

    # Integrazione dei Dati Traffico Aggregati
    # traffic_aggregated_df dovrebbe già essere un DataFrame con 'osmid' come colonna
    if traffic_aggregated_df is not None and not traffic_aggregated_df.empty:
        if 'osmid' in traffic_aggregated_df.columns:
            traffic_aggregated_df = traffic_aggregated_df.set_index('osmid')

        # Unisci le feature di traffico al DataFrame finale
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

        # Se ora_del_giorno e giorno_della_settimana sono nel traffic_aggregated_df
        if 'ora_del_giorno' in traffic_aggregated_df.columns and 'ora_del_giorno' not in final_features_df.columns:
            final_features_df['ora_del_giorno'] = traffic_aggregated_df['ora_del_giorno'].iloc[0] if not traffic_aggregated_df.empty else 0
        if 'giorno_della_settimana' in traffic_aggregated_df.columns and 'giorno_della_settimana' not in final_features_df.columns:
            final_features_df['giorno_della_settimana'] = traffic_aggregated_df['giorno_della_settimana'].iloc[0] if not traffic_aggregated_df.empty else 0
    else:
        print("ATTENZIONE: Dati traffico aggregati non disponibili o vuoti. Le feature traffico saranno inizializzate a zero.")
        final_features_df['avg_conteggio_veicoli'] = 0.0
        final_features_df['avg_velocita'] = 0.0
        final_features_df['avg_indice_congestione'] = 0.0
        if 'ora_del_giorno' not in final_features_df.columns:
            final_features_df['ora_del_giorno'] = 0
        if 'giorno_della_settimana' not in final_features_df.columns:
            final_features_df['giorno_della_settimana'] = 0


    # Pulizia Finale e Selezione delle Colonne per il Modello
    # Rimuovo la colonna 'geometry' e altre colonne non numeriche non necessarie per il modello
    # La geometria serve per le operazioni spaziali, ma non per il modello ML stesso

    if 'geometry' in final_features_df.columns:
        final_features_df = final_features_df.drop(columns=['geometry'])

    if 'original_poi_id' in final_features_df.columns:
        final_features_df = final_features_df.drop(columns=['original_poi_id'])

    # Riporta l'indice 'osmid' a una colonna normale
    final_features_df = final_features_df.reset_index()

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
    # Importante che incidents_proj contenga la colonna 'gravità'
    if 'Gravita' not in incidents_proj.columns:
        raise ValueError("La colonna 'Gravita' non è presente in incidents_gdf. Assicurati che il pre-processing degli incidenti l'abbia creata e sia numerica.")

    sjoin_incidents = gpd.sjoin(incidents_proj, nodes_buffered_gdf, how="inner", predicate="intersects")

    if not sjoin_incidents.empty:
        # Aggrega per 'osmid'
        incident_severity_agg = sjoin_incidents.groupby('osmid').agg(
            gravita_media_incidente=('Gravita', 'mean')
        ).reset_index()
    else:
        print("Nessuna intersezione tra incidenti e nodi bufferizzati per il calcolo della gravità media.")
        # Se non ci sono intersezioni, crea un DataFrame vuoto con le colonne corrette
        incident_severity_agg = pd.DataFrame(columns=['osmid', 'gravita_media_incidente'])

    return incident_severity_agg
