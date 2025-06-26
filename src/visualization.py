import os

import folium
from folium.plugins import MarkerCluster
import osmnx as ox
import pandas as pd
import webbrowser


def plot_recommendations_on_map(G, recommendations_df, filepath="reports/recommendations_map.html"):
    """
    Crea una mappa interattiva con le raccomandazioni evidenziate.
    """
    if G is None or recommendations_df.empty:
        print("Impossibile generare la mappa delle raccomandazioni: dati mancanti o vuoti.")
        return

    priority_colors = {
        "Priorità Alta": "red",
        "Priorità Media": "orange",
        "Priorità Bassa": "green",
    }

    priority_order = [
        "Priorità Alta",
        "Priorità Media",
        "Priorità Bassa",
    ]

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    if nodes_gdf.empty:
        print("Il grafo non contiene nodi. Impossibile calcolare il centro della mappa.")
        return

    center_lat = nodes_gdf['y'].mean()
    center_lon = nodes_gdf['x'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

    # Aggiunge gli edges (strade) per contesto, ma in un layer separato e meno invasivo
    nodes, edges = ox.graph_to_gdfs(G)
    folium.GeoJson(
        edges.geometry.__geo_interface__,
        name="Rete Stradale", # Nome del layer
        style_function=lambda x: {
            "color": "#cccccc", # Colore grigio chiaro per le strade di sfondo
            "weight": 0.8,
            "opacity": 0.5
        },
        show=True
    ).add_to(m)

    # Crea FeatureGroup separati per ogni categoria di priorità
    feature_groups = {}
    for category in priority_order:
        if category == "Priorità Alta":
            show_default = True # Visibile di default
        else:
            show_default = False # Nascosto di default
        feature_groups[category] = folium.FeatureGroup(name=category, show=show_default).add_to(m)


    for idx, row in recommendations_df.iterrows():
        osmid = row['osmid']
        try:
            node_data = G.nodes[osmid]
            lat = node_data['y']
            lon = node_data['x']

            # Recupera la categoria di priorità e il colore
            current_priority_category = row.get('categoria_priorita', 'Priorità Bassa')
            marker_color = priority_colors.get(current_priority_category, "gray") # Fallback a grigio se la categoria non è mappata

            popup_html = f"""
            <h4>Raccomandazione Incrocio: {osmid}</h4>
            <p><strong>Punteggio Priorità:</strong> <span style="color:{marker_color}; font-weight:bold;">{row['indice_rischio']:.2f}</span></p>
            <p><strong>Categoria Priorità:</strong> <span style="color:{marker_color}; font-weight:bold;">{current_priority_category}</span></p>
            <p><strong>Rischio Incidente Predetto:</strong> {row['probabilità_rischio_incidente_predetta']:.3f}</p>
            <p><strong>Gravità Media Incidenti Storici:</strong> {row['gravita_media_incidente']:.2f}</p>
            <p><strong>N. Incidenti (storico):</strong> {row['num_incidenti_vicini']:.0f}</p>
            <p><strong>Raccomandazione Specifico:</strong> <span style="color:blue; font-weight:bold;">{row['intervento']}</span></p>
            """

            iframe = folium.IFrame(html=popup_html, width=350, height=250)
            popup = folium.Popup(iframe, max_width=450)

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                popup=popup,
                tooltip=f"{current_priority_category}: {row['indice_rischio']:.2f}"
            ).add_to(feature_groups[current_priority_category])

        except KeyError as e:
            print(f"KeyError: Colonna o dato '{e}' mancante nel popup o nel tooltip per nodo {osmid}. Saltando la visualizzazione.")
            continue
        except Exception as e:
            print(f"Errore generico durante l'elaborazione del nodo {osmid}: {e}. Saltando la visualizzazione.")
            continue

    folium.LayerControl().add_to(m)

    m.save(filepath)
    print(f"Mappa delle raccomandazioni salvata in: {filepath}.")

    try:
        webbrowser.open(f'file://{os.path.abspath(filepath)}')
        print("Apertura della mappa delle raccomandazioni nel browser...")
    except Exception as e:
        print(f"Impossibile aprire automaticamente la mappa nel browser: {e}")


def plot_osm_data_on_map(G, pois_gdf, filepath="reports/osm_data_map.html"):
    """
    Crea una mappa interattiva per visualizzare la rete stradale OSM e i POI scaricati.
    """
    if G is None:
        print("Impossibile generare la mappa dei dati OSM: grafo non disponibile.")
        return

    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    if nodes_gdf.empty:
        print("Il grafo non contiene nodi. Impossibile calcolare il centro della mappa.")
        return

    center_lat = nodes_gdf['y'].mean()
    center_lon = nodes_gdf['x'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="cartodbpositron")

    # --- Rete Stradale ---
    nodes, edges = ox.graph_to_gdfs(G)

    if edges.crs and not edges.crs.is_geographic:
        edges = edges.to_crs(epsg=4326)

    folium.GeoJson(
        edges.geometry.__geo_interface__,
        name="Rete Stradale OSM", # Nome del layer per la legenda
        style_function=lambda x: {
            "color": "#666666",
            "weight": 0.8,
            "opacity": 0.5
        }
    ).add_to(m)

    # --- POI con Legenda e Layer Controllabili ---

    # Dizionario per mappare i tipi di POI a icone e colori
    poi_icons_config = {
        'traffic_signals': {'icon': 'traffic-light', 'color': 'orange', 'prefix': 'fa', 'name': 'Semaforo'},
        'crossing': {'icon': 'walking', 'color': 'blue', 'prefix': 'fa', 'name': 'Strisce Pedonali'}, #Attraversamenti Pedonali Segnalati e Controllati
        'school': {'icon': 'school', 'color': 'purple', 'prefix': 'fa', 'name': 'Scuola'},
        'hospital': {'icon': 'hospital', 'color': 'red', 'prefix': 'fa', 'name': 'Ospedale'},
        'leisure_area': {'icon': 'tree', 'color': 'darkgreen', 'prefix': 'fa', 'name': 'Area Ricreativa'}, #Considera parchi, ecc.
        'default': {'icon': 'info-circle', 'color': 'blue', 'prefix': 'fa', 'name': 'Altro POI'}
    }

    # Crea un MarkerCluster per i POI di default per evitare l'affollamento
    default_poi_cluster = MarkerCluster(name="Altri Punti di Interesse").add_to(m)

    # Crea FeatureGroup per POI specifici per la leggenda
    traffic_signals_layer = folium.FeatureGroup(name="Semafori").add_to(m)
    crossings_layer = folium.FeatureGroup(name="Strisce Pedonali").add_to(m)
    schools_layer = folium.FeatureGroup(name="Scuole").add_to(m)
    hospitals_layer = folium.FeatureGroup(name="Ospedali").add_to(m)
    parks_playgrounds_layer = folium.FeatureGroup(name="Parchi").add_to(m)

    if pois_gdf is not None and not pois_gdf.empty:
        if pois_gdf.crs and not pois_gdf.crs.is_geographic:
            pois_gdf = pois_gdf.to_crs(epsg=4326)

        for _, row in pois_gdf.iterrows():
            lat, lon = None, None
            geom_type = row.geometry.geom_type

            if geom_type == 'Point':
                lat, lon = row.geometry.y, row.geometry.x
            elif geom_type in ('Polygon', 'LineString'):
                if row.geometry.centroid.is_valid and not row.geometry.centroid.is_empty:
                    lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                else:
                    bounds = row.geometry.bounds
                    if bounds:
                        lat, lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2

            if lat is None or lon is None:
                continue

            popup_html = f"""
            <b>Tipo:</b> {row.get('highway', row.get('amenity', row.get('leisure', row.get('shop', row.get('building', 'N/A')))))}<br>
            <b>Nome:</b> {row.get('name', 'N/A')}<br>
            <b>Indirizzo:</b> {row.get('addr:street', 'N/A')} {row.get('addr:housenumber', '')}<br>
            <b>Dettagli:</b> {', '.join([f"{k}:{v}" for k, v in row.items() if pd.notna(v) and k not in ['geometry', 'osmid', 'name', 'highway', 'amenity', 'addr:street', 'addr:housenumber', 'leisure', 'shop', 'building']])}
            """

            # Determina icona e colore in base al tipo di POI
            icon_config = poi_icons_config['default']
            target_layer = default_poi_cluster

            if 'highway' in row and pd.notna(row['highway']):
                highway_value = str(row['highway']).strip().lower()
                if highway_value == 'traffic_signals':
                    icon_config = poi_icons_config['traffic_signals']
                    target_layer = traffic_signals_layer
                elif highway_value == 'crossing':
                    #Alcune strisce pedonali vengono considerate anche con altro tag OSM (non solo "zebra")
                    if str(row.get('crossing_ref', '')).strip().lower() == 'zebra' or \
                            str(row.get('crossing', '')).strip().lower() == 'marked' or \
                            str(row.get('crossing', '')).strip().lower() == 'traffic_signals':
                        icon_config = poi_icons_config['crossing']
                        target_layer = crossings_layer
            elif 'amenity' in row and pd.notna(row['amenity']):
                amenity_value = str(row['amenity']).strip().lower()
                if amenity_value == 'school' or amenity_value == 'kindergarten':
                    icon_config = poi_icons_config['school']
                    target_layer = schools_layer
                elif amenity_value == 'hospital':
                    icon_config = poi_icons_config['hospital']
                    target_layer = hospitals_layer
            elif 'leisure' in row and pd.notna(row['leisure']):
                leisure_value = str(row['leisure']).strip().lower()
                if leisure_value in ['park', 'playground']:
                    icon_config = poi_icons_config['leisure_area']
                    target_layer = parks_playgrounds_layer

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(
                    color=icon_config['color'],
                    icon=icon_config['icon'],
                    prefix=icon_config['prefix']
                )
            ).add_to(target_layer)
    else:
        print("GeoDataFrame dei POI non disponibile o vuoto. Nessun POI da plottare.")

    # Aggiunge il controllo dei layer alla mappa
    folium.LayerControl().add_to(m)

    m.save(filepath)
    print(f"Mappa interattiva dei dati OSM salvata in: {filepath}.")

    try:
        webbrowser.open(f'file://{os.path.abspath(filepath)}')
        print("Apertura della mappa dei dati OSM nel browser...")
    except Exception as e:
        print(f"Impossibile aprire automaticamente la mappa nel browser: {e}")
