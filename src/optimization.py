# src/optimization.py
import pandas as pd
import numpy as np

##def score_candidate_locations(df, risk_proba_col, traffic_count_col, severity_col):
##    # Esempio di punteggio: combinazione di probabilità di rischio, conteggio veicoli e gravità media
##    # Puoi regolare i pesi (es. 0.6, 0.2, 0.2) in base a cosa ritieni più importante
##
##    # "Dove dovremmo intervenire prima, considerando sia la probabilità che le potenziali conseguenze e l'impatto?"
##    df['indice_rischio'] = (
##            0.6 * df[risk_proba_col] +
##            0.2 * (df[traffic_count_col] / df[traffic_count_col].max()) + # Normalizza il traffico
##            0.2 * (df[severity_col] / df[severity_col].max())            # Normalizza la gravità
##    )
##    # Rimuovi NaN che potrebbero derivare da max() su colonne con tutti 0 se non gestito bene.
##    df['indice_rischio'] = df['indice_rischio'].fillna(0)
##    return df.sort_values(by='indice_rischio', ascending=False)

def score_candidate_locations(df): # Non servono più i parametri di colonna individuali
    """
    Calcola un 'indice_rischio' per ogni location candidata basandosi su vari fattori.

    Args:
        df (pd.DataFrame): DataFrame contenente le feature dei nodi, incluso 'osmid'
                          e le colonne rilevanti per il calcolo del rischio.

    Returns:
        pd.DataFrame: Il DataFrame originale con l'aggiunta della colonna 'indice_rischio'.
    """

    # Assicurati che 'osmid' sia una colonna, se è l'indice
    if df.index.name == 'osmid':
        df = df.reset_index()
    elif 'osmid' not in df.columns:
        print("Errore: la colonna 'osmid' non è stata trovata nel DataFrame per lo scoring.")
        return pd.DataFrame()

    # Gestione dei valori mancanti (NaN):
    # Per i conteggi e gradi, un NaN potrebbe significare 0 o assenza, quindi 0 è un buon default.
    # Per gravità media, 0 o la media potrebbero essere appropriati. Per il rischio predetto, 0 è sicuro.
    # Per semplicità, possiamo riempire NaNs in alcune colonne prima del ranking/uso.
    # Nota: rank(pct=True) di default gestisce i NaN mettendoli in fondo alla classifica (rank più basso).
    # Se un NaN significa 'nessun rischio' o 'dato non disponibile', il default di rank può essere accettabile.
    # Altrimenti, potresti voler riempire con 0 o con la media/mediana.
    # Esempio: df['num_incidenti_vicini'] = df['num_incidenti_vicini'].fillna(0)
    # Per il momento, assumiamo che rank() gestisca i NaN come più bassi, il che è sensato per il rischio.

    w_num_incidenti = 0.25        # Numero di incidenti storici (molto significativo)
    w_avg_conteggio_veicoli = 0.15 # Volume di traffico
    w_grado_incrocio = 0.10       # Complessità dell'incrocio
    w_num_attraversamenti_pedonali = 0.10 # Rischio pedonale
    w_num_scuole = 0.10           # Presenza di scuole (utenti vulnerabili)
    w_avg_velocita = 0.10         # Velocità media (influenza la gravità)
    w_avg_indice_congestione = 0.05 # Congestione (rischio di tamponamenti minori)
    w_num_pois = 0.075            # Punti di interesse (indica attività generale)
    w_num_negozi = 0.075          # Negozi (indica attività pedonale/veicolare)

    # La somma dei pesi dovrebbe essere 1.0 (o normalizzeremo il risultato finale)
    # (0.35 + 0.20 + 0.15 + 0.10 + 0.10 + 0.10 = 1.00)

    # --- Normalizzazione delle feature tramite ranking percentile ---
    # Rango percentile trasforma il valore in una scala da 0 a 1, indicando la posizione relativa.
    # Valori più alti (es. più incidenti, più traffico, incrocio più complesso) avranno un rango più alto.

    required_columns = [
        'grado_incrocio', 'num_incidenti_vicini', 'num_pois_vicini',
        'num_attraversamenti_pedonali_vicini', 'num_scuole_vicine', 'num_negozi_vicini',
        'avg_conteggio_veicoli', 'avg_velocita', 'avg_indice_congestione'
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"ATTENZIONE: Colonna '{col}' mancante nel DataFrame. Verrà trattata come zeros o NaN da rank().")
            # Puoi scegliere di riempire con 0 o la media qui, se NaN significa assenza di tale caratteristica.
            # df[col] = df[col].fillna(0) # Esempio di riempimento con 0

    ranked_grado_incrocio = df['grado_incrocio'].rank(pct=True, method='average')
    ranked_num_incidenti = df['num_incidenti_vicini'].rank(pct=True, method='average')
    ranked_num_pois = df['num_pois_vicini'].rank(pct=True, method='average')
    ranked_num_attraversamenti_pedonali = df['num_attraversamenti_pedonali_vicini'].rank(pct=True, method='average')
    ranked_num_scuole = df['num_scuole_vicine'].rank(pct=True, method='average')
    ranked_num_negozi = df['num_negozi_vicini'].rank(pct=True, method='average')
    ranked_avg_conteggio_veicoli = df['avg_conteggio_veicoli'].rank(pct=True, method='average')
    ranked_avg_velocita = df['avg_velocita'].rank(pct=True, method='average')
    ranked_avg_indice_congestione = df['avg_indice_congestione'].rank(pct=True, method='average')

    # Combinazione dei fattori per gli utenti vulnerabili e ranking
    #df['score_vulnerabili_raw'] = df['num_scuole_vicine'] + df['num_attraversamenti_pedonali_vicini']
    #ranked_vulnerabili = df['score_vulnerabili_raw'].rank(pct=True, method='average')

    # --- Calcolo dell'indice di rischio combinato ---
    df['indice_rischio'] = (
            w_num_incidenti * ranked_num_incidenti +
            w_avg_conteggio_veicoli * ranked_avg_conteggio_veicoli +
            w_grado_incrocio * ranked_grado_incrocio +
            w_num_attraversamenti_pedonali * ranked_num_attraversamenti_pedonali +
            w_num_scuole * ranked_num_scuole +
            w_num_negozi * ranked_num_negozi +
            w_avg_velocita * ranked_avg_velocita +
            w_avg_indice_congestione * ranked_avg_indice_congestione +
            w_num_pois * ranked_num_pois
    )

    # Assicurati che l'indice_rischio finale sia normalizzato tra 0 e 1, se desiderato.
    # Dato che i pesi sommano a 1 e i ranghi sono tra 0 e 1, il risultato dovrebbe già essere tra 0 e 1.
    # Puoi aggiungere un min-max scaling finale se vuoi essere sicuro (raramente necessario qui):
    # df['indice_rischio'] = (df['indice_rischio'] - df['indice_rischio'].min()) / (df['indice_rischio'].max() - df['indice_rischio'].min())


    return df


###def score_candidate_locations(unified_features_df, traffic_predictor, accident_predictor,
###                              traffic_imputer, accident_imputer,
###                              traffic_features_cols, accident_features_cols):
###    """
###    Calcola un punteggio per ogni potenziale posizione candidata (incrocio o segmento).
###    Il punteggio combina il rischio di incidenti e la congestione del traffico.
###    """
###    print("Valutazione delle posizioni candidate...")
###
###    # Filtra solo i nodi che non hanno già un semaforo o strisce pedonali (se questa info è disponibile)
###    # Assicurati che 'semaforo_esistente' e 'strisce_esistenti' siano colonne numeriche (0 o 1)
###    candidate_locations_df = unified_features_df[
###        (unified_features_df['semaforo_esistente'] == 0) &
###        (unified_features_df['strisce_esistenti'] == 0)
###        ].copy()
###
###    if candidate_locations_df.empty:
###        print("Nessuna posizione candidata trovata che non abbia già semafori o strisce. Tutte le aree sono già attrezzate.")
###        return pd.DataFrame()
###
###    # Prevedi il rischio di incidente per i candidati
###    if accident_predictor:
###        candidate_locations_df['predicted_accident_risk'] = predict_accident_risk(
###            accident_predictor,
###            candidate_locations_df,
###            accident_imputer,
###            accident_features_cols
###        )
###    else:
###        candidate_locations_df['predicted_accident_risk'] = 0.0
###        print("Modello incidenti non disponibile. Il rischio di incidente sarà 0.")
###
###    # Prevedi il flusso di traffico/congestione per i candidati
###    # Questa parte è più complessa perché il traffico dipende dall'ora.
###    # Per una raccomandazione di pianificazione, potremmo voler la media delle ore di punta.
###    # Per semplicità, useremo un punteggio di traffico già calcolato (o una feature derivata).
###    if 'punteggio_traffico' in candidate_locations_df.columns:
###        candidate_locations_df['predicted_congestion_score'] = candidate_locations_df['punteggio_traffico'] # Usiamo la feature aggregata
###    else:
###        candidate_locations_df['predicted_congestion_score'] = 0.0
###        print("Colonna 'punteggio_traffico' non trovata. Il punteggio di congestione sarà 0.")
###
###    # Definisci i pesi per i criteri (puoi aggiustarli)
###    weight_safety = 0.6  # Peso per la sicurezza (rischio incidenti)
###    weight_congestion = 0.3 # Peso per la congestione
###    weight_cost = 0.1    # Peso per il costo (es. i nodi con più strade sono più costosi)
###
###    # Calcola un costo stimato (es. basato sul grado dell'incrocio o su altre features)
###    candidate_locations_df['costo_stimato'] = candidate_locations_df['grado_incrocio'] * 1000 # Costo più alto per incroci complessi
###
###    # Normalizza i punteggi per combinarli (scala da 0 a 1)
###    # Rischio più alto = peggiore, Congestione più alta = peggiore, Costo più alto = peggiore
###    # Per il punteggio finale: vogliamo che sia più alto dove c'è ALTO rischio e ALTA congestione e BASSO costo (se lo consideriamo come un fattore negativo)
###    # Se vogliamo trovare il miglior luogo per l'investimento, allora un alto rischio/congestione dovrebbe dare un alto punteggio.
###    # E un costo dovrebbe abbassare il punteggio o essere gestito come un vincolo.
###
###    # Per questo esempio, definiamo un "indice di priorità" che cresce con il rischio/congestione e diminuisce con il costo.
###    max_risk = candidate_locations_df['predicted_accident_risk'].max()
###    max_congestion = candidate_locations_df['predicted_congestion_score'].max()
###    max_cost = candidate_locations_df['costo_stimato'].max()
###
###    candidate_locations_df['normalized_risk'] = candidate_locations_df['predicted_accident_risk'] / (max_risk + 1e-6)
###    candidate_locations_df['normalized_congestion'] = candidate_locations_df['predicted_congestion_score'] / (max_congestion + 1e-6)
###    candidate_locations_df['normalized_cost'] = candidate_locations_df['costo_stimato'] / (max_cost + 1e-6)
###
###    # Indice di Priorità: (peso_sicurezza * rischio) + (peso_congestione * congestione) - (peso_costo * costo)
###    candidate_locations_df['priority_index'] = (
###            weight_safety * candidate_locations_df['normalized_risk'] +
###            weight_congestion * candidate_locations_df['normalized_congestion'] -
###            weight_cost * candidate_locations_df['normalized_cost']
###    )
###
###    candidate_locations_df.sort_values(by='priority_index', ascending=False, inplace=True)
###    print("Posizioni candidate valutate.")
###    return candidate_locations_df

def recommend_interventions(scored_df):
    top_candidates = scored_df.sort_values(by='indice_rischio', ascending=False)

    recommendations = []
    # Assicurati che scored_df abbia già la colonna 'indice_rischio' e altre features utili
    for index, row in top_candidates.iterrows():
        osmid = row['osmid']
        # Ottieni i valori delle feature per questo nodo
        num_inc = row['num_incidenti_vicini']
        avg_sev = row['gravita_media_incidente']
        #avg_traf = row['avg_conteggio_veicoli']
        risk_proba = row['probabilità_rischio_incidente_predetta']
        score = row['indice_rischio']

        # Esempio di logica per le raccomandazioni
        recommendation_text = ""
        # --- Nuova Logica di Classificazione Priorità ---
        # Definisci le tue soglie per Alta, Media, Bassa basandoti su 'score' (indice_rischio)
        if score >= 0.75: # Esempio: score da 0.75 a 1.0 (o oltre) è Alta
            priority_category = "Priorità Alta"
            if num_inc > 3 and avg_sev > 2: # Molti incidenti, alta gravità
                recommendation_text = "AZIONI URGENTI: Riprogettazione complessiva dell'intersezione, installazione/aggiornamento semafori intelligenti, revisione drastica dei limiti di velocità e controlli rafforzati. Considerare interventi di moderazione del traffico."
            elif num_inc > 0: # Incidenti presenti, anche se meno gravi/frequenti
                recommendation_text = "ALTA PRIORITÀ: Valutazione dettagliata per modifiche geometriche (es. rotonda, canalizzazione), potenziamento segnaletica orizzontale/verticale, analisi e gestione dei flussi veicolari per il controllo della velocità."
            else: # Rischio calcolato alto ma senza incidenti storici (potenziale hotspot)
                recommendation_text = "ALTA PRIORITÀ (PREVENTIVA): Monitoraggio approfondito dei comportamenti di guida e dei flussi di traffico. Indagine per identificare anomalie latenti nel design stradale o nel contesto urbano che generano alto rischio teorico."

        elif score >= 0.5: # Esempio: score da 0.5 a <0.75 è Medio
            priority_category = "Priorità Media"
            if num_inc > 0: # Incidenti presenti
                recommendation_text = "PRIORITÀ MEDIA: Interventi di miglioramento della segnaletica (anche luminosa), potenziamento dell'illuminazione pubblica, analisi punti ciechi e visibilità per utenti vulnerabili (pedoni, ciclisti)."
            else: # Rischio moderato senza incidenti storici
                recommendation_text = "PRIORITÀ MEDIA (PREVENTIVA): Revisione periodica della segnaletica, piccole migliorie alla visibilità in prossimità dell'incrocio, analisi di potenziali conflitti veicolari/pedonali."
        else: # Score < 0.5 = Rischio Basso
            priority_category = "Priorità Bassa"
            recommendation_text = "PRIORITÀ BASSA: Mantenimento dello stato attuale con manutenzione ordinaria della segnaletica e del manto stradale. Controlli di routine per garantire la sicurezza standard."

        recommendations.append({
            'osmid': osmid,
            'indice_rischio': score,
            'probabilità_rischio_incidente_predetta': risk_proba,
            'gravita_media_incidente': avg_sev,
            'num_incidenti_vicini': num_inc,
            #'avg_conteggio_veicoli': avg_traf,
            'intervento': recommendation_text,
            'categoria_priorita': priority_category
        })
    return recommendations

###def recommend_interventions(candidate_locations_scored, top_n=5):
###    """
###    Genera le raccomandazioni finali basate sul punteggio di priorità.
###    """
###    print(f"Generazione delle {top_n} migliori raccomandazioni...")
###    if candidate_locations_scored.empty:
###        print("Nessuna raccomandazione da generare.")
###        return []
###
###    top_recommendations = candidate_locations_scored.head(top_n).copy()
###
###    recommendations_list = []
###    for index, row in top_recommendations.iterrows():
###        rec_type = "Semaforo" if row['predicted_congestion_score'] > row['predicted_accident_risk'] else "Strisce Pedonali" # Esempio di logica
###        if rec_type == "Semaforo" and row['grado_incrocio'] < 3: # Un semaforo non ha senso in una strada dritta
###            rec_type = "Strisce Pedonali o Dosso" # Suggerimento alternativo
###
###        recommendations_list.append({
###            'osmid': row['osmid'],
###            'latitudine': row['geometry'].y,
###            'longitudine': row['geometry'].x,
###            'tipo_raccomandazione': rec_type,
###            'punteggio_priorita': row['priority_index'],
###            'rischio_incidente_predetto': row['predicted_accident_risk'],
###            'punteggio_congestione_predetto': row['predicted_congestion_score'],
###            'grado_incrocio': row['grado_incrocio']
###        })
###
###    print("Raccomandazioni generate.")
###    return recommendations_list