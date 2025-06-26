from owlready2 import *
import os
import datetime


class OntologyManager:
    def __init__(self, ontology_filename="ontology/Ontologia_PianificazioneUrbana.rdf"):
        self.ontology_filename = ontology_filename
        self.onto = None
        self.base_iri = None
        self.pianificazione_urbana_ns = None

        self._load_ontology() # Carica l'ontologia all'inizializzazione

    def _load_ontology(self):
        """Carica l'ontologia dal file specificato."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ontology_path = os.path.join(current_dir, self.ontology_filename)

        try:
            self.onto = get_ontology(f"file://{ontology_path}").load()
            self.base_iri = self.onto.base_iri
            self.pianificazione_urbana_ns = self.onto.get_namespace(self.base_iri)
            print(f"Ontologia '{self.ontology_filename}' caricata con successo.")

            # Tentativo di sincronizzare il ragionatore (opzionale ma utile per le inferenze nelle query)
            # Nota: questo può richiedere tempo e una configurazione appropriata del ragionatore
            # try:
            #     print("Sincronizzazione del ragionatore (potrebbe richiedere tempo)...")
            #     sync_reasoner_pellet() # o sync_reasoner_hermit()
            #     print("Ragionatore sincronizzato.")
            # except Exception as e:
            #     print(f"Attenzione: Impossibile sincronizzare il ragionatore. Errore: {e}")

        except Exception as e:
            print(f"Errore critico nel caricamento dell'ontologia: {e}")
            print("Assicurati che il file esista e che sia sintatticamente corretto.")
            self.onto = None # Imposta a None per indicare fallimento
            self.base_iri = None
            self.pianificazione_urbana_ns = None

    def is_loaded(self):
        """Controlla se l'ontologia è stata caricata con successo."""
        return self.onto is not None

    def visualizza_classi(self):
        """Visualizza le classi nell'ontologia."""
        if not self.is_loaded():
            print("Ontologia non caricata.")
            return

        print("\n--- Classi nell'Ontologia ---")
        found_classes = False
        for cls in self.onto.classes():
            if self.pianificazione_urbana_ns and cls.namespace.base_iri == self.pianificazione_urbana_ns.base_iri:
                print(f"- {cls.name}")
                found_classes = True
        if not found_classes:
            print("Nessuna classe trovata.")
        print("----------------------------")

    def visualizza_proprieta_oggetto(self):
        """Visualizza le proprietà d'oggetto nell'ontologia."""
        if not self.is_loaded():
            print("Ontologia non caricata.")
            return

        print("\n--- Object properties nell'Ontologia ---")
        found_properties = False
        for prop in self.onto.object_properties():
            if self.pianificazione_urbana_ns and prop.namespace.base_iri == self.pianificazione_urbana_ns.base_iri:
                domain_names = [d.name for d in prop.domain] if prop.domain else ["Nessuno"]
                range_names = [r.name for r in prop.range] if prop.range else ["Nessuno"]
                print(f"- {prop.name} (Dominio: {', '.join(domain_names)}, Range: {', '.join(range_names)})")
                found_properties = True
        if not found_properties:
            print("Nessuna Object property trovata.")
        print("------------------------------------------")

    def visualizza_proprieta_dati(self):
        """Visualizza le proprietà dei dati nell'ontologia."""
        if not self.is_loaded():
            print("Ontologia non caricata.")
            return

        print("\n--- Data properties nell'Ontologia ---")
        found_properties = False
        for prop in self.onto.data_properties():
            if self.pianificazione_urbana_ns and prop.namespace.base_iri == self.pianificazione_urbana_ns.base_iri:
                domain_names = [d.name for d in prop.domain] if prop.domain else ["Nessuno"]
                range_names = [str(r).split('#')[-1] for r in prop.range] if prop.range else ["Nessuno"]
                print(f"- {prop.name} (Dominio: {', '.join(domain_names)}, Range: {', '.join(range_names)})")
                found_properties = True
        if not found_properties:
            print("Nessuna Data property trovata.")
        print("-----------------------------------------")

    def esegui_query_ontologia(self):
        """Permette all'utente di eseguire una query predefinita."""
        if not self.is_loaded():
            print("Ontologia non caricata. Impossibile eseguire query.")
            return

        ONT_PREFIX = f"PREFIX ex: <{self.base_iri}>\n"

        PRESET_QUERIES = {
            1: {
                "desc": "Trova tutti gli individui di IncidenteStradale.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT ?incidente
                    WHERE {
                        ?incidente a ex:IncidenteStradale .
                    }
                """
            },
            2: {
                "desc": "Trova incidenti causati da 'Distrazione' e la loro latitudine.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT ?incidente ?latitudine
                    WHERE {
                        ?incidente ex:haCausa ex:CausaIncidente_Distrazione ;
                                   ex:hasLatitudine ?latitudine .
                    }
                """
            },
            3: {
                "desc": f"Trova tutti gli incidenti avvenuti in data odierna ({datetime.date.today().strftime('%Y-%m-%d')}).",
                "type": "SPARQL",
                "query": ONT_PREFIX + f"""
                    SELECT ?incidente ?ora
                    WHERE {{
                        ?incidente ex:hasData "{datetime.date.today().strftime('%Y-%m-%d')}"^^xsd:date ;
                                   ex:hasOra ?ora .
                    }}
                """
            },
            4: {
                "desc": "Trova i Nodi Stradali e la loro longitudine.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT ?nodo ?longitudine
                    WHERE {
                        ?nodo a ex:NodoStradale ;
                              ex:hasLongitudine ?longitudine .
                    }
                """
            },
            5: {
                "desc": "Conta il numero totale di Incidenti Stradali.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT (COUNT(?incidente) AS ?numeroIncidenti)
                    WHERE {
                        ?incidente a ex:IncidenteStradale .
                    }
                """
            },
            6: {
                "desc": "Trova Nodi Stradali con incidenti e punti di interesse vicini.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT DISTINCT 
                        (STRAFTER(STR(?nodo), STR(ex:)) AS ?nodoNome) 
                        (STRAFTER(STR(?poi), STR(ex:)) AS ?poiNome)
                    WHERE {
                        ?incidente a ex:IncidenteStradale ;
                                   ex:occorreVicinoA ?nodo .
                        ?nodo a ex:NodoStradale .
                        
                        OPTIONAL { # Questo blocco è opzionale: il nodo verrà mostrato anche senza POI
                            ?nodo ex:haPuntoDiInteresseVicino ?poi .
                            VALUES ?poiClass { ex:PuntoDiInteresse ex:Scuola ex:Negozio ex:Ospedale }
                            ?poi a ?poiClass .
                        }
                    }
                    ORDER BY ?nodoNome ?poiNome
                """
            },
            7: {
                "desc": "Incidenti avvenuti di sera (dopo le 18:00) con gravità lieve.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT ?incidente ?data ?ora ?latitudine ?longitudine
                    WHERE {
                        ?incidente a ex:IncidenteStradale ;
                                   ex:hasData ?data ;
                                   ex:hasOra ?ora ;
                                   ex:hasLatitudine ?latitudine ;
                                   ex:hasLongitudine ?longitudine ;
                                   ex:haGravita ex:GravitaIncidente_Lieve .
                        FILTER (?ora > "18:00:00"^^xsd:time)
                    }
                """
            },
            8: {
                "desc": "Incidenti avvenuti di mattina (prima delle 12:00) con condizione meteo 'Neve'.",
                "type": "SPARQL",
                "query": ONT_PREFIX + """
                    SELECT ?incidente ?data ?ora
                    WHERE {
                        ?incidente a ex:IncidenteStradale ;
                                   ex:hasData ?data ;
                                   ex:hasOra ?ora ;
                                   ex:haCondizioneMeteo ex:CondizioneMeteo_Neve .
                        FILTER (?ora < "12:00:00"^^xsd:time)
                    }
                """
            }
        }

        print("\n--- Esegui Query Predefinite ---")
        for key, value in PRESET_QUERIES.items():
            print(f"{key}: {value['desc']}")
        print("0: Torna indietro")

        query_choice = -1
        while query_choice not in PRESET_QUERIES and query_choice != 0:
            try:
                query_choice = int(input("\nInserisci il numero della query da eseguire (o 0 per tornare indietro): "))
                if query_choice not in PRESET_QUERIES and query_choice != 0:
                    print("Opzione non valida. Riprova.")
            except ValueError:
                print("Input non valido. Inserisci un numero.")

        if query_choice == 0:
            print("Tornando al menu Ontologia.")
            return

        selected_query_info = PRESET_QUERIES[query_choice]
        print(f"\nEsecuzione della query: '{selected_query_info['desc']}'")

        try:
            results = list(default_world.sparql(selected_query_info["query"]))
            if results:
                print("\nRisultati:")
                for row in results:
                    print(row)
            else:
                print("Nessun risultato per questa query.")
        except Exception as e:
            print(f"Errore nell'esecuzione della query SPARQL: {e}")
        print("------------------")
