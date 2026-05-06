# Analisi dello stato attuale del progetto e possibili sviluppi futuri

## Stato attuale (snapshot tecnico)

Il repository è già una **pipeline Retrieval completa** ben separata per responsabilità:
- configurazione centralizzata (`src/config.py`),
- ingest + chunking (`src/data_loader.py`, `src/chunking.py`),
- embedding e cache (`src/embeddings.py`),
- indicizzazione FAISS + baseline TF-IDF (`src/indexing.py`, `src/tfidf.py`),
- retrieval runtime (`src/retrieval.py`),
- valutazione riproducibile (`src/evaluate_retrieval.py`, `src/metrics.py`),
- UI Streamlit per confronto side-by-side (`app.py`).

### Punti di forza
1. **Architettura chiara e modulare**: i componenti sono disaccoppiati e testabili.
2. **Baseline comparativa presente**: semantic vs TF-IDF evita valutazioni “a vista”.
3. **Caching ben pensato**: embeddings persistiti con fingerprint su testi+modello.
4. **UX demo completa**: CLI, report JSON e UI interattiva.

### Limiti attuali
1. **Benchmark poco sfidante**: dataset sintetico piccolo (14 query) e metriche aggregate in pareggio.
2. **Gap tra ranking e metrica**: ranking su chunk, scoring a livello documento.
3. **RAG minimale**: risposta estrattiva senza ranking answer-level né valutazione dedicata.
4. **Mancanza di metriche IR più fini**: assente nDCG@k / success@1 / analisi robusta long-tail.

## Evidenze principali dal codice e dagli artefatti

- README dichiara esplicitamente dataset sintetico, confronto semantic/TF-IDF e risultati correnti in pareggio su Recall@5/Precision@5/MRR.
- `reports/evaluation_report.json` conferma la parità su metriche aggregate.
- Il retrieval semantico usa **IndexFlatIP** FAISS con embeddings normalizzati (cosine similarity via inner product), scelta corretta per baseline densa.
- L’app mostra metriche aggregate e confronto qualitativo dei risultati ma non dispone ancora di analisi errori strutturata.

## Migliorie immediate già applicate in questo branch

Per ridurre attrito operativo:
- aggiunto `pytest.ini` con `pythonpath = .` e `testpaths = tests`;
- verificato che la suite test sia ora eseguibile out-of-the-box in questo ambiente (`9 passed`).

## Roadmap consigliata

### Fase 1 — Hardening (breve termine)
- CI minima: lint + test + smoke eval.
- Pinning/versioning di modello, indice, dataset e report.
- Validazione input robusta (query vuote, top-k e edge cases).

### Fase 2 — Valutazione credibile
- Espandere dataset con query ambigue, sinonimi, typo, multi-intent.
- Aggiungere **nDCG@k**, success@1, breakdown per categoria query.
- Separare dev/test e tracciare esperimenti (seed + config snapshot).

### Fase 3 — Qualità Retrieval
- Hybrid retrieval (BM25/TF-IDF + dense) con score fusion.
- Re-ranking cross-encoder sui top-N.
- Ottimizzazione chunking (dimensione/overlap adattivi per tipologia documento).

### Fase 4 — Verso produzione
- Osservabilità (latency, failure rate, query drift).
- Regression suite con golden queries.
- Pipeline di aggiornamento indice incrementale e rollback versionato.

## Priorità operative (ordine consigliato)

1. Rendere stabile la qualità del benchmark (dataset + metriche).
2. Introdurre hybrid + reranker e confrontare in A/B su report.
3. Aggiungere CI/monitoring per evitare regressioni silenziose.
4. Portare il RAG da estrattivo a grounded answer con citazioni chunk-level.

## Valutazione sintetica finale

Il progetto è già **molto buono come portfolio tecnico**: completo, leggibile, dimostrabile.
Il prossimo salto di livello non è “più codice”, ma **migliore metodologia di valutazione** e **disciplina operativa** (benchmark realistico, CI, versioning, osservabilità).
