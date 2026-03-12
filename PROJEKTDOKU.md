# Projektdokumentation

## Zweck des Projekts

Dieses Repository stellt ein Benchmarking- und Auswertungs-Environment für:

- `FJSSP` (Flexible Job Shop Scheduling Problem)
- `FJSSP-W` (Flexible Job Shop Scheduling Problem mit Worker-Flexibilität)
- optional robuste Bewertung unter Unsicherheit

bereit.

Der Kernnutzen des Projekts ist nicht ein vollständiges Solver-Framework, sondern eine gemeinsame Arbeitsumgebung für:

- Laden und Parsen von Benchmark-Instanzen
- einheitliche Repräsentation der Instanzen in Python
- Bewertung von Lösungen über Makespan und Worker-Balance
- Vergleich und Visualisierung von Solver-Ergebnissen
- Simulation von Unsicherheit für FJSSP-W

## Schnellstart

### 1. Umgebung aktivieren

Im Repository ist bereits eine virtuelle Umgebung `.venv` vorhanden. Lokal war `python` nicht global verfügbar, daher am besten direkt die Projekt-Umgebung verwenden:

```bash
source .venv/bin/activate
python --version
pip install -r requirements.txt
```

Alternativ ohne Aktivierung:

```bash
./.venv/bin/python benchmark_load_example.py
```

### 2. Abhängigkeiten

Die benötigten Python-Pakete stehen in [`requirements.txt`](/home/jb_itp/projects/FJSSP-W-Competition/requirements.txt):

- `numpy`
- `matplotlib`
- `pandas`
- `autorank`

Für die Notebook-Beispiele ist Jupyter hilfreich; in der vorhandenen `.venv` sind passende Binaries bereits vorhanden.

### 3. Einstiegspunkte

Für den praktischen Start sind diese Dateien relevant:

- [`README.md`](/home/jb_itp/projects/FJSSP-W-Competition/README.md): fachlicher Überblick über Benchmarking und Metriken
- [`index.md`](/home/jb_itp/projects/FJSSP-W-Competition/index.md): Wettbewerbsbeschreibung und Submission-Logik
- [`benchmark_load_example.py`](/home/jb_itp/projects/FJSSP-W-Competition/benchmark_load_example.py): kleinstes Parser-Beispiel
- [`full_example.ipynb`](/home/jb_itp/projects/FJSSP-W-Competition/full_example.ipynb): umfassenderer Workflow
- [`example_visualization.ipynb`](/home/jb_itp/projects/FJSSP-W-Competition/example_visualization.ipynb): Plotting und Vergleich
- [`uncertainty_example.ipynb`](/home/jb_itp/projects/FJSSP-W-Competition/uncertainty_example.ipynb): Unsicherheits-Simulation

## Projektaufbau

### Root-Ebene

- `util/`: Kernlogik des Environments
- `instances/`: Benchmark-Instanzen und Metadaten
- `*.ipynb`: interaktive Beispiele
- `benchmark_load_example.py`: minimales CLI-artiges Beispiel
- `README.md` / `index.md`: fachliche Dokumentation

### Datenstruktur in `instances/`

- `instances/Instances_FJSSP/`: klassische FJSSP-Instanzen, nach Benchmark-Familien sortiert
- `instances/Example_Instances_FJSSP-WF/`: FJSSP-W-Instanzen
- `instances/InstanceData/FJSSP/`: CSV-Metadaten und Referenzwerte für FJSSP
- `instances/InstanceData/FJSSP-W/`: CSV-Metadaten und Referenzwerte für FJSSP-W

Die Ladefunktionen greifen auf diese feste Ordnerstruktur zu. Wenn Ordner oder Dateinamen umbenannt werden, funktionieren die Helper nicht mehr ohne Code-Anpassung.

## Wie das Environment intern funktioniert

### 1. Parsen der Benchmarks

Die Parser in [`util/benchmark_parser.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/benchmark_parser.py) lesen `.fjs`-Dateien und erzeugen Encodings.

- `BenchmarkParser`: für FJSSP
- `WorkerBenchmarkParser`: für FJSSP-W

Das Ergebnis ist ein Objekt aus [`util/encoding.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/encoding.py):

- `Encoding` für FJSSP
- `WorkerEncoding` für FJSSP-W

Diese Objekte kapseln vor allem:

- `durations()`: Dauer-Matrix bzw. Dauer-Tensor
- `job_sequence()`: Zuordnung jeder Operation zu ihrem Job
- Hilfsfunktionen, um zulässige Maschinen und Worker abzufragen

### 2. Interne Datenrepräsentation

Die repräsentierte Problemgröße ist operationenbasiert.

Wichtige Konventionen:

- Maschinen und Worker werden intern `0-basiert` gespeichert.
- Der Parser zieht die in den `.fjs`-Dateien üblichen `1-basierten` IDs auf `0-basiert` herunter.
- Nicht zulässige Kombinationen haben Dauer `0`.

Typische Formen:

- FJSSP: `durations[operation][machine]`
- FJSSP-W: `durations[operation][machine][worker]`

Die Reihenfolge aller Ergebnisvektoren richtet sich nach der fixen Operationsreihenfolge des Encodings, nicht nach einer frei wählbaren Solver-Reihenfolge.

### 3. Lösungen und Encodings

Das Projekt arbeitet an mehreren Stellen mit unterschiedlichen Sichtweisen auf eine Lösung:

- `job_sequence`: beschreibt, welche Operation im globalen Operationsvektor zu welchem Job gehört
- `sequence`: Reihenfolge, in der Jobs im Decoding verarbeitet werden
- `machines`: Maschinenzuweisung pro Operation
- `workers`: Workerzuweisung pro Operation
- `start_times`: Startzeit pro Operation

Die Hilfsfunktionen in [`util/evaluation.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/evaluation.py) können aus einer Job-Reihenfolge plus Zuweisungen konkrete Startzeiten erzeugen:

- `translate_fjssp(...)`
- `translate(...)`

Das ist relevant, wenn dein Solver nicht direkt Startzeiten optimiert, sondern ein indirektes Encoding verwendet.

### 4. Bewertung einer Lösung

Die wichtigsten Bewertungsfunktionen liegen in [`util/evaluation.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/evaluation.py):

- `makespan(...)`: Makespan für FJSSP-W
- `makespan_fjssp(...)`: Makespan für FJSSP
- `workload_balance(...)`: Lastverteilung über Worker
- `minizinc_score(...)`: Vergleich mehrerer Solver über mehrere Instanzen

Für eine gültige Bewertung müssen die Vektoren:

- dieselbe Länge haben
- zur Operationsreihenfolge des Encodings passen
- nur zulässige Maschinen-/Worker-Kombinationen enthalten

Wichtig: Die Bewertungsfunktionen berechnen Metriken, prüfen aber nicht vollständig, ob eine Lösung semantisch gültig ist. Konsistenz und Feasibility müssen vom Solver bzw. von vorgeschalteter Logik sichergestellt werden.

### 5. Laden größerer Benchmark-Mengen

[`util/load_benchmarks.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/load_benchmarks.py) bietet:

- `load_fjssp(bounds)`
- `load_fjssp_w(bounds)`

Damit können Instanzen anhand von CSV-Metadaten gefiltert werden. `bounds` ist ein Dictionary der Form:

```python
bounds = {
    "N": (20, 100),
    "m": (5, 10),
}
```

Rückgabe ist jeweils ein Dictionary:

```python
{
    "Instanzname": EncodingOderWorkerEncoding,
}
```

### 6. Baseline-Solver

[`util/greedy_solver.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/greedy_solver.py) enthält einfache Greedy-Baselines:

- `GreedyFJSSPSolver`
- `GreedyFJSSPWSolver`

Diese Solver sind eher Referenz- oder Demo-Code als Wettbewerbslösung. Sie liefern:

- eine Reihenfolge der Jobs
- Maschinenzuweisungen
- bei FJSSP-W auch Workerzuweisungen

Die eigentlichen Startzeiten werden anschließend über die Translate-Funktionen abgeleitet.

### 7. Unsicherheit und robuste Bewertung

Für den Unsicherheitsfall sind vor allem zwei Dateien wichtig:

- [`util/uncertainty.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/uncertainty.py)
- [`util/graph.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/graph.py)

`create_uncertainty_vector(...)` erzeugt Parameter pro Worker oder Maschine.

`Graph` modelliert einen konkreten Produktionsplan mit:

- Startzeiten `s`
- Endzeiten `e`
- Maschinen `m`
- Workern `w`
- Job-Sequenz `js`

Darauf können verschiedene Störungen simuliert werden:

- zufällig veränderte Bearbeitungszeiten
- Maschinen-Ausfälle
- Worker-Unverfügbarkeiten

Für den Wettbewerb ist laut [`index.md`](/home/jb_itp/projects/FJSSP-W-Competition/index.md) vor allem die Unsicherheit in den Bearbeitungszeiten relevant.

Die Hilfsfunktion `run_n_simulations(...)` führt mehrere Durchläufe aus und liefert:

- einzelne Simulationsresultate
- mittleren robusten Makespan
- Standardabweichung
- Verhältnis `R = robust_makespan / geplanter_makespan`

### 8. Visualisierung und Solver-Vergleich

[`util/visualization.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/visualization.py) stellt mehrere Plot-Funktionen bereit:

- `visualize_gaps(...)`: ECDF-artige Gap-Darstellung über viele Instanzen
- `visualize_timeline(...)`: Solver-Fortschritt über die Zeit auf einer Instanz
- `rank_plot(...)`: Rangvergleich mit `autorank`
- `show_simulation_results(...)`
- `show_simulation_comparison(...)`

Damit lassen sich sowohl Einzellösungen als auch Benchmarks über Instanzmengen vergleichen.

## Typischer Workflow

Ein üblicher Arbeitsablauf im Projekt sieht so aus:

1. Instanz laden
2. Encoding auslesen
3. Solver darauf anwenden
4. Lösung in das erwartete Vektorformat bringen
5. Makespan oder robuste Kennzahlen berechnen
6. Ergebnisse optional visualisieren

Minimales Beispiel:

```python
from util.benchmark_parser import WorkerBenchmarkParser
from util.evaluation import makespan

parser = WorkerBenchmarkParser()
encoding = parser.parse_benchmark("instances/Example_Instances_FJSSP-WF/Fattahi20.fjs")

d = encoding.durations()
js = encoding.job_sequence()

# Beispielhafte, solverseitig erzeugte Vektoren
s = [...]
m = [...]
w = [...]

value = makespan(s, m, w, d)
print(value)
```

## Was man beachten muss

### Pfade und Plattform

- Das Projekt ist kein installierbares Python-Package mit `setup.py` oder `pyproject.toml`, sondern ein Script-/Notebook-Repository.
- Imports funktionieren am einfachsten aus dem Repo-Root.
- In [`benchmark_load_example.py`](/home/jb_itp/projects/FJSSP-W-Competition/benchmark_load_example.py) wird ein Windows-Pfad mit Backslashes verwendet. Unter Linux/macOS besser normale `/` verwenden.

### Indizes

- Maschinen- und Worker-Indizes sind intern `0-basiert`.
- Wenn dein Solver mit `1-basierten` IDs arbeitet, musst du vor der Bewertung sauber umrechnen.

### Reihenfolge der Vektoren

- `start_times`, `machines` und `workers` müssen exakt zur festen Operationsreihenfolge passen.
- Die Reihenfolge ist nicht automatisch die Reihenfolge, in der dein Solver Operationen konstruiert oder verbessert.

### Zulässigkeit

- Eine Dauer `0` bedeutet: Kombination ist nicht erlaubt.
- Wenn `translate(...)` auf eine unzulässige Kombination trifft, wirft der Code eine Exception.
- Vor der finalen Auswertung sollte dein Solver also Zulässigkeit aktiv absichern.

### FJSSP-W-Instanzen sind instanzspezifisch

- Die bereitgestellten FJSSP-W-Instanzen sind konkret generierte Instanzen.
- Bestwerte aus dem Repo sind nur zu genau diesen Instanzen vergleichbar.
- Neu generierte FJSSP-W-Instanzen sind nicht direkt mit den vorhandenen Best Known Results vergleichbar.

### Evaluation ist nicht gleich Feasibility-Check

- `makespan(...)` berechnet den Zielwert aus gegebenen Vektoren.
- Daraus folgt nicht automatisch, dass technologische oder Ressourcen-Nebenbedingungen korrekt eingehalten wurden.
- Wenn du eigene Encodings nutzt, brauchst du eine saubere Decoding- oder Reparaturlogik.

### Visualisierung erwartet saubere Datenformate

- Progress-Plots erwarten Listen von Tupeln `(timestamp, value)`.
- Solver-Vergleiche über Instanzen erwarten Dictionaries mit konsistenten Instanznamen.
- Fehlende Einträge werden teils toleriert, führen aber zu `inf`-Werten in Vergleichen.

## Empfohlene Nutzung für eigene Solver

Wenn du einen eigenen Solver anbindest, ist diese Struktur sinnvoll:

1. Instanz mit `WorkerBenchmarkParser` oder `load_fjssp_w(...)` laden.
2. Aus `encoding.durations()` und `encoding.job_sequence()` die internen Solverdaten aufbauen.
3. Solver intern mit eigenem Encoding arbeiten lassen.
4. Vor der Bewertung auf das Repo-Format zurücktransformieren:
   - `s`
   - `m`
   - `w`
5. Mit `makespan(...)` und optional `workload_balance(...)` auswerten.
6. Für robuste Szenarien Endzeiten `e` berechnen und `run_n_simulations(...)` verwenden.

## Welche Dateien man zuerst lesen sollte

Für neue Teammitglieder ist diese Reihenfolge sinnvoll:

1. [`README.md`](/home/jb_itp/projects/FJSSP-W-Competition/README.md)
2. [`benchmark_load_example.py`](/home/jb_itp/projects/FJSSP-W-Competition/benchmark_load_example.py)
3. [`util/benchmark_parser.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/benchmark_parser.py)
4. [`util/encoding.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/encoding.py)
5. [`util/evaluation.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/evaluation.py)
6. [`util/graph.py`](/home/jb_itp/projects/FJSSP-W-Competition/util/graph.py)
7. die Notebooks für konkrete Beispiele

## Kurzfazit

Das Repository ist vor allem ein Benchmarking-Toolkit mit festem Datenformat. Der zentrale technische Pfad lautet:

`Instanz laden -> Encoding lesen -> Solver laufen lassen -> Lösung auf Vektoren abbilden -> evaluieren -> optional simulieren/visualisieren`

Wenn du die Indexierung, die feste Operationsreihenfolge und die Trennung zwischen Feasibility und reiner Metrikberechnung beachtest, lässt sich das Environment relativ direkt an eigene Solver anbinden.
