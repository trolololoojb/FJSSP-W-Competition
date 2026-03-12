# Einfacher Solver unter `Solver/`

Dieser Ordner enthaelt einen bewusst einfachen Beispiel-Solver fuer **FJSSP-W**. Er ist nicht auf beste Ergebnisse optimiert, sondern darauf, die **Anforderungen an eine zulaessige Loesung** nachvollziehbar zu machen.

## Was der Solver macht

Datei: `simple_worker_solver.py`

Der Solver arbeitet als einfache Konstruktionsheuristik:

1. Fuer jeden Job ist immer nur die **naechste noch ungeplante Operation** freigegeben.
2. Fuer jede freigegebene Operation prueft der Solver alle zulaessigen **Maschine/Worker-Kombinationen**.
3. Er berechnet, wann diese Kombination fruehestens starten kann:
   - nach Ende der vorherigen Operation desselben Jobs
   - wenn die Maschine frei ist
   - wenn der Worker frei ist
4. Er waehlt global die Operation mit der **fruehesten Fertigstellung**.
5. Das wird wiederholt, bis alle Operationen eingeplant sind.

## Welche Anforderungen eine gueltige Loesung erfuellen muss

Fuer jede Operation brauchst du am Ende:

- eine **Startzeit**
- eine **Maschine**
- einen **Worker**

Die Loesung ist nur gueltig, wenn:

- jede Operation genau einmal geplant ist
- ein Job seine Reihenfolge einhaelt
- auf derselben Maschine keine zwei Operationen gleichzeitig laufen
- derselbe Worker nicht zwei Operationen gleichzeitig bearbeitet
- die gewaehlte Maschine/Worker-Kombination in der Instanz erlaubt ist

Die Zielfunktion ist hier der **Makespan**, also der Endzeitpunkt der letzten fertigen Operation.

## Eingabeformat

Der Solver nutzt den vorhandenen `WorkerBenchmarkParser` aus `util/benchmark_parser.py`.

Die Beispielinstanzen in `instances/Example_Instances_FJSSP-WF/*.fjs` enthalten:

- Anzahl Jobs
- Anzahl Maschinen
- Anzahl Worker
- pro Job mehrere Operationen
- pro Operation mehrere erlaubte Maschinen
- pro Maschine mehrere erlaubte Worker mit Bearbeitungsdauer

## Ausfuehren

Aus dem Projektwurzelverzeichnis:

```bash
python Solver/simple_worker_solver.py
```

Mit expliziter Instanz:

```bash
python Solver/simple_worker_solver.py --instance instances/Example_Instances_FJSSP-WF/Fattahi17.fjs
```

Mit detaillierter Ausgabe aller geplanten Operationen:

```bash
python Solver/simple_worker_solver.py --show-ops
```

Falls du die virtuelle Umgebung im Repo nutzen willst:

```bash
.venv/bin/python Solver/simple_worker_solver.py --show-ops
```

## Was du in der Ausgabe siehst

- `Job-Reihenfolge der Konstruktion`: in welcher Reihenfolge Jobs ausgewaehlt wurden
- `Maschinenzuweisung pro Operation`: Maschine fuer jede Operation im festen Operationsindex
- `Workerzuweisung pro Operation`: Worker fuer jede Operation
- `Startzeiten pro Operation`: Startzeit je Operation
- `Makespan`: Qualitaet der gebauten Loesung

## Wie du den Solver spaeter erweitern kannst

Naechste sinnvolle Schritte:

- statt "earliest finish" andere Prioritaetsregeln testen
- mehrere Kandidaten zufaellig variieren
- Local Search auf den erzeugten Plaenen anwenden
- FJSSP ohne Worker als abgespeckte Variante ableiten
