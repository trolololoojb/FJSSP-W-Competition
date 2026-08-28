# Wissenschaftliche Auswertung der Hyperparameteroptimierungen

> **Nachtrag vom 28. August 2026:** Die anschließend ausgeführte vollständige
> 2×2-Komponentenablation aus Surrogat-/Local-Search-Pipeline und RL ist separat in
> [`KOMPONENTEN_ABLATIONSSTUDIE.md`](KOMPONENTEN_ABLATIONSSTUDIE.md) dokumentiert.
> Sie beantwortet H1–H3 innerhalb des festgelegten Ablationsprotokolls, ist aufgrund
> überlappender Auswahl- und Auswertungsinstanzen aber kein unabhängiger Holdout-Test.

## Kurzfazit

Die Nicht-RL-HPO ist als **mehrstufiges Such- und Auswahlverfahren grundsätzlich sinnvoll aufgebaut**, ihre Verbesserung gegenüber der finalen Submission ist mit den abgelegten Aggregaten jedoch **nicht belastbar quantifizierbar**. Die als Holdout bezeichneten sechs Instanzen wurden zur Auswahl eines von fünf Finalisten benutzt und sind damit kein unabhängiger Testsatz für den anschließend benannten Sieger. Außerdem unterscheiden sich Baseline und HPO deutlich im Zeitbudget und teilweise in der stochastischen Auswertung.

Die RL-HPO ist als **kleines, reproduzierbares 3×3-Screening zweier RL-Hyperparameter** brauchbar. Sie belegt, welcher der neun getesteten RL-Parametersätze unter genau diesem Versuchsaufbau am besten abschnitt. Sie belegt aber weder eine Verbesserung gegenüber der finalen Submission noch einen isolierten Nutzen von RL: Ein identisch behandelter Nicht-RL-Kontrollarm fehlt, und der Gewinner wurde auf denselben acht Instanzen und Läufen ausgewählt, auf denen sein Ergebnis berichtet wird.

Damit lautet das wissenschaftlich vertretbare Gesamtergebnis:

- **Belegt:** Beide HPOs wurden erfolgreich durchgeführt; ihre jeweiligen Rangfolgen und Gewinner sind durch die vorhandenen Artefakte nachvollziehbar.
- **Explorativ:** Die Suche hat Konfigurationen gefunden, die innerhalb ihrer jeweiligen Auswahlphase besser gerankt wurden als andere Kandidaten.
- **Nicht belegt:** Eine bestimmte prozentuale Verbesserung gegenüber der finalen Submission, statistische Signifikanz, Generalisierung auf ungesehene Instanzen oder ein kausaler Vorteil des RL-Mutationscontrollers.

## Datengrundlage und Evidenzklassen

Diese Auswertung verwendet ausschließlich bereits vorhandene Dateien unter `results/hpo_scenario2`, `results/hpo_rl_scenario2` und `results/scenario2_submission`. Es wurden keine Solver-, HPO-, Test- oder Analyseskripte ausgeführt und keine Ergebnisdateien neu erzeugt oder verändert.

Als praktische Baseline gilt die gespeicherte finale Submission unter `results/scenario2_submission`. Ihr Manifest weist 30 Instanzen, zehn Läufe je Instanz und damit 300 erfolgreiche Läufe aus. Die gespeicherten Hyperparameter beschreiben C0 ohne RL: unter anderem Population 200, Offspring-Menge 1.000, Elitismus 0,10, zehn interne Simulationen und aktivierte Local Search. Maßgeblich ist dieser gespeicherte Zustand, nicht ein möglicherweise inzwischen abweichender Stand des Quellcodes.

Die Aussagen werden wie folgt klassifiziert:

- **Belegt:** unmittelbar durch vorhandene Spezifikationen, Manifeste oder Zusammenfassungen gestützt.
- **Explorativ:** aus für Auswahl oder Tuning verwendeten Daten abgeleitet und daher wahrscheinlich optimistisch.
- **Nicht belegt:** würde einen unabhängigen oder fair budgetierten Kontrollversuch beziehungsweise eine noch nicht durchgeführte statistische Auswertung voraussetzen.

## Nicht-RL-HPO

### Versuchsaufbau

| Phase | Konfigurationen | Instanzen | Läufe je Konfiguration und Instanz | maximales FE-Budget | finale Simulationen | Datenrolle |
|---|---:|---:|---:|---:|---:|---|
| Screening | 48 | 8 | 2 | 250.000 | 20 | Training |
| TPE | 72 | 12 | 3 | 750.000 | 20 | Training |
| Race 1 | 15 | 18 | 4 | 1.500.000 | 50 | Training |
| Race 2 | 8 | 24 | 6 | 3.000.000 | 50 | Training + Validierung |
| Final | 5 | 6 | 10 | 5.000.000 | 50 | als Holdout bezeichnet |

Der Aufbau verbindet ein breites Screening, eine TPE-gestützte Suche und sukzessiv größere Racing-Budgets. Das ist recheneffizient und besser begründet als die Auswahl aus einem einzigen kleinen Lauf. Die Instanzen werden stufenweise erweitert, die Anzahl der Wiederholungen steigt, und die Finalphase verwendet sechs zuvor zurückgehaltene Instanzen.

Die Suchparameter umfassen nicht nur klassische GA-Größen wie Population, Offspring-Verhältnis, Elitismus und Restart-Intervall, sondern auch stochastische Auswertung, Surrogatmodell und Local Search. Damit wird ein breiter und praktisch relevanter Konfigurationsraum untersucht. Zugleich erhöht diese Breite die Zahl impliziter Vergleiche und damit das Risiko, zufällige gute Resultate auszuwählen.

### Nachvollziehbares Ergebnis

**Belegt:** Alle fünf Finalisten besitzen laut finaler Zusammenfassung 60 erfolgreiche von 60 erwarteten Läufen. Rang 1 ist
`final_rank03_race2_rank04_race1_rank08_TPE0071_8834ed6804` mit einem gespeicherten Score von 1,0173747. Seine Zusammenfassung weist einen mittleren robusten Makespan von 3.001,78, einen Median von 2.641,92, eine mittlere Laufzeit von 7.203,71 Sekunden und im Mittel 429.170 als wirksam ausgewiesene Function Evaluations aus. Diese absoluten Makespan-Aggregate mischen sechs unterschiedlich skalierte Instanzen und eignen sich deshalb nicht allein als globales Gütemaß.

**Belegt:** Die C0-Repo-Baseline wurde im Screening mit 16 von 16 erfolgreichen Läufen bewertet, dort aber nur auf acht Trainingsinstanzen, mit 250.000 maximalen Function Evaluations und 20 finalen Simulationen. Sie belegte Rang 43 von 48. Der Screening-Sieger hatte einen Score von 0,9577339, C0 einen Score von 1,0008995. Die Differenz der Scores darf nicht als 4,3-prozentige Verbesserung der finalen Methode ausgegeben werden: Sie gehört zu einer frühen Auswahlphase, basiert auf nur zwei Läufen je Instanz und derselben Stichprobe, die zur Auswahl des Siegers genutzt wurde.

**Explorativ:** Die starke Platzierung mehrerer anderer Konfigurationen gegenüber C0 im Screening spricht dafür, dass die ursprüngliche Parametrisierung in diesem kleinen Suchbudget nicht optimal war. Sie ist ein Suchsignal, kein unabhängiger Leistungsnachweis.

### Vergleich mit der finalen Submission

Ein direkter Ergebnisvergleich wäre prinzipiell auf den sechs gemeinsamen Instanzen und den zehn gemeinsamen Run-/Seed-Positionen möglich. Die vorhandenen allgemeinen Zusammenfassungen enthalten diesen gepaarten, instanznormalisierten Vergleich jedoch nicht. Er wird hier nicht nachträglich aus inkompatiblen Gesamtmittelwerten geschätzt.

Zusätzlich bestehen folgende Unterschiede:

| Merkmal | Finale Submission (Baseline) | Nicht-RL-HPO, Finalphase |
|---|---:|---:|
| Instanzen insgesamt | 30 | 6 |
| Läufe je Instanz | 10 | 10 |
| maximales FE-Budget | 5.000.000 | 5.000.000 |
| Zeitlimit | 129.600 s | 7.200 s |
| interne Simulationen | 10 | 12 beim Sieger |
| finale Simulationen | 50 | 50 |

Das gleiche nominelle FE-Limit stellt wegen des stark verschiedenen Zeitlimits kein gleiches tatsächlich verfügbares Rechenbudget sicher. Der HPO-Sieger lief im Mittel ungefähr bis an das zweistündige Limit und meldete im Mittel deutlich weniger als fünf Millionen wirksame Function Evaluations. Die Baseline besitzt je nach Instanz wesentlich längere Laufzeiten. Ein niedrigerer Makespan des HPO-Siegers wäre trotz des kleineren Zeitbudgets praktisch interessant; ohne den gepaarten Vergleich darf seine Größe aber nicht behauptet werden.

### Wissenschaftliches Urteil

**Stärken:** dokumentierter mehrstufiger Suchplan, zunehmende Budgets und Wiederholungen, separate Instanzgruppen, vollständige Finalresultate, gespeicherte Konfigurationen und Manifeste sowie eine robuste, instanzbezogene Referenzbildung in den Ergebnisartefakten.

**Schwächen:** C0 wurde nicht als Kontrollarm durch alle Phasen bis in die Finalphase mitgeführt; der Final-„Holdout“ wählt noch zwischen fünf Konfigurationen und ist damit Validierungs- statt unabhängiger Testdatensatz; phasenübergreifende Scores besitzen wechselnde Referenzen und Budgets; es fehlen Konfidenzintervalle, vorab definierte Hauptvergleiche und eine Korrektur für die Vielzahl untersuchter Konfigurationen. Zudem macht das Zeitlimit von rund zwei Stunden die Überschrift „5.000.000 Function Evaluations“ allein zu keiner hinreichenden Budgetbeschreibung.

**Urteil:** Die HPO war für das **Finden eines vielversprechenden Parametersatzes methodisch angemessen**, aber nicht ausreichend für den wissenschaftlichen Nachweis seiner Überlegenheit gegenüber der finalen Submission. Die Verbesserung zur Baseline ist mit den vorhandenen Aggregaten **nicht belegt und wird deshalb nicht als Prozentwert angegeben**.

## RL-HPO

### Versuchsaufbau

Die RL-HPO übernimmt den Rang-1-Sieger der Nicht-RL-HPO und aktiviert den RL-Mutationscontroller. Variiert werden ausschließlich:

- Learning Rate: 0,0001; 0,0003; 0,001
- Update-Intervall: 8; 16; 32

Damit entstehen neun vollständig gekreuzte Konfigurationen. Weitere RL-Parameter sind fixiert, unter anderem \(\gamma=0{,}99\), \(\lambda=0{,}95\), Clip-Epsilon 0,2, Hidden Size 32 und zehn Warm-up-Generationen. Jede Konfiguration wurde auf acht Trainingsinstanzen mit zwei Läufen bewertet. Das Manifest nennt 5.000.000 maximale Function Evaluations, 7.200 Sekunden Zeitlimit, zwölf interne und 20 finale Simulationen.

### Nachvollziehbares Ergebnis

**Belegt:** Alle neun Konfigurationen besitzen 16 von 16 erwarteten erfolgreichen Läufen. Rang 1 ist `rl_lr1e-04_u008` mit Learning Rate 0,0001 und Update-Intervall 8. Der gespeicherte Score beträgt 1,0198186; mittlerer Makespan 3.276,10, Median 1.620,45 und mittlere Laufzeit 7.201,31 Sekunden.

**Explorativ:** Dieser Parametersatz ist innerhalb des untersuchten 3×3-Rasters der beste beobachtete RL-Kandidat. Der Abstand zum zweiten Rang ist ein Unterschied des HPO-internen Scores, kein Konfidenzintervall und kein unabhängiger Generalisierungsnachweis. Bei nur zwei Läufen je Instanz kann stochastische Streuung die Rangfolge merklich beeinflussen.

### Warum kein RL-Effekt nachgewiesen ist

In der RL-HPO fehlt die über dieselben acht Instanzen, Seeds, Zeitlimits, internen und finalen Simulationen erneut ausgeführte Quellkonfiguration mit deaktiviertem RL. Deshalb sind gleichzeitig mehrere Unterschiede wirksam:

1. Wechsel von der gespeicherten Submission-Baseline auf den bereits HPO-optimierten Nicht-RL-Sieger,
2. Aktivierung von RL,
3. Auswahl von Learning Rate und Update-Intervall aus neun Kandidaten.

Ein Vergleich des RL-Gewinners mit der finalen Submission würde somit den gesamten Konfigurationswechsel messen, nicht den isolierten Beitrag von RL. Außerdem sind die acht Instanzen genau die Daten, auf denen der RL-Gewinner ausgewählt wurde. Jede dort beobachtete Verbesserung wäre eine optimistische In-Sample-Schätzung.

Auch die Resultate der alten C0-gegen-C0-mit-RL-Untersuchung unter `results/c0_rl_scenario2_old` beantworten diese Frage nicht für den neuen HPO-Sieger: Sie betreffen eine andere Basiskonfiguration und andere RL-Hyperparameter. Sie können höchstens als separate Ablationsstudie zur alten C0-Konfiguration behandelt werden.

### Wissenschaftliches Urteil

**Stärken:** kleiner und klar begrenzter Suchraum, vollständiges faktorielles Raster, fixe übrige Parameter, einheitliche Instanzen und Budgets innerhalb der neun Kandidaten, vollständige Läufe und gespeicherte Manifeste.

**Schwächen:** nur zwei Wiederholungen je Instanz; ausschließlich Trainingsinstanzen; kein Nicht-RL-Kontrollarm; keine unabhängige Evaluation des gewählten RL-Siegers; keine Unsicherheitsintervalle; nur 20 finale Simulationen gegenüber 50 in Baseline und Nicht-RL-Finalphase. Die HPO-interne Referenz wird aus den getesteten RL-Kandidaten gebildet und ist daher keine Baseline.

**Urteil:** Die RL-HPO ist als **exploratives Parameterscreening sauber nachvollziehbar**, aber als Wirksamkeitsnachweis für RL unzureichend. Eine Verbesserung gegenüber der finalen Submission oder gegenüber derselben optimierten Konfiguration ohne RL ist **nicht belegt**.

## Übergreifende Validitätsprüfung

| Kriterium | Nicht-RL-HPO | RL-HPO |
|---|---|---|
| Suchraum dokumentiert | ja | ja |
| Seeds/Run-Positionen gespeichert | ja | ja |
| Kandidaten innerhalb einer Phase vergleichbar | überwiegend | ja |
| vollständige Gewinnerdaten | ja, 60/60 final | ja, 16/16 |
| unabhängiger Testsatz nach Auswahl | nein | nein |
| identisch budgetierte Submission-Baseline | nein | nein |
| isolierter RL-Kontrollarm | nicht relevant | nein |
| Konfidenzintervalle/Tests vorhanden | nein | nein |
| Verbesserung zur Submission belastbar quantifizierbar | nein | nein |

Die Wiederverwendung gleicher Instanz-/Run-Seeds innerhalb einer Phase ist für gepaarte Vergleiche grundsätzlich vorteilhaft, sofern die resultierenden Differenzen auf Ebene der unabhängigen Instanzen analysiert werden. Einzelne Läufe derselben Instanz dürfen nicht fälschlich als vollständig unabhängige Stichproben behandelt werden.

Die HPO-Scores sind gute Auswahlkriterien innerhalb ihrer jeweiligen Phase. Sie sind jedoch keine allgemein vergleichbaren Effektgrößen: Instanzmengen, Referenzwerte, Wiederholungszahlen, Simulationszahlen und Budgets wechseln zwischen den Phasen. Insbesondere kann ein Score über 1 nicht pauschal als Verschlechterung gegenüber der finalen Submission gelesen werden.

## Erforderlicher Bestätigungstest

Für einen belastbaren wissenschaftlichen Nachweis sollte vor Beginn ein separates Protokoll festgelegt werden.

### Varianten

1. gespeicherte finale Submission/C0 ohne RL,
2. Gewinner der Nicht-RL-HPO ohne RL,
3. Gewinner der RL-HPO mit Learning Rate 0,0001 und Update-Intervall 8.

Variante 2 und 3 müssen bis auf die Aktivierung und festgelegten Parameter des RL-Controllers identisch sein. So misst Vergleich 3 gegen 2 den RL-Beitrag, während Vergleich 2 gegen 1 den Beitrag der allgemeinen HPO misst.

### Versuchsbedingungen

- bislang ungesehene Instanzen verwenden; keine der neuen Resultate erneut zur Parameterauswahl nutzen,
- mindestens zehn unabhängige Läufe je Variante und Instanz,
- identische Run-Seeds und identische Unsicherheitsrealisationen über alle drei Varianten,
- identisches Zeitlimit **und** identisches maximales FE-Budget,
- identische Anzahl interner und finaler Simulationen,
- identische Hardware- und Parallelisierungseinstellungen,
- Abbrüche und Fehlläufe vorab definiert behandeln und vollständig berichten.

Falls keine neuen Instanzen beschafft werden können, ist eine äußere, verschachtelte Resampling- oder Cross-Validation-Struktur die zweitbeste Lösung. Dabei muss die gesamte HPO in jedem äußeren Fold erneut stattfinden. Eine bloße erneute Auswertung des bereits ausgewählten Siegers auf den sechs Nicht-RL-Finalinstanzen beziehungsweise den acht RL-Tuninginstanzen beseitigt den Auswahlbias nicht.

### Endpunkte und Statistik

Primärer Endpunkt ist je Instanz das Verhältnis

\[
r_i = \frac{\operatorname{MedianMakespan}_{\text{Kandidat},i}}
           {\operatorname{MedianMakespan}_{\text{Baseline},i}}.
\]

Ein Wert unter 1 bedeutet eine Verbesserung. Als Gesamteffekt sollte der Mittelwert der Log-Verhältnisse beziehungsweise dessen rücktransformiertes geometrisches Mittel mit 95-%-Konfidenzintervall berichtet werden. Die Instanz, nicht der einzelne Run, ist dabei die primäre unabhängige Einheit. Ergänzend sind Gewinn-/Gleichstand-/Verlustanzahl je Instanz und alle instanzweisen Effekte anzugeben.

Sekundäre Endpunkte sind Laufzeit, tatsächlich verbrauchte Function Evaluations, Streuung des robusten Makespans, Erfolgs-/Abbruchrate und gegebenenfalls die Robustheitskennzahl \(R\). Für die zwei vorab festgelegten Hauptvergleiche — Nicht-RL-HPO gegen Baseline und RL gegen denselben Nicht-RL-Sieger — sollten gepaarte Bootstrap-Konfidenzintervalle verwendet werden. Formale Tests sind nur ergänzend sinnvoll; bei zwei Tests ist die Fehlerkontrolle vorab festzulegen, beispielsweise Holm-Korrektur. Effektgröße und Konfidenzintervall bleiben wichtiger als ein isolierter p-Wert.

## Methodische Einordnung und Primärquellen

Die vorsichtige Interpretation folgt dem allgemeinen Problem, dass die Optimierung eines endlichen, verrauschten Auswahlkriteriums dieses Kriterium selbst überanpassen kann. Cawley und Talbot zeigen, dass der daraus entstehende Selektionsbias in derselben Größenordnung wie typische Unterschiede zwischen Algorithmen liegen kann und empfehlen, Modell-/Hyperparameterauswahl als Teil des Verfahrens innerhalb einer äußeren Evaluation zu behandeln ([JMLR, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). Reunanen demonstriert ebenfalls, dass während der Suche verwendete Leistungsdaten keine verlässliche Grundlage für den abschließenden Vergleich der Auswahlverfahren sind und ein unabhängiger Testsatz benötigt wird ([JMLR, 2003](https://www.jmlr.org/papers/v3/reunanen03a.html)).

Für breite Suchräume ist Random Search eine anerkannte Baseline; Bergstra und Bengio zeigen, warum zufällige Suche bei vielen nur teilweise relevanten Hyperparametern effizienter als vollständige Gitter sein kann ([JMLR, 2012](https://www.jmlr.org/papers/v13/bergstra12a.html)). Das stützt den Screening-Anteil der Nicht-RL-HPO, ersetzt aber keinen unabhängigen Bestätigungstest.

Für Vergleiche mehrerer Algorithmen auf mehreren Datensätzen diskutiert Demšar nichtparametrische, datensatzweise Vergleiche und warnt vor ungeeigneter Aggregation abhängiger Wiederholungen ([JMLR, 2006](https://www.jmlr.org/papers/v7/demsar06a.html)). Bei mehreren paarweisen Vergleichen ist eine vorab definierte Fehlerkontrolle erforderlich; Holms sequenziell verwerfendes Verfahren kontrolliert die familienweise Fehlerrate ([Scandinavian Journal of Statistics, 1979](https://www.jstor.org/stable/4615733)).

## Schlussfolgerung

Die vorhandenen Resultate rechtfertigen die Auswahl je eines **vielversprechenden Nicht-RL- und RL-Parametersatzes**. Sie rechtfertigen derzeit keine belastbare Aussage der Form „HPO verbessert die Baseline um x %“ und keine Aussage „RL verbessert den optimierten Solver“. Der Grund ist nicht ein offensichtlicher Ausführungsfehler, sondern das Evaluationsdesign nach der Auswahl: Es fehlt jeweils ein unabhängiger, fair budgetierter und gepaarter Kontrollvergleich.

Für eine wissenschaftliche Arbeit sollte daher formuliert werden:

> Die Hyperparameteroptimierung identifizierte unter den untersuchten Kandidaten die angegebenen Siegerkonfigurationen. Aufgrund unterschiedlicher Budgets und der Nutzung der berichteten Instanzen zur Konfigurationsauswahl kann aus den vorliegenden Daten noch keine unverzerrte Verbesserung gegenüber der finalen Submission beziehungsweise kein isolierter RL-Effekt abgeleitet werden. Diese Aussagen erfordern den beschriebenen unabhängigen Bestätigungstest.
