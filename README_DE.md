# FJSSP- und FJSSP-W-Benchmarking-Umgebung
Die bereitgestellte Umgebung kann verwendet werden, um verschiedene Algorithmen für das Flexible Job Shop Scheduling Problem (FJSSP) und das FJSSP mit Worker-Flexibilität (FJSSP-W) zu testen und zu vergleichen. Die bereitgestellte Benchmark-Instanzsammlung befindet sich im Unterverzeichnis `instances`, einschließlich einiger Beispielinstanzen für das FJSSP-W. Ebenfalls im Verzeichnis `InstanceData` enthalten sind bekannte beste Ergebnisse und untere Schranken für die verschiedenen Probleme und ihre Instanzen.
`util` enthält die bereitgestellten Evaluierungs- und Vergleichsfunktionen der Benchmarking-Umgebung, einschließlich des Algorithmus, der verwendet wird, um FJSSP-W-Probleminstanzen aus den bekannten FJSSP-Instanzen zu erzeugen.

## FJSSP-W-Benchmarking-Instanzen

Beachten Sie, dass zwar ein Übersetzungsalgorithmus zwischen FJSSP- und FJSSP-W-Instanzen bereitgestellt wird, die FJSSP-W-Instanzen jedoch zufällig (auf Basis der jeweiligen FJSSP-Instanz) erzeugt werden. Die bereitgestellten Best Known Results für das FJSSP-W sind spezifisch für die bereitgestellten FJSSP-W-Instanzen und können nicht für den Vergleich mit neu erzeugten Instanzen verwendet werden.

## Struktur
Das Repository ist in die Unterbereiche [util](https://github.com/jrc-rodec/FJSSP-W-Benchmarking/tree/main/util) und [instances](https://github.com/jrc-rodec/FJSSP-W-Benchmarking/tree/main/instances) sowie in die Beispieldateien im Wurzelverzeichnis unterteilt.

### Util
Der Bereich `util` enthält die bereitgestellten APIs für Vergleich und Evaluierung.
Die bereitgestellten APIs umfassen einen Algorithmus zur Übersetzung von FJSSP- in FJSSP-W-Instanzen, Methoden zur Visualisierung und zum Vergleich von Ergebnissen, Evaluierungsmetriken sowie Hilfsfunktionen zum Laden und Parsen von Benchmark-Instanzen für sowohl FJSSP als auch FJSSP-W in Python-Objekte.
Zusätzlich ist als Basislinie für Vergleiche ein Greedy-Solver enthalten, der die gegebenen Benchmark-Probleme lösen kann.

### Instanzen
Der Bereich `instances` enthält die Probleminstanzen, einschließlich von Beispiel-FJSSP-W-Instanzen, die mit dem im Bereich `util` bereitgestellten Übersetzungsalgorithmus erzeugt wurden.
Zusätzlich enthält der Unterordner [InstanceData](https://github.com/jrc-rodec/FJSSP-W-Benchmarking/tree/main/instances/InstanceData) Daten über die Benchmark-Instanzen und ihre Eigenschaften sowie eine Sammlung bekannter bester Ergebnisse (untere Schranke (LB) und obere Schranke (UB)). Die obere Schranke repräsentiert das beste gefundene zulässige Ergebnis. Wenn untere und obere Schranke einer Instanz gleich sind, kann davon ausgegangen werden, dass es sich um das optimale Ergebnis handelt.

### Beispiele
Die Beispieldateien demonstrieren verschiedene Aspekte der Benchmarking-Umgebung, um das Verständnis der bereitgestellten APIs zu unterstützen.

## Benchmark-Auswahl
Die Benchmarking-Umgebung bietet eine Filteroption, mit der Benchmark-Instanzen entsprechend den gewünschten Benchmark-Eigenschaften ausgewählt werden können. 

FJSSP-Übersicht           |  FJSSP-W-Übersicht
:-------------------------:|:-------------------------:
![FJSSP Instances](readme_resources/fjssp_overview.png) | ![FJSSP-W Instances](readme_resources/fjssp_w_overview.png)

Die oben dargestellten Abbildungen zeigen die Verteilung der Benchmark-Instanzen in Bezug auf einige der filterbaren Problemeigenschaften für die bereitgestellten FJSSP- und FJSSP-W-Instanzen. Es gibt mehrere Cluster von Instanzen, die sich in diesen Eigenschaften sehr ähnlich sind. Um effizient zu testen, wie gut ein Algorithmus auf das allgemeinere Problem anwendbar ist, kann es wünschenswert sein, zunächst nur Repräsentanten der Cluster auszuwählen, um Rechenzeit zu sparen und die repräsentativste mögliche Teilmenge von Instanzen zu erzeugen, damit potenzielle Verzerrungen in der Ergebnisanalyse vermieden werden.
Die filterbaren Merkmale umfassen die Anzahl der Operationen $\texttt{N}$, die Anzahl der Maschinen, die Anzahl der Worker, die Flexibilität $\beta$, die Dauer-Varianz $dv$, die Anzahl der Aufträge $n$ und verschiedene weitere Merkmale. Jede Eigenschaft kann optional mit einer vordefinierten unteren und oberen Grenze für Problemfälle gefiltert werden, die in die Experimente aufgenommen werden sollen. Diese Option ist sowohl für die bereitgestellten FJSSP- als auch für die FJSSP-W-Instanzen verfügbar.

## Benchmark-Vorbereitung
Basierend auf den bereitgestellten Filtern (falls vorhanden) können die angeforderten Benchmarks mit eingebauten Funktionen geladen werden. Der Ladeprozess gibt alle angeforderten Instanzen zurück und stellt mehrere APIs bereit, um ihre Daten auszulesen. Dazu gehören die Job-Sequenz, die zeigt, wie viele Operationen zu jedem Job gehören, einschließlich ihrer festen Position für die Zuweisungsvektoren, die Dauermatrix $T$, die Anzahl der Operationen $\texttt{N}$, Maschinen $m$, Worker $w$ und mehrere weitere Eigenschaften.

## Benchmark-Experimente
Die bereitgestellten besten Ergebnisse sowohl für das FJSSP als auch für das FJSSP-W wurden aus einer Kombination von State-of-the-Art-Solvern und einem GA [Hutter2024](https://doi.org/10.1109/CEC60901.2024.10611934) gewonnen. Die Experimente zur Ermittlung der Ergebnisse wurden auf demselben Testrechner verarbeitet (Intel Core i7-6700 (3.40GHz) CPU mit 16 GB RAM, Windows 10) mit einem Zeitlimit von 20 Minuten. Beachten Sie, dass für das FJSSP-W nur die besten bekannten Ergebnisse für die bereitgestellten Instanzen angegeben sind. Wenn andere FJSSP-W-Instanzen erzeugt werden, sind sie nicht mit den bereitgestellten Best Known Results vergleichbar. Die von der Benchmarking-Umgebung bereitgestellten Visualisierungsmethoden können verwendet werden, um beliebige Metriken zu visualisieren, wobei für den Vergleich mit den bereitgestellten Best Known Results der Makespan als Zielfunktion erforderlich ist. Die von den verwendeten Solvern gefundenen besten unteren Schranken werden ebenfalls bereitgestellt und können verglichen werden. Für den Vergleich spezifischer Probleminstanzen kann ein Fortschrittsplot erstellt werden. Dies erfordert eine Aufzeichnung der gewünschten Metriken über die Zeit in Form einer Liste. Die Listenelemente sollten Tupel enthalten, wobei der erste Eintrag der Zeitstempel der Aufzeichnung und der zweite der aufgezeichnete Wert ist. Diese Daten können vom Visualizer verwendet werden, um einen vergleichenden Plot zu erzeugen.

Beachten Sie, dass für Solver mit stochastischem Charakter wie Metaheuristiken mehrere Experimente pro Probleminstanz durchgeführt werden sollten, um belastbare Ergebnisse zu erhalten. Die minimal empfohlene Anzahl unabhängiger Algorithmus-Wiederholungen beträgt 20. Dies stellt eine leichte Reduktion gegenüber der üblicherweise empfohlenen Anzahl von etwa 30 Wiederholungen für statistisch relevante Ergebnisse [Vecek2017](doi.org/10.1016/j.asoc.2017.01.011) dar und ist auf den hohen Rechenaufwand für die Solver-Ausführung auf den einzelnen Instanzen zurückzuführen.
Zu diesem Zweck erscheint es sinnvoll, die Dauer der Experimente zu verkürzen, indem die einzelnen Läufe parallel durchgeführt werden.
Der Prozess der Parallelisierung wird jedoch dem Benutzer der Benchmarking-Umgebung überlassen und in den untenstehenden Nutzungsbeispielen nicht gezeigt, da parallelisierte Ansätze nicht automatisch nachverfolgt werden können. Dennoch können die besten Ergebnisse und die durchschnittlichen Optimierungsergebnisse in das erforderliche Format gebracht und mit den bereitgestellten Visualisierungstools verwendet werden.

## Leistungsbewertung
Die Benchmarking-Umgebung bietet eine Evaluierungsfunktion für den Makespan, um sicherzustellen, dass die finale Lösung korrekt bewertet wird. Als Eingabe für die Evaluierung werden die Startzeiten, die Maschinenzuweisungen und die Workerzuweisungen (im Fall des FJSSP-W) in der festen Reihenfolge benötigt, die von der Benchmarking-Umgebung vorgegeben wird. Eine Übersetzung zwischen dem in [Hutter2024](https://doi.org/10.1109/CEC60901.2024.10611934) verwendeten Encoding für sowohl das FJSSP als auch das FJSSP-W ist verfügbar.
Für Vergleiche mit anderen Solvern bietet die Benchmarking-Umgebung verschiedene Visualisierungen der über alle verwendeten Benchmark-Instanzen zusammengefassten Ergebnisse. Die Daten der evaluierten Solver sind enthalten und Best Known Results können im Laufe der Zeit aktualisiert werden. Zusätzlich bietet die Benchmarking-Umgebung eine API zur Berechnung des MiniZinc-Scores für Vergleichszwecke. Eine Demonstration findet sich in den unten bereitgestellten Beispielen.

Ergebnisübersicht           |  Vergrößerte Ergebnisübersicht
:-------------------------:|:-------------------------:
![Reuls Overview](readme_resources/results_gap_example_formatted.png) | ![Limited Result Overview](readme_resources/results_gap_limit_example_formatted.png)

Die Beispiel-Ergebnisplots sind in der obigen Abbildung dargestellt. Für jeden verglichenen Scheduling-Algorithmus visualisiert die Darstellung den Anteil der Probleminstanzen, die bis zu einer gegebenen Abweichung von den Best Known Results auf solchen Instanzen gelöst werden können. Dazu wird für jede Instanz die relative Abweichung oder Lücke $\delta_{rel}$ der final besten Lösung eines Solvers $C_{fb}=C(\mathbf{y}_{fb})$ vom Best Known Result $C_{best}$ auf einer bestimmten Instanz berechnet

$\delta_{rel} = \frac{C_{fb} - C_{best}}{C_{best}}$

Alle Instanzen, die innerhalb der Abweichung $\delta_{rel}$ gelöst werden konnten, werden dann gezählt.
Schließlich wird das Verhältnis der innerhalb der Lücken gelösten Instanzen gegen die jeweiligen Werte von $\delta_{rel}$ aufgetragen.

Für mehrere Solver gilt die Algorithmusleistung als besser, je näher die Kurve entlang der vertikalen Achse verläuft. In solchen Fällen werden größere Anteile der Instanzen mit vergleichsweise kleineren Abweichungen von der Best Known Solution gelöst.
Während die linke Abbildung einen allgemeinen Überblick über die Leistungsunterschiede zwischen vier beispielhaften Solvern zeigt, zoomt die rechte Abbildung in den unteren linken Teil der Ergebnisse in der ersten Abbildung hinein, um subtilere Leistungsunterschiede zwischen den Solvern hervorzuheben. Man beobachtet, dass Solver A die beste Leistung zeigt, da er bei ungefähr $55\%$ der Instanzen die Best Known Solution realisiert. Im Gegensatz dazu zeigt Solver C die schlechteste Leistung.
Beachten Sie, dass diese Darstellung nicht geeignet ist, um Schlussfolgerungen über die Leistung auf einzelnen Instanzen zu ziehen, sondern sich ausschließlich auf den Anteil der Instanzen konzentriert, die mit einer gegebenen Qualität gelöst werden konnten.

Wie in der Abbildung zu sehen ist, kann es vorkommen, dass einige Leistungslinien $\delta_{rel} = n1.0$ nicht erreichen. Dies passiert, wenn nicht alle Solver für alle enthaltenen Instanzen zulässige Lösungen finden konnten. Der Visualizer ist so implementiert, dass er fehlende Daten für den Fall im Beispiel verarbeiten kann.

### MiniZinc-Score und Test auf statistische Signifikanz
Zusätzlich zu den oben dargestellten Leistungsplots stellt die Benchmarking-Umgebung weitere Leistungsindikatoren bereit, um die getesteten Solver über alle Instanzen beider Problemtypen (FJSSP oder FJSSP-W) hinweg zu vergleichen. Einerseits bietet die Suite die Möglichkeit, einen Gesamtscore für die Ergebnisse aller Benchmark-Instanzen zu erzeugen, dem Beispiel des jährlichen MiniZinc-Wettbewerbs [MiniZinc](https://challenge.minizinc.org/) für Constraint-Programming-Solver [Stuckey2014](https://doi.org/10.1609/aimag.v35i2.2539) folgend. Andererseits werden die Rangfolgen der benchmarketen Solver berechnet und mit einem Friedman-Test als Omnibus-Test und gegebenenfalls dem Nemenyi-Test für paarweise Post-hoc-Untersuchungen [Hollander2013](doi.org/10.1002/9781119196037) auf statistisch signifikante Unterschiede geprüft.

| FJSSP-W-Solver   | Solver A | Solver B | Solver C | Solver D |
|-------------------------|----------|----------|----------|----------|
| MiniZinc-Score | 866.0    | 590.0    | 460.0    | 496.0    |

Der sogenannte MiniZinc-Score ist ein zusammenfassendes Maß für die gegenseitige Leistung der verglichenen Solver über alle ausgewählten Instanzen eines Problemtyps. Der Score wird berechnet, indem die Ergebnisse jedes Solvers relativ zu den Ergebnissen aller anderen Solver auf jeder Instanz verglichen werden. Auf jeder Instanz erhält ein Solver einen Punkt für jeden Solver, der eine schlechtere Lösungsqualität oder unzulässige Ergebnisse erreicht. Im Falle gleicher Lösungsqualität wird zusätzlich die Zeit berücksichtigt, die benötigt wurde, um das Ergebnis zu erreichen. In diesem Fall teilen die beiden verglichenen Solver den Punkt, gewichtet nach ihrer Zeit bis zum Erreichen des Ergebnisses. Der maximale Score eines Solvers ist durch die Anzahl der Instanzen mal die Anzahl der verglichenen Solver minus 1 begrenzt, d. h.

$MiniZinc score  \leq  \\# instances \times \left( \\# solvers -1 \right).$

Für die in den Demonstrationsplots verwendeten Beispieldaten wird der MiniZinc-Score der vier Solver A bis D bestimmt und in der obigen Ergebnistabelle angezeigt. Bei einem Vergleich von 4 Solvern und der Verwendung von 402 Instanzen wird der maximale Score eines Solvers 1206. 

Für eine statistisch belastbare Aussage über die Signifikanz der Leistungsunterschiede werden die verglichenen Solver auf den einzelnen Instanzen gerankt. Die Verteilungen dieser Ränge werden dann mit einem Friedman-Test auf signifikante Unterschiede bei einem Konfidenzniveau von $\alpha=5\%$ untersucht. Wenn der vom Friedman-Test zurückgegebene $p$-Wert unter $\alpha$ liegt, ist dies ein Hinweis auf signifikante Unterschiede.  
Anschließend wird ein Nemenyi-Test durchgeführt, um einen paarweisen Vergleich vorzunehmen. Die Ergebnisse der statistischen Tests zwischen allen Solvern werden in einem Nemenyi-Diagramm zusammengefasst.<br>
![Example Rank Plot](readme_resources/example_rank_plot.png)<br>*Ein Nemenyi-Diagramm basierend auf den Beispieldaten unter Verwendung von 4 Solvern für 402 Probleme.*
 
Die obige Abbildung zeigt ein Beispiel eines Nemenyi-Diagramms für die vier betrachteten Beispiel-Solver A bis D
zusammen mit dem resultierenden $p$-Wert, der vom entsprechenden Friedman-Test zurückgegeben wurde. Von links nach rechts
ordnet das Diagramm die verglichenen Algorithmen von schlechteren zu besseren Rängen. Es zeigt den durchschnittlichen Rang jedes Solvers sowie die kritische Distanz $CD$, die überschritten werden muss, um einen statistisch signifikanten Leistungsunterschied anzuzeigen. Unterhalb des Bereichs zulässiger Ränge verbinden dicke schwarze Linien die Solver, die keine
statistisch signifikanten Unterschiede erzeugt haben. Wie in der obigen Abbildung zu sehen ist,
 übertrifft Solver A die verbleibenden drei Methoden signifikant, während Solver C und Solver D statistisch nicht unterscheidbare Ergebnisse liefern. Gleichzeitig zeigen Solver B und D keine signifikanten Unterschiede. Dennoch weist Solver B eine signifikant bessere Leistung als Solver C auf.

### Einzelinstanz-Evaluierung
Die unten dargestellten Plots zeigen den Fortschritt eines Solvers auf einer ausgewählten einzelnen Instanz über die Zeit $t$ (gemessen in Sekunden). Die Beispielplots zeigen die relative Abweichung $\delta_{rel}$ der von jedem Solver gefundenen besten Lösung von der besten Lösung, die von allen betrachteten Solvern beobachtet wurde, in der linken Abbildung. Entsprechend zeigt der Solver, der zuerst $\delta_{rel} = 0$ erreicht, eine überlegene Leistung in Bezug auf die Geschwindigkeit (d. h. Solver A im bereitgestellten Beispiel).

Fortschritt $\delta_{rel}$          |  Fortschritt $\delta_{rel} \le 10\%$
:-------------------------:|:-------------------------:
![Overall Progress](readme_resources/example_progress_plot.png) | ![Progress to threshold](readme_resources/example_progress_limit_plot.png)

Die rechte Abbildung zeigt die Dynamik von $\delta_{rel}$ hin zu einer benutzerdefinierten Abweichung (hier 10\%) von der Best Known Solution (hier bereitgestellt von Solver A).
Diese Zielabweichung wird vom Benutzer festgelegt und kann dem Visualisierungstool als Parameter übergeben werden. 

Innerhalb der Beispieldaten erreichen alle Solver in den gegebenen Plots die Schwellenentfernung zum besten Ergebnis. Wenn jedoch nicht alle Solver das beste Ergebnis erreichen oder innerhalb der angegebenen Schwelle bleiben, schneiden möglicherweise nicht alle Linien die durchgezogene horizontale rote Linie. 
Dies kann auch passieren, wenn dem Plot eine Begrenzung auf der horizontalen Achse auferlegt wird. Beachten Sie, dass nicht alle Solver bei $t = 0$ starten, da für die Visualisierung der Dynamik nur zulässige Lösungen berücksichtigt werden.


## Referenzen für die enthaltenen FJSSP-Probleme
1. P. Brandimarte. Routing and Scheduling in a Flexible Job Shop by Tabu Search. Annals of Operations Research, 41(3):157–183, 1993.
2. J. Hurink, B. Jurisch, and M. Thole. Tabu search for the job-shop scheduling problem with multi-purpose machines. Operations-Research-Spektrum, vol. 15, no. 4, pp. 205–215, 1994.
3. S. Dauzère-Pérès and J. Paulli. Solving the General Multiprocessor Job-Shop Scheduling Problem. Technical report, Rotterdam School of Management, Erasmus Universiteit Rotterdam, 1994.
4. J. B. Chambers and J. W. Barnes. Flexible Job Shop Scheduling by Tabu Search. The University of Texas, Austin, TX, Technical Report Series ORP96-09, Graduate Program in Operations Research and Industrial Engineering, 1996.
5. I. Kacem, S. Hammadi, and P. Borne. Pareto-Optimality Approach for Flexible, Job-Shop Scheduling Problems: Hybridization of Evolutionary Algorithms and Fuzzy Logic. Mathematics and Computers in Simulation, 60(3-5):245–276, 2002.
6. P. Fattahi, M. S. Mehrabad, and F. Jolai. Mathematical Modeling and Heuristic Approaches to Flexible Job Shop Scheduling Problems. Journal of Intelligent Manufacturing, 18(3):331–342, 2007.
7. Behnke, D., & Geiger, M. J. (2012). Test instances for the flexible job shop scheduling problem with work centers. Arbeitspapier/Research Paper/Helmut-Schmidt-Universität, Lehrstuhl für Betriebswirtschaftslehre, insbes. Logistik-Management.
