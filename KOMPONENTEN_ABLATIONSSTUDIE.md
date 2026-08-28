# Komponenten-Ablationsstudie: Surrogat und lokale Suche × RL

Stand: 28. August 2026

## Kurzfazit

Die vollständige 2×2-Komponentenmatrix umfasst 1.200 erfolgreiche Solverläufe: vier Varianten, 30 Instanzen und zehn gepaarte Läufe je Instanz. Es fehlen keine Ergebniszeilen, es gibt keine Duplikate und alle Varianten verwenden dieselben Instanz-, Run- und Seed-Schlüssel.

Die Ergebnisse zeigen ein klares Muster:

- Die gemeinsame Surrogat-/Local-Search-Pipeline verbessert den finalen robusten Makespan gegenüber dem reinen GA um 6,33 % und reduziert den tatsächlichen Such-Simulationsverbrauch um 24,26 %. **H1 wird innerhalb des festgelegten Ablationsprotokolls gestützt.**
- Das Zuschalten von RL zur bereits aktiven Pipeline verbessert die Qualität im Punktschätzer nur um 0,21 %. Das 95-%-Konfidenzintervall schließt den Nulleffekt ein; gleichzeitig werden 2,71 % mehr Raw-FE verbraucht. **H2 wird nicht gestützt.**
- Die vollständige Kombination schlägt RL-only deutlich, aber nicht die Surrogat-/Local-Search-Pipeline ohne RL statistisch belastbar. **H3 wird nicht gestützt.**
- `hpo_with_rl` besitzt den geringfügig besten primären Punktschätzer, ist aber nicht statistisch besser als `hpo_no_rl`. Eine nachgewiesene Überlegenheit der RL-Variante darf daraus nicht abgeleitet werden.

Die wissenschaftlich zentrale Aussage lautet daher: Der beobachtete Gewinn ist vor allem mit der **gemeinsamen Surrogat-/Local-Search-Pipeline** verbunden. Ein isolierter oder zusätzlicher Nutzen der RL-basierten Operatorsteuerung konnte mit diesen Daten nicht belegt werden.

## Fragestellung und Hypothesen

In dieser Ablation werden zwei Faktoren gekreuzt:

1. Random-Forest-basierte Surrogatbewertung **einschließlich der daran gekoppelten lokalen Suche**, und
2. RL-basierte Operatorsteuerung.

Damit werden die Hypothesen wie folgt operationalisiert:

- **H1:** Die surrogatgestützte Verfahrensvariante einschließlich lokaler Suche mit festen Operatorraten reduziert gegenüber dem reinen evolutionären GA die Zahl echter Such-Simulationsbewertungen, ohne die finale Qualität um mehr als 2 % zu verschlechtern.
- **H2:** Die zusätzliche RL-basierte Operatorsteuerung verbessert gegenüber einer ansonsten identischen Surrogat-/Local-Search-Variante mit festen Operatorraten die finale Lösungsqualität und/oder die Budgeteffizienz. Als vorab festgelegter primärer Entscheidungsendpunkt dient die finale Qualität; die Raw-FE werden ergänzend ausgewiesen.
- **H3:** Die Kombination aus Surrogat-/Local-Search-Pipeline und RL ist qualitativ besser als sowohl die Pipeline ohne RL als auch RL ohne Pipeline.

Die Matrix isoliert Surrogat und lokale Suche nicht voneinander. Aussagen über den Effekt des Surrogats **allein** oder der lokalen Suche **allein** sind mit diesem Design nicht möglich.

## Komponentenmatrix

| Kürzel | Variante | Surrogat + lokale Suche | RL | Datenquelle | Erfolgreiche Runs |
|---|---|:---:|:---:|---|---:|
| A | `hpo_plain_ga` | nein | nein | neuer Ablationslauf | 300/300 |
| B | `hpo_no_rl` | ja | nein | vorhandener Referenzlauf | 300/300 |
| C | `hpo_rl_only` | nein | ja | neuer Ablationslauf | 300/300 |
| D | `hpo_with_rl` | ja | ja | vorhandener Referenzlauf | 300/300 |

A und C wurden unter `results/hpo_component_factorial_scenario2` neu berechnet. B und D wurden nicht wiederholt, sondern schreibgeschützt aus `results/hpo_rl_factorial_scenario2` übernommen. Dadurch wurden nur die zwei fehlenden Zellen mit insgesamt 60 SLURM-Tasks ergänzt.

Alle vier Varianten beruhen auf der HPO-Konfiguration `final_rank03_race2_rank04_race1_rank08_TPE0071_8834ed6804`. Für C und D werden zusätzlich die gespeicherten RL-Gewinnerparameter `rl_lr1e-04_u008` verwendet.

## Einheitliches Versuchsprotokoll

| Merkmal | Festgelegter Wert |
|---|---:|
| Instanzen | 30 |
| Runs je Instanz und Variante | 10 |
| Runs je Variante | 300 |
| Runs insgesamt | 1.200 |
| interne Simulationen | 12 |
| separate finale Bewertungssimulationen je Run | 50 |
| finale Bewertungssimulationen insgesamt | 60.000 |
| maximales Raw-FE-Budget je Run | 5.000.000 |
| Solver-Zeitlimit je Run | 129.600 s = 36 h |
| Solver-Worker | 10 |
| Simulations-Worker | 2 |
| Surrogat-Worker | 2 |
| Bootstrap-Stichproben | 10.000 |
| Bootstrap-Seed | `20260803` |

Die Seeds und Unsicherheitsrealisationen sind über alle vier Zellen gepaart. Insgesamt enthalten die 1.200 Runs 5.421.548.268 gespeicherte `raw_function_evaluations`. Dieser Wert zählt bereits die einzelnen stochastischen Suchsimulationen und darf **nicht nochmals mit zwölf multipliziert** werden. Die 50 finalen Bewertungssimulationen pro Run sind separat und nicht in den Raw-FE enthalten.

Damit gelten für alle Zellen dieselben festgelegten Competition-Bedingungen. Die wissenschaftliche Rolle des Experiments bleibt dennoch eine Komponentenablation und kein neuer blinder Competition- oder Holdout-Test.

## Vollständigkeit und Datenintegrität

| Prüfung | A | B | C | D |
|---|---:|---:|---:|---:|
| Instanzen | 30 | 30 | 30 | 30 |
| erwartete Runs | 300 | 300 | 300 | 300 |
| erfolgreiche Runs | 300 | 300 | 300 | 300 |
| fehlgeschlagene Runs | 0 | 0 | 0 | 0 |
| eindeutige Instanz-/Run-/Seed-Schlüssel | 300 | 300 | 300 | 300 |
| doppelte Schlüssel | 0 | 0 | 0 | 0 |

Die Schlüsselmengen aller vier Varianten sind identisch. Damit sind sowohl die 300 Run-Paare je Vergleich als auch die 30 instanzweisen Aggregate vollständig gepaart. Die 60 Fehlerlogs der neuen A-/C-Tasks sind leer.

## Statistische Auswertung

Die statistische Einheit ist die **Instanz**, nicht jeder der zehn Runs einzeln. Dadurch werden die Wiederholungen derselben Probleminstanz nicht fälschlich als 300 unabhängige Beobachtungen behandelt.

Für jede Instanz wird zunächst aggregiert:

- Qualität: Median der zehn final robust bewerteten Makespans,
- Raw-FE und sekundäre Metriken: Mittelwert der zehn Runs.

Danach wird für jede Instanz das Verhältnis `Kandidat/Baseline` gebildet. Der Gesamteffekt ist das geometrische Mittel der 30 Instanzverhältnisse. Ein Verhältnis unter 1 begünstigt den Kandidaten. Die 95-%-Konfidenzintervalle stammen aus einem gepaarten Bootstrap über die 30 Instanzen mit 10.000 Stichproben.

Vorab festgelegte Entscheidungskriterien:

- H1: obere KI-Grenze der Raw-FE kleiner als 1 **und** obere KI-Grenze der Qualität kleiner als 1,02,
- H2: obere KI-Grenze der Qualität von D/B kleiner als 1,
- H3: obere KI-Grenzen der Qualität von D/B und D/C jeweils kleiner als 1.

## Deskriptive Gegenüberstellung aller 1.200 Runs

| Variante | Ø Makespan | Median Makespan | Ø Raw-FE | Ø FE bis Bestfund | Ø Laufzeit in s | Ø finale Stdabw. | Ø finales R | FE-Limit erreicht |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A `hpo_plain_ga` | 3.415,41 | 2.221,55 | 4.891.412 | 4.004.441 | 129.603,00 | 48,53 | 2,1155 | 290/300 = 96,67 % |
| B `hpo_no_rl` | 3.207,13 | 2.056,03 | 4.119.704 | 2.241.303 | 129.600,87 | 47,75 | 2,1221 | 191/300 = 63,67 % |
| C `hpo_rl_only` | 3.414,05 | 2.157,58 | 4.913.265 | 4.111.965 | 129.603,43 | 49,55 | 2,1161 | 290/300 = 96,67 % |
| D `hpo_with_rl` | 3.188,32 | 2.075,83 | 4.147.446 | 2.250.429 | 129.600,88 | 46,08 | 2,1208 | 191/300 = 63,67 % |

Diese Gesamtmittel und Gesamtmediane mischen unterschiedlich skalierte Instanzen. Sie dienen nur der Beschreibung und nicht der Hypothesenentscheidung. Dafür sind die instanznormalisierten, gepaarten Verhältnisse im folgenden Abschnitt maßgeblich.

## Primäre Paarvergleiche

| Effekt | Kandidat/Basis | Qualitätsverhältnis [95-%-KI] | Qualitätsänderung | Qualitäts-S/G/N | Raw-FE-Verhältnis [95-%-KI] | FE-Änderung | Raw-FE-S/G/N |
|---|---|---:|---:|---:|---:|---:|---:|
| Pipeline ohne RL | B/A | 0,9367 [0,9188; 0,9534] | **6,33 % besser** | 29/0/1 | 0,7574 [0,6268; 0,8906] | **24,26 % weniger** | 11/19/0 |
| RL ohne Pipeline | C/A | 1,0022 [0,9960; 1,0097] | 0,22 % schlechter | 13/0/17 | 1,0107 [1,0000; 1,0324] | 1,07 % mehr | 0/29/1 |
| RL mit Pipeline | D/B | 0,9979 [0,9897; 1,0059] | 0,21 % besser | 16/0/14 | 1,0271 [1,0052; 1,0540] | 2,71 % mehr | 4/19/7 |
| Pipeline mit RL | D/C | 0,9327 [0,9155; 0,9475] | **6,73 % besser** | 30/0/0 | 0,7697 [0,6478; 0,8905] | **23,03 % weniger** | 11/19/0 |
| Gesamteffekt | D/A | 0,9347 [0,9166; 0,9504] | **6,53 % besser** | 30/0/0 | 0,7779 [0,6623; 0,8941] | **22,21 % weniger** | 11/19/0 |

S/G/N bezeichnet auf Instanzebene Siege/Gleichstände/Niederlagen des Kandidaten. Bei Raw-FE gilt ein niedrigerer Verbrauch als Sieg.

### Pipeline-Effekt ohne RL: B gegen A

Die Surrogat-/Local-Search-Pipeline verbessert den finalen robusten Makespan im geometrischen Mittel um 6,33 %. Das gesamte 95-%-KI liegt unter 1; die Pipeline ist damit nicht nur innerhalb der 2-%-Marge nichtunterlegen, sondern nach dem festgelegten KI-Kriterium qualitativ überlegen. B gewinnt 29 der 30 Instanzvergleiche.

Gleichzeitig sinken die Raw-FE um 24,26 %. Auch hier liegt die obere KI-Grenze unter 1. Die beiden Teilkriterien von H1 sind damit erfüllt.

### RL-Effekt ohne Pipeline: C gegen A

RL-only ist qualitativ im Punktschätzer 0,22 % schlechter als der reine GA. Das KI umfasst 1, sodass weder eine Verbesserung noch eine Verschlechterung statistisch abgesichert ist. Auch der Raw-FE-Verbrauch wird nicht reduziert. Damit gibt es keinen Hinweis auf einen isolierten Vorteil des RL-Controllers.

### Zusätzlicher RL-Effekt: D gegen B

Durch RL verbessert sich der primäre Qualitätspunktschätzer innerhalb der Pipeline lediglich um 0,21 %. Das KI [0,9897; 1,0059] schließt 1 ein. Die beobachteten 16 Instanzsiege und 14 Niederlagen zeigen ebenfalls keinen klaren, einheitlichen Vorteil.

Der Raw-FE-Verbrauch steigt um 2,71 %; das gesamte KI liegt oberhalb von 1. RL liefert damit weder den geforderten belastbaren Qualitätsgewinn noch eine Verbesserung der Simulationseffizienz.

### Pipeline-Effekt mit RL: D gegen C

Die vollständige Pipeline verbessert RL-only um 6,73 % und gewinnt alle 30 Instanzvergleiche. Die Raw-FE sinken um 23,03 %. Dies bestätigt den starken Pipeline-Effekt auch bei aktiviertem RL.

### Gesamteffekt: D gegen A

Die vollständige Kombination verbessert den reinen GA um 6,53 %, gewinnt alle 30 Instanzvergleiche und benötigt 22,21 % weniger Raw-FE. Da RL weder allein noch zusätzlich zur Pipeline einen nachweisbaren Vorteil zeigt, kann dieser Gesamteffekt jedoch nicht als RL-Effekt interpretiert werden.

## Hypothesenentscheidungen

| Hypothese | Geprüfter Vergleich | Ergebnis des Kriteriums | Entscheidung |
|---|---|---|---|
| H1 | B/A für Qualität und Raw-FE | Qualitäts-KI oben 0,9534 < 1,02; Raw-FE-KI oben 0,8906 < 1 | **gestützt** |
| H2 | D/B für Qualität | Qualitäts-KI oben 1,0059 > 1 | **nicht gestützt** |
| H3 | D/B und D/C für Qualität | D/C erfüllt; D/B erfüllt das Kriterium nicht | **nicht gestützt** |

„Nicht gestützt“ bedeutet nicht, dass ein RL-Effekt logisch ausgeschlossen ist. Die beobachteten Daten reichen lediglich nicht aus, um den vorab geforderten positiven Effekt statistisch zu belegen.

## Sekundäre Endpunkte

Auch hier bedeuten Verhältnisse unter 1 einen niedrigeren Wert beim Kandidaten.

| Kandidat/Basis | FE bis runinternem Bestfund [95-%-KI] | Laufzeit [95-%-KI] | finale Stdabw. [95-%-KI] | Δ finales R (Kandidat − Basis) |
|---|---:|---:|---:|---:|
| B/A | 0,4800 [0,3722; 0,6087] | 0,999984 [0,999949; 1,000001] | 0,9514 [0,9200; 0,9825] | +0,00658 |
| C/A | 1,0373 [0,9976; 1,0812] | 1,000003 [0,999999; 1,000011] | 0,9984 [0,9698; 1,0283] | +0,00064 |
| D/B | 1,0147 [0,9211; 1,1107] | 1,000000 [0,999999; 1,000001] | 0,9771 [0,9520; 1,0027] | −0,00124 |
| D/C | 0,4696 [0,3714; 0,5874] | 0,999980 [0,999940; 1,000001] | 0,9310 [0,9062; 0,9568] | +0,00470 |
| D/A | 0,4871 [0,3838; 0,6131] | 0,999984 [0,999950; 1,000001] | 0,9296 [0,8998; 0,9563] | +0,00534 |

Besonders auffällig ist der Zeitpunkt des runinternen Bestfunds: B erreicht die jeweils eigene später beste Lösung mit rund 52,00 % weniger FE als A; D benötigt gegenüber C rund 53,04 % weniger FE. Dies ist kein Time-to-target-Vergleich bei identischer Zielqualität. Die Laufzeiten unterscheiden sich praktisch nicht, weil nahezu alle Runs bis zum gemeinsamen 36-Stunden-Limit laufen.

## Explorative Komponenteninteraktion

Die explorative Interaktion ist definiert als `(D/B)/(C/A)`. Ein Wert unter 1 würde eine positive Interaktion zugunsten der Kombination anzeigen.

| Metrik | Ratio-of-ratios [95-%-KI] | Interpretation |
|---|---:|---|
| Qualität | 0,9957 [0,9848; 1,0062] | KI umfasst 1; kein nachweisbarer Interaktionseffekt |
| Raw-FE | 1,0162 [0,9979; 1,0387] | KI umfasst 1; kein nachweisbarer Interaktionseffekt |

Die Daten liefern somit keinen statistischen Hinweis darauf, dass RL und Pipeline gemeinsam mehr bewirken als durch ihre separaten Haupteffekte zu erwarten wäre.

## Instanzweise Gegenüberstellung der Qualität

Die folgende Tabelle zeigt den Median der zehn final robusten Makespans je Variante und Instanz. Ein niedrigerer Wert ist besser. A und C sind auf keiner Instanz die beste der vier Varianten; B besitzt auf 14 und D auf 16 Instanzen den niedrigsten beobachteten Median. Diese knappe 14:16-Aufteilung zwischen B und D ersetzt nicht den gepaarten KI-Vergleich, nach dem D gegenüber B nicht überlegen ist.

| Instanz | A | B | C | D | niedrigster Median |
|---|---:|---:|---:|---:|:---:|
| `0_BehnkeGeiger_42_workers.fjs` | 198,27 | 184,10 | 198,61 | 184,47 | **B** |
| `0_BehnkeGeiger_46_workers.fjs` | 315,64 | 283,69 | 314,56 | 288,67 | **B** |
| `0_BehnkeGeiger_60_workers.fjs` | 1.824,34 | 1.871,33 | 1.825,09 | 1.743,00 | **D** |
| `1_Brandimarte_12_workers.fjs` | 1.203,59 | 1.103,05 | 1.196,82 | 1.149,46 | **B** |
| `1_Brandimarte_14_workers.fjs` | 1.530,30 | 1.478,42 | 1.558,69 | 1.444,09 | **D** |
| `1_Brandimarte_7_workers.fjs` | 414,65 | 368,96 | 412,36 | 367,50 | **D** |
| `2a_Hurink_sdata_18_workers.fjs` | 2.837,45 | 2.832,33 | 2.861,08 | 2.770,93 | **D** |
| `2a_Hurink_sdata_1_workers.fjs` | 119,09 | 113,07 | 119,29 | 113,33 | **B** |
| `2a_Hurink_sdata_38_workers.fjs` | 5.382,24 | 5.096,06 | 5.379,78 | 4.899,54 | **D** |
| `2a_Hurink_sdata_40_workers.fjs` | 3.699,04 | 3.477,32 | 3.791,31 | 3.566,98 | **B** |
| `2a_Hurink_sdata_54_workers.fjs` | 19.618,86 | 19.177,04 | 19.108,14 | 18.863,90 | **D** |
| `2a_Hurink_sdata_61_workers.fjs` | 2.090,03 | 1.998,86 | 2.110,51 | 2.012,83 | **B** |
| `2a_Hurink_sdata_63_workers.fjs` | 911,16 | 894,01 | 898,53 | 886,87 | **D** |
| `2b_Hurink_edata_1_workers.fjs` | 108,59 | 107,61 | 109,35 | 107,64 | **B** |
| `2b_Hurink_edata_6_workers.fjs` | 1.272,23 | 1.259,43 | 1.263,46 | 1.246,59 | **D** |
| `2c_Hurink_rdata_28_workers.fjs` | 2.322,72 | 2.073,29 | 2.336,57 | 2.166,26 | **B** |
| `2c_Hurink_rdata_38_workers.fjs` | 4.701,14 | 4.330,07 | 4.744,92 | 4.423,48 | **B** |
| `2c_Hurink_rdata_50_workers.fjs` | 13.618,96 | 12.984,76 | 13.764,02 | 12.936,13 | **D** |
| `2d_Hurink_vdata_18_workers.fjs` | 2.521,83 | 2.405,98 | 2.543,72 | 2.429,90 | **B** |
| `2d_Hurink_vdata_30_workers.fjs` | 3.036,52 | 2.778,28 | 3.063,32 | 2.819,95 | **B** |
| `2d_Hurink_vdata_5_workers.fjs` | 1.227,96 | 1.157,08 | 1.223,70 | 1.184,49 | **B** |
| `3_DPpaulli_15_workers.fjs` | 7.331,42 | 6.140,80 | 7.202,11 | 6.108,40 | **D** |
| `3_DPpaulli_18_workers.fjs` | 7.177,36 | 6.141,67 | 6.994,21 | 6.214,94 | **B** |
| `3_DPpaulli_1_workers.fjs` | 6.469,52 | 6.206,64 | 6.455,79 | 6.238,63 | **B** |
| `3_DPpaulli_9_workers.fjs` | 6.135,91 | 5.485,16 | 6.056,85 | 5.412,49 | **D** |
| `4_ChambersBarnes_10_workers.fjs` | 2.338,77 | 2.251,29 | 2.278,88 | 2.201,61 | **D** |
| `5_Kacem_3_workers.fjs` | 17,33 | 16,86 | 18,84 | 16,70 | **D** |
| `5_Kacem_4_workers.fjs` | 40,01 | 32,36 | 40,07 | 31,55 | **D** |
| `6_Fattahi_14_workers.fjs` | 1.234,19 | 1.214,67 | 1.247,46 | 1.204,35 | **D** |
| `6_Fattahi_20_workers.fjs` | 3.066,74 | 2.850,81 | 3.085,37 | 2.833,06 | **D** |

Die vollständigen instanzweisen Raw-FE-, Bestfund-FE- und Laufzeitwerte stehen in `variant_instance_summary.csv`; die Tabelle wird hier nicht mit weiteren 30 breiten Zeilen dupliziert.

## Welche Variante ist die beste?

Diese Frage hängt vom verwendeten Kriterium ab:

- **Rein nach dem primären Punktschätzer:** D `hpo_with_rl` ist gegenüber B um 0,21 % besser und erzielt auf 16 statt 14 Instanzen den niedrigeren Median.
- **Nach statistisch belegter Qualität:** D und B sind nicht unterscheidbar; das KI von D/B umfasst 1.
- **Nach Simulationseffizienz und Einfachheit:** B `hpo_no_rl` ist vorzuziehen, weil RL keinen belegten Qualitätsgewinn liefert und D 2,71 % mehr Raw-FE benötigt.
- **Gegenüber dem reinen GA:** Sowohl B als auch D sind klar besser; A und C bilden die deutlich schwächere Gruppe.

Für die Masterarbeit sollte daher nicht formuliert werden, D sei „bewiesen die beste Variante“. Korrekt ist:

> Die Variante mit RL erreicht den geringfügig besten beobachteten Qualitätspunktschätzer. Eine statistisch belastbare Überlegenheit gegenüber derselben Surrogat-/Local-Search-Pipeline ohne RL konnte jedoch nicht gezeigt werden.

## Einschränkungen

### Retrospektive Ablation statt unabhängigem Holdout

Die HPO- und RL-Konfigurationen wurden auf überlappenden Scenario-2-Instanzen ausgewählt. Die aktuelle Matrix bewertet festgelegte Komponenten sauber unter einem gemeinsamen Protokoll, verwendet aber keinen vollkommen unabhängigen Testsatz nach Abschluss aller Auswahlentscheidungen. Die Entscheidungssignale sind deshalb als `retrospective_component_ablation` und nicht als unabhängige konfirmatorische Evidenz einzuordnen.

### Zeit- oder FE-Abbruch

Jeder Run endet nach 36 Stunden oder fünf Millionen Raw-FE. A und C erreichen das FE-Limit in 290 von 300 Runs, B und D in 191 von 300 Runs. Da nahezu alle Runs bis zum Zeitlimit laufen, beeinflussen Komponenten-Overhead und Hardwaredurchsatz, wie viele Suchsimulationen innerhalb der 36 Stunden möglich sind. Ohne Anytime-Checkpoints sind die Endpunkt-FE deshalb nur eingeschränkt als kausale Recheneffizienz interpretierbar.

### Hardwareprovenienz

A und C liefen nachweislich auf der CPU-Partition `mpp.share`. Für die älteren B-/D-Referenzruns wurden Partition und Knoten nicht gespeichert. Die FE-Ergebnisse erfüllen die festgelegten operationalen Kriterien, sollten aber nicht als vollständig hardwareunabhängiger Effizienznachweis dargestellt werden.

### Gebündelter Pipeline-Faktor

Surrogatbewertung und lokale Suche werden gemeinsam ein- oder ausgeschaltet. Die starke Wirkung von B/A und D/C ist daher ein Pipeline-Effekt. Für die getrennte Attribution an Random Forest und lokale Suche wäre eine zusätzliche Ablation dieser beiden Komponenten erforderlich.

## Formulierungsvorschlag für die Masterarbeit

> Die Komponenten wurden in einer vollständigen 2×2-Ablation aus Surrogat-/Local-Search-Pipeline und RL-basierter Operatorsteuerung untersucht. Jede der vier Varianten wurde auf 30 Instanzen mit zehn gepaarten Wiederholungen ausgewertet. Als primäre statistische Einheit diente die Instanz; berichtet werden geometrische Mittel instanzweiser Verhältnisse und gepaarte 95-%-Bootstrap-Konfidenzintervalle.
>
> Die Surrogat-/Local-Search-Pipeline ohne RL verbesserte den final robusten Makespan gegenüber dem reinen GA um 6,33 % (Verhältnis 0,9367; 95-%-KI [0,9188; 0,9534]) und reduzierte die Raw-FE um 24,26 % (0,7574; [0,6268; 0,8906]). H1 wird damit innerhalb des Untersuchungsprotokolls gestützt. Das Zuschalten von RL zur Pipeline ergab lediglich eine nicht abgesicherte Qualitätsverbesserung von 0,21 % (0,9979; [0,9897; 1,0059]) bei 2,71 % höherem Raw-FE-Verbrauch. H2 wird daher nicht gestützt. Die vollständige Kombination war RL-only deutlich überlegen, jedoch nicht der Pipeline ohne RL. Folglich wird auch H3 nicht gestützt. Insgesamt ist der beobachtete Leistungsgewinn vor allem mit der gemeinsamen Surrogat-/Local-Search-Pipeline verbunden; ein zusätzlicher Nutzen der RL-Steuerung konnte nicht nachgewiesen werden.

## Ergebnisartefakte und Reproduzierbarkeit

Die Tabellen dieses Berichts wurden aus den bereits erzeugten Auswertungsartefakten übernommen. Für die Erstellung dieser Markdown-Datei wurden keine Solverruns neu gestartet.

- Vollständiger maschinenlesbarer Bericht: [`comparison_report.json`](results/hpo_component_factorial_scenario2/comparison_report.json)
- Alle fünf primären Paarvergleiche: [`comparison_summary.csv`](results/hpo_component_factorial_scenario2/comparison_summary.csv)
- Deskriptive Variantenaggregate: [`variant_summary.csv`](results/hpo_component_factorial_scenario2/variant_summary.csv)
- 120 Varianten-Instanz-Aggregate: [`variant_instance_summary.csv`](results/hpo_component_factorial_scenario2/variant_instance_summary.csv)
- 150 instanzweise Paarvergleiche: [`instance_comparison.csv`](results/hpo_component_factorial_scenario2/instance_comparison.csv)
- 1.500 gepaarte Run-Vergleiche: [`paired_run_comparison.csv`](results/hpo_component_factorial_scenario2/paired_run_comparison.csv)
- Design, Konfigurationen und Provenienz: [`experiment_manifest.json`](results/hpo_component_factorial_scenario2/experiment_manifest.json)
- Rohresultate A/C: [`results/hpo_component_factorial_scenario2`](results/hpo_component_factorial_scenario2)
- Schreibgeschützte Referenzresultate B/D: [`results/hpo_rl_factorial_scenario2`](results/hpo_rl_factorial_scenario2)
- Auswertungsrunner: [`scripts/compare_hpo_component_factorial_scenario2.py`](scripts/compare_hpo_component_factorial_scenario2.py)
- Historische HPO-Auswertung und Auswahlkontext: [`HPO_AUSWERTUNG.md`](HPO_AUSWERTUNG.md)
