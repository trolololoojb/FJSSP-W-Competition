import matplotlib.pyplot as plt
import statistics

def calculate_value(fitness, best):
    """
    Berechnet die relative Abweichung eines Fitnesswerts vom besten Referenzwert.

    Args:
        fitness: Der zu bewertende Fitness- oder Makespan-Wert.
        best: Der beste bekannte Referenzwert.

    Return:
        Die relative Abweichung `(fitness - best) / best`.
    """
    return ((fitness - best) / best)

def ecdf_inf(vectors, column, n_instances : int = 402, labels : list[str] = [], instances : list[str] = [], x_lim = (-0.1, 1.0), xlabel : str = None, ylabel : str = None, marker_frequence : int = 10, markers :list[str] = ['x', 'o', '^', '>', 'v', '<', '*']):
    """
    Erzeugt einen ECDF-aehnlichen Plot fuer sortierte Ergebnisvektoren und behandelt dabei auch `inf`-Werte.

    Args:
        vectors: Sortierte Werte pro Solver, die geplottet werden sollen.
        column: Titel des Plots.
        n_instances: Anzahl der betrachteten Instanzen zur Normierung der y-Achse.
        labels: Anzeigenamen der Solver oder Datenreihen.
        instances: Optionaler Platzhalter fuer Instanznamen; wird in der aktuellen Implementierung nicht verwendet.
        x_lim: Untere und obere Grenze der x-Achse.
        xlabel: Beschriftung der x-Achse.
        ylabel: Beschriftung der y-Achse.
        marker_frequence: Zielgroesse fuer die Markerhaeufigkeit pro Linie.
        markers: Verfuegbare Marker fuer die unterschiedlichen Linien.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """
    plot_vectors = []
    for vector in vectors:
        plot_vectors.append([[0.0],[0.0]])
        i = 1
        while i < len(vector):
            if vector[i] == float('inf'):
                break
            if vector[i] == -float('inf'):
                break
            while i < len(vector) and vector[i] == vector[i-1]:
                i += 1
            plot_vectors[-1][0].append(vector[i-1])
            plot_vectors[-1][1].append((i-1)/n_instances)
            i += 1
    for i in range(len(vectors)):
        if i >= len(markers):
            print(f'NOTE: not enough markers defined, recycling already used markers for {labels[i]}')
        plt.plot(plot_vectors[i][0], plot_vectors[i][1], label=[labels[i]], marker=markers[i%len(markers)], markevery=list(range(0, len(plot_vectors[i][0]), max(1, int(len(plot_vectors[i][0])/marker_frequence)))))
    plt.xlim(x_lim[0], x_lim[1])
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, 'both')
    plt.title(column)
    plt.show()

def get_plot_vectors(data : dict[str, list[float]], delta_scope : float = 1.0) -> tuple[list[float], list[str]]:
    """
    Wandelt Solver-Ergebnisse pro Instanz in sortierte Gap-Vektoren fuer die Visualisierung um.

    Args:
        data: Ergebnisdaten pro Solver und Instanz, typischerweise als finale Makespans.
        delta_scope: Skalierungsfaktor fuer den Referenzwert bei der Gap-Berechnung.

    Return:
        Ein Tupel aus sortierten Plot-Vektoren und den zugehoerigen Solver-Labels.
    """
    plot_data = dict()
    best_results = dict()
    example = list(data.keys())[0]
    for key in data[example].keys():
        best = float('inf')
        best_results[key] = best
        plot_data[key] = dict()
        for solver in data.keys():
            if key in data[solver] and data[solver][key] < best:
                best = data[solver][key]
                best_results[key] = best
        for solver in data.keys():
            if key in data[solver]:
                plot_data[key][solver] = max(calculate_value(data[solver][key], best * delta_scope), 0)
            else:
                plot_data[key][solver] = float('inf')
    vectors = []
    labels = []
    for solver in data.keys():
        labels.append(solver)
        vector = []
        for instance in data[example].keys():
            vector.append(plot_data[instance][solver])
        vector.sort()
        vectors.append(vector)
    return vectors, labels

def visualize_gaps(data : dict[str, list[float]], title : str = 'Fitness', n_instances : int = 1000,  x_lim_lb=-0.1, x_lim_ub=1.75, delta_scope : float = 1.0) -> None:
    """
    Visualisiert relative Abweichungen mehrerer Solver ueber viele Instanzen als ECDF-aehnlichen Gap-Plot.

    Args:
        data: Ergebnisdaten pro Solver und Instanz.
        title: Basistitel des Plots.
        n_instances: Anzahl der beruecksichtigten Instanzen fuer die y-Achsen-Normierung.
        x_lim_lb: Untere Grenze der x-Achse.
        x_lim_ub: Obere Grenze der x-Achse.
        delta_scope: Optionaler Schwellenfaktor fuer begrenzte Gap-Darstellungen.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """
    plot_vectors, labels = get_plot_vectors(data, delta_scope)
    if delta_scope < 1.0:
        plot_title = title + ' $\delta_{rel}$ <= '+ f'{(1.0-delta_scope)*100:.2f}%'
    else:
        plot_title = title + ' $\delta_{rel}$'
    ecdf_inf(plot_vectors, plot_title, labels=labels, n_instances=n_instances, x_lim=(x_lim_lb, x_lim_ub), xlabel='$\delta_{rel}$', ylabel='Portion of Instances $\leq\delta_{rel}$')

def progress_plot(fitness_data : list[float], timestamps : list[float], labels : list[str], title : str, marker_frequence : int = 10, markers :list[str] = ['x', 'o', '^', '>', 'v', '<', '*'], xlim_lb : float = -0.01, xlim_ub : float = None, hline : float = None) -> None:
    """
    Plottet den zeitlichen Fortschritt mehrerer Solver auf Basis aufgezeichneter Fitnesswerte.

    Args:
        fitness_data: Fitness- oder Gap-Verlaeufe pro Solver.
        timestamps: Zeitstempel passend zu den Fitnesswerten pro Solver.
        labels: Anzeigenamen der Solver.
        title: Titel des Plots.
        marker_frequence: Zielgroesse fuer die Markerhaeufigkeit pro Linie.
        markers: Verfuegbare Marker fuer die unterschiedlichen Linien.
        xlim_lb: Untere Grenze der x-Achse.
        xlim_ub: Obere Grenze der x-Achse; falls `None`, wird sie aus den Daten bestimmt.
        hline: Optionale horizontale Referenzlinie, z. B. fuer einen Ziel-Gap.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """
    n = 0
    for i in range(len(labels)):
        x = timestamps[i]
        y = fitness_data[i]
        label = labels[i]
        if max(x) > n:
            n = max(x)
        if i >= len(markers):
            print(f'NOTE: not enough markers defined, recycling already used markers for {labels[i]}')
        plt.plot(x, y, label=[label], marker=markers[i%len(markers)], markevery=list(range(0, len(x), max(1, int(len(x)/marker_frequence)))))
    if not xlim_ub:
        xlim_ub = n
    if hline:
        plt.axhline(y=hline, color='r')
        plt.yscale('log')
    plt.xlim(xlim_lb, xlim_ub)
    plt.legend()
    plt.grid(True, 'both')
    plt.xlabel('$t$')
    plt.ylabel('$\delta_{rel}$')
    plt.title(title)
    plt.show()


def visualize_timeline(data : dict[str, list[tuple[float, float]]], title : str = 'Progress', delta_scope : float = 1.0, xlim_lb = None, xlim_ub = None) -> None:
    """
    Bereitet Solver-Zeitreihen auf und visualisiert deren relative Fortschritte ueber die Zeit.

    Args:
        data: Zeitreihen pro Solver als Liste von Tupeln `(zeitpunkt, fitness)`.
        title: Basistitel des Plots.
        delta_scope: Optionaler Schwellenfaktor fuer begrenzte Gap-Darstellungen.
        xlim_lb: Untere Grenze der x-Achse.
        xlim_ub: Obere Grenze der x-Achse.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """
    fitness_data = []
    timestamps = []
    labels = []
    best = float('inf')
    for solver in data:
        for entry in data[solver]:
            if entry[1] < best:
                best = entry[1]
    for solver in data:
        labels.append(solver)
        fitness_data.append([max(calculate_value(entry[1], best), 0) for entry in data[solver]])
        timestamps.append([entry[0] for entry in data[solver]])
    if delta_scope < 1.0:
        plot_title = title + ' $\delta_{rel}$ <= '+ f'{(1.0-delta_scope)*100:.2f}%'
    else:
        plot_title = title + ' $\delta_{rel}$'
    progress_plot(fitness_data=fitness_data, timestamps=timestamps, labels=labels, title=plot_title, xlim_lb=xlim_lb, xlim_ub=xlim_ub, hline=calculate_value(best * (1+(1.0-delta_scope)), best) if delta_scope < 1.0 else None)


def rank_plot(data : dict[str, dict[str, tuple[float,float]]], alpha : float = 0.05, ignore_time : bool = False) -> None:
    """
    Erstellt einen Rangvergleich mehrerer Solver mit `autorank` und visualisiert statistische Unterschiede.

    Args:
        data: Ergebnisdaten pro Solver und Instanz, typischerweise mit Zeit- und Fitnessinformation.
        alpha: Signifikanzniveau fuer den statistischen Test.
        ignore_time: Wenn True, wird nur die Loesungsqualitaet und nicht die Laufzeit betrachtet.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """
    import pandas as pd
    from autorank import autorank, plot_stats
    
    plt_data = dict()
    instances = []
    for solver in data:
        for instance in data[solver]:
            if instance not in instances:
                instances.append(instance)
    for solver in data:
        plt_data[solver] = []
        for instance in instances:
            if instance in data[solver]:
                if ignore_time:
                    plt_data[solver].append(data[solver][instance])
                else:
                    plt_data[solver].append(data[solver][instance][-1][1])
            else:
                plt_data[solver].append(float('inf'))
    df = pd.DataFrame()
    for solver in plt_data:
        df[solver] = plt_data[solver]
    result = autorank(df, alpha=alpha, verbose=False, order='ascending')
    plot_stats(result)
    plt.text(0.0, 0.0, f'Friedman test returns $p$ = {result.pvalue:.3e}\n')
    plt.show()

def show_simulation_results(instance : dict, results : list[float]):
    """
    Visualisiert die Ergebnisse mehrerer Unsicherheits-Simulationen fuer einen einzelnen Plan.

    Args:
        instance: Instanz- oder Planinformationen; genutzt wird insbesondere `instance["e"]` fuer den geplanten Makespan.
        results: Makespans aus den einzelnen Simulationen.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """

    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.scatter(range(len(results)), sorted(results), label='Simulation results', s=10)
    plt.hlines(statistics.mean(results), 0, len(results), color='red', label='Average Makespan')
    plt.hlines(statistics.mean(results) - statistics.stdev(results), 0, len(results), color='red', alpha=0.4, linestyles='--')
    plt.hlines(statistics.mean(results) + statistics.stdev(results), 0, len(results), color='red', alpha=0.4, linestyles='--', label='STDEV Bounds')
    plt.ylabel('Makespan')
    plt.xlabel('Sorted simulations')
    plt.title(f'Original (Planned) Makespan: {max(instance["e"])} | R: {statistics.mean(results)/max(instance["e"]):.4f}')
    plt.legend()
    plt.show()

def show_simulation_comparison(results : list[list[float]], labels : list[str], instance : dict, title : str = None, mark_average : bool = False):
    """
    Vergleicht mehrere Mengen von Simulationsergebnissen uebersichtlich als Sortierplot und Boxplot.

    Args:
        results: Simulationsergebnisse pro Solver oder Variante.
        labels: Anzeigenamen passend zu den Ergebnislisten.
        instance: Instanz- oder Planinformationen; genutzt wird insbesondere `instance["e"]` fuer den geplanten Makespan.
        title: Optionaler Gesamttitel der Abbildung.
        mark_average: Wenn True, wird der Mittelwert jeder Ergebnisliste im linken Plot markiert.

    Return:
        None. Die Funktion erzeugt und zeigt einen Matplotlib-Plot an.
    """
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    i = 0
    averages = []
    lines = []
    for result in results:
        sorted_result = sorted(result)
        x = range(len(result))
        line = axs[0].plot(x, sorted_result, label=labels[i])
        lines.append(line)
        axs[0].scatter(x, sorted_result, s = 5, marker='*')
        if mark_average:
            r = statistics.mean(result)
            index = 0
            for j in range(len(sorted_result)):
                if sorted_result[j] == r:
                    index = j
                    break
                if sorted_result[j] > r:
                    index = j-1
                    break
            averages.append((int(x[index]+0.5), r))
        i+=1
    if mark_average:
        axs[0].scatter([average[0] for average in averages], [average[1] for average in averages], s=20, marker='+', color='red', zorder=10000)
    axs[0].legend()
    axs[0].set_ylabel('Makespan')
    axs[0].set_xlabel('Sorted simulations')
    axs[1].boxplot(results, labels=labels)
    if not title:
        fig.suptitle(f'Original (Planned) Makespan: {max(instance["e"])}')
    else:
        fig.suptitle(title)
    
    plt.show()