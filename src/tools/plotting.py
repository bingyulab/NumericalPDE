import matplotlib.pyplot as plt

def plot_comparison(df, metric, title, ylabel):
    """
    Plot comparison of different solvers based on a specified metric.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing solver results with columns ['solver', 'N', metric]
    - metric (str): The metric to plot (e.g., 'error', 'time')
    - title (str): The title of the plot
    - ylabel (str): The label for the y-axis
    """
    plt.figure(figsize=(8, 6))
    for solver in df['solver'].unique():
        solver_data = df[df['solver'] == solver]
        plt.plot(solver_data['N'], solver_data[metric], marker='o', label=solver)
    
    plt.xlabel('Number of Intervals N')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xscale('log')
    if metric == 'error':
        plt.yscale('log')
    plt.show()
