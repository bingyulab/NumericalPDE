import logging
import os
import datetime
import matplotlib.pyplot as plt
import yaml 


def get_logger():
    # Configure logging to output to both the terminal and a log file
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a date-based log directory
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_dir = os.path.join(os.path.dirname(__file__), '../log')
    os.makedirs(log_dir, exist_ok=True)

    # Define the log file path within the date-based directory
    log_file_path = os.path.join(log_dir, f'poisson_problem_{current_date}.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    return logging.getLogger()



def plot2D(X, Y, Z, title=""):
    # Define a new figure with given size and resolution
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,  # Sampling rates for the x and y input data
        cmap=plt.cm.viridis)  # Use the new fancy colormap
    # Set initial view angle
    ax.view_init(30, 225)
    # Set labels and show figure
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    plt.show()
    
    
def load_config():    
    config_path = './configs/config.yaml'
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '../configs', 'config.yaml')

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    return config