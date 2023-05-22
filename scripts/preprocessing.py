import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
def get_missing_cols(df, lower_threshold, upper_threshold):
    """
    Retrieves columns from a DataFrame that have a percentage of missing values within a specified threshold.
    
    Args:
        df (pandas.DataFrame): The DataFrame to check for missing values.
        lower_threshold (float): The lower threshold for the percentage of missing values (exclusive).
        upper_threshold (float): The upper threshold for the percentage of missing values (exclusive).
    
    Returns:
        list: A list of column names that meet the specified threshold criteria.
    """
    cols_missing = []
    total_rows = df.shape[0]
    for col in df.columns:
        missing_values = df[col].isnull().sum()
        percent_missing = missing_values / total_rows
        if percent_missing > lower_threshold and percent_missing < upper_threshold:
            print(f"{col} has {missing_values} missing values ({percent_missing:.2%} missing)")
            cols_missing.append(col)
    return cols_missing

def plot_histograms(df, cols, size):
    """
    Plots histograms for specified columns in a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the columns to plot.
        cols (list): A list of column names to plot histograms for.
        size (tuple): The size of the resulting figure (width, height).
    """
    n_cols = min(len(cols), 5)
    n_rows = - (-len(cols) // n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=size)
    axs = axs.flatten()
    for i, col in enumerate(cols):
        axs[i].set_xlabel(col, fontsize=16)
        axs[i].set_ylabel('Count', fontsize=16)
        if df[col].dtype == bool:
            df[col].dropna().astype(int).hist(ax=axs[i])
        elif df[col].dtype == object:
            pass
        else:
            df[col].dropna().hist(ax=axs[i])
    plt.show()
    
def plot_percent_by_sex(df):
    #title_fig = '.jpg'
    group_by_df = (df.groupby('sex')["phq9_cat_end"]
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())
    plt.figure(figsize=(14,8))
    g = sns.histplot(x = 'sex', 
                     hue = "phq9_cat_end",
                     weights= 'percent',
                     #hue_order = ['4.0','3.0','2.0','1.0','0.0',],
                     #palette=[sns.color_palette()[2],sns.color_palette()[0],sns.color_palette()[1]],
                     multiple = 'stack',
                     data= group_by_df,
                     shrink = 0.5,
                     discrete=True,
                     legend=True)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0, 101, 20), np.arange(100, -1, -20))
    plt.xlabel('Sex', fontsize=18)
    g.set(xticks=range(0,3,1),xticklabels=["Female","Male","Other"])
    plt.ylabel('Percentage', fontsize=18)
    plt.title("", y=-0.2)
    sns.move_legend(g, loc = "center left",labels=['4 (high severity)','3','2','1','0 (low severity)'],title="Depression Severity", bbox_to_anchor=(1, .51))

    for rect in g.patches:

        h = rect.get_height()
        w = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        if h>0:
            g.annotate(f'{float(h):.0f}%', xy=(x + w/2,y +h/2), 
                       xytext=(0, 0), textcoords='offset points', ha='center', va='center'
                      )
    #g.figure.savefig(title_fig,bbox_inches='tight')
    

def save_list_to_pkl(list_object, path, file_name):
    """
    Save a list object as a pickle file.

    Parameters:
        list_object (list): The list object to be saved.
        path (str): The directory path where the file will be saved.
        file_name (str): The desired name of the pickle file (without the extension).

    Returns:
        None
    """
    with open(path + file_name + '.pkl', 'wb') as file:
        pickle.dump(list_object, file)
        
def load_list_from_pkl(file_path):
    """
    Load a pickled list object from a file.

    Parameters:
        file_path (str): The path to the pickle file.

    Returns:
        list: The loaded list object.
    """
    with open(file_path, 'rb') as file:
        list_object = pickle.load(file)
    return list_object