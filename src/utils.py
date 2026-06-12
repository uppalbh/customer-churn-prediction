import pickle
import matplotlib.pyplot as plt

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_plot(filename):
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

def print_top_risk(results_df, n=10):
    print(results_df.nlargest(n, "Revenue_At_Risk"))
