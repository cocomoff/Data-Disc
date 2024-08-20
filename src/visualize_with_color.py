import random
import pandas as pd
from util import WCP_x, WCP_y
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use("ggplot")

if __name__ == '__main__':
    df = pd.read_pickle("data/data_WCP500.pickle")
    print(df.shape)
    print(df.head(20))
    print(df.iloc[0])

    # 色の割当
    unique_integers = df['user'].unique()
    colors = list(mcolors.CSS4_COLORS.values())
    random.shuffle(colors)
    color_map = {value: colors[i] for i, value in enumerate(unique_integers)}
    df['color'] = df['user'].map(color_map)


    f = plt.figure(figsize=(5, 5))
    a = f.gca()
    df.plot.scatter(x="x", y="y", c="color", s=5, ax=a)
    a.scatter([WCP_x], [WCP_y], marker="x", s=20)
    plt.tight_layout()
    plt.savefig("outputs/scatter.png")
    plt.close()