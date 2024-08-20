import pandas as pd
import matplotlib.pyplot as plt
from util import WCP_x, WCP_y
from normalize import normalize_and_count_users_in_grid

if __name__ == "__main__":
    plt.style.use("ggplot")
    df = pd.read_pickle("./data/data_WCP500.pickle")
    df = df[["x", "y", "user"]]

    for d in range(1, 8):
        ratio, norm_df, norm_x, norm_y, ix, iy = normalize_and_count_users_in_grid(df, d, WCP_x, WCP_y)
        f = plt.figure(figsize=(7, 5))
        a = f.gca()
        img1 = a.imshow(ratio, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
        a.scatter([norm_x], [norm_y])
        # POI
        a.plot([1 / (2 ** d) * ix, 1 / (2 ** d) * ix], [1 / (2 ** d) * iy, 1 / (2 ** d) * (iy + 1)], lw=3, c="k")
        a.plot([1 / (2 ** d) * ix, 1 / (2 ** d) * (ix + 1)], [1 / (2 ** d) * iy, 1 / (2 ** d) * iy], lw=3, c="k")
        a.plot([1 / (2 ** d) * (ix + 1), 1 / (2 ** d) * (ix + 1)], [1 / (2 ** d) * iy, 1 / (2 ** d) * (iy + 1)], lw=3, c="k")
        a.plot([1 / (2 ** d) * ix, 1 / (2 ** d) * (ix + 1)], [1 / (2 ** d) * (iy + 1), 1 / (2 ** d) * (iy + 1)], lw=3, c="k")
        plt.colorbar(img1, label='Count', ax=a)
        # plt.scatter(normalized_data[:, 0], normalized_data[:, 1], color='red', edgecolor='white')
        plt.tight_layout()
        plt.savefig(f"outputs/data_count_{d}.png")
        plt.show()