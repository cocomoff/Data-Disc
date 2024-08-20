import pandas as pd
import os
import numpy as np
from util import WCP_x, WCP_y
from normalize import normalize_and_count_users_in_grid

if __name__ == "__main__":
    if not os.path.exists("./problems"):
        os.makedirs("./problems", exist_ok=True)

    df = pd.read_pickle("./data/data_WCP500.pickle")
    df = df[["x", "y", "user"]]

    for d in range(1, 8):
        ratio, norm_df, norm_x, norm_y, ix, iy = normalize_and_count_users_in_grid(df, d, WCP_x, WCP_y)

        if d == 1:
            normalized_data = norm_df[["norm_x", "norm_y"]].values
            np.savetxt("problems/normalized_data.csv", normalized_data, fmt="%.6f", delimiter=",")

        np.savetxt(f"problems/poi_{d}.csv", np.array([[norm_x, norm_y]]), fmt="%.6f", delimiter=",")
        np.savetxt(f"problems/poi_idx_{d}.csv", np.array([[ix, iy]]), fmt="%d", delimiter=",")
        np.savetxt(f"problems/ratio_{d}.csv", ratio, fmt="%.6f", delimiter=",")