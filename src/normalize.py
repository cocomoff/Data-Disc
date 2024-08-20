import numpy as np

def normalize_and_count_users_in_grid(
    df, d, poi_x: int, poi_y: int, user_column="user"
):
    """
    ユーザごとに各グリッドセルを1回でも経由した割合を計算する
    """
    users = df[user_column].unique()
    total_users = len(users)

    data = df[["x", "y"]].to_numpy()
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    df["norm_x"] = normalized_data[:, 0]
    df["norm_y"] = normalized_data[:, 1]

    # POI
    xy = np.array([[poi_x, poi_y]])
    norm_xy = (xy - min_vals) / (max_vals - min_vals)
    norm_x, norm_y = norm_xy[0]

    # 離散化・カウント
    grid_size = 2**d
    user_grid_counts = np.zeros((grid_size, grid_size, total_users), dtype=bool)

    for user_idx, user_id in enumerate(users):
        user_data = df[df[user_column] == user_id]
        indices = (user_data[["norm_x", "norm_y"]] * grid_size).astype(int)
        # indices = (user_data.iloc[:, 1:3] * grid_size).astype(int)
        indices = np.clip(indices, 0, grid_size - 1)

        for x, y in indices.values:
            user_grid_counts[y, x, user_idx] = True

    # norm_x, norm_y をカウント
    indices_poi = (norm_xy * grid_size).astype(int)
    ix, iy = indices_poi[0]

    # 各グリッドセルでのユニークユーザの割合を計算
    user_grid_ratios = np.sum(user_grid_counts, axis=2) / total_users
    return user_grid_ratios, df, norm_x, norm_y, ix, iy

