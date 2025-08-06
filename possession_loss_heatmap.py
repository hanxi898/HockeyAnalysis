import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
import scipy.stats as stats


def possession_change_events(df, teamid_filter=None):
    """
    返回所有丢失球权事件，基于 ishomegame 和 period 统一转换坐标到“向右进攻”视角。

    参数:
        df: 原始事件数据（必须包含 teamid, teaminpossession, xadjcoord, yadjcoord, ishomegame, period）
        teamid_filter: 若提供，仅返回该 team 的事件

    返回:
        DataFrame: 球权丢失事件 + 坐标统一到右攻视角
    """
    df = df.dropna(subset=['teaminpossession', 'ishomegame'])
    df = df.sort_values(by=['gameid', 'compiledgametime']).reset_index(drop=True)

    # 当前和下一行控球情况
    current_poss = df['teaminpossession'].values[:-1]
    next_poss = df['teaminpossession'].values[1:]
    current_team = df['teamid'].values[:-1]

    # 丢球事件（本队控球 -> 非本队控球）
    lost_mask = (current_poss == current_team) & (next_poss != current_team)
    loss_df = df[:-1][lost_mask].copy()

    # 获取 ishomegame 和 period
    loss_df['ishomegame'] = df['ishomegame'].values[:-1][lost_mask]
    loss_df['period'] = df['period'].values[:-1][lost_mask]
    loss_df['next_teaminpossession'] = next_poss[lost_mask]

    # 计算方向：如果当前 period 是主队向右（1或3节 + ishomegame True）或 客队向右（2节 + ishomegame False）
    loss_df['attack_dir'] = np.where(
        ((loss_df['ishomegame'] == True) & (loss_df['period'].isin([1, 3]))) |
        ((loss_df['ishomegame'] == False) & (loss_df['period'] == 2)),
        1, -1
    )

    # 坐标变换：以“向右”为正方向
    loss_df['x_plot'] = loss_df['xadjcoord'] * loss_df['attack_dir']
    loss_df['y_plot'] = loss_df['yadjcoord'] * loss_df['attack_dir']

    # 可选筛选
    if teamid_filter is not None:
        loss_df = loss_df[loss_df['teamid'] == teamid_filter]

    return loss_df[['gameid', 'teamid', 'compiledgametime', 'period', 'ishomegame',
                    'teaminpossession', 'next_teaminpossession',
                    'eventname', 'type', 'x_plot', 'y_plot']]





def kde_density(xy_coords, grid_x, grid_y, bandwidth=20):
    """
    在给定网格上计算 KDE 密度。
    """
    kde = gaussian_kde(xy_coords.T, bw_method=0.2)  # 固定平滑程度
    mesh_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    z = kde(mesh_coords).reshape(grid_x.shape)
    return z


def visualize_style_possession_losses_zscore(style_name, agg_df, possession_changes, grid_res=100):
    """
    可视化指定风格的 Z-score 标准化丢球位置热力图，与全体平均相比。

    参数:
        style_name: 风格名字符串
        agg_df: 聚合数据（含 style、teamid、gameid）
        possession_changes: 事件级别数据（含 x_plot, y_plot）
        grid_res: 网格分辨率（越大越细）
    """
    # 1. 提取符合该风格的样本
    style_teams = agg_df[agg_df['style'] == style_name][['teamid', 'gameid']]
    merged_style = possession_changes.merge(style_teams, on=['teamid', 'gameid'], how='inner')
    all_xy = possession_changes[['x_plot', 'y_plot']].dropna().values
    style_xy = merged_style[['x_plot', 'y_plot']].dropna().values

    if len(style_xy) < 30:
        print(f"风格 {style_name} 样本过少（{len(style_xy)} 个事件），无法绘制。")
        return

    # 2. 构造网格
    x = np.linspace(-100, 100, grid_res)
    y = np.linspace(-42.5, 42.5, grid_res)
    X, Y = np.meshgrid(x, y)

    # 3. 计算 KDE 密度（全体 & 当前风格）
    global_density = kde_density(all_xy, X, Y)
    style_density = kde_density(style_xy, X, Y)

    # 4. 计算 Z-score
    z_map = (style_density - global_density.mean()) / global_density.std()

    # 5. 画图
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(X, Y, z_map, levels=20, cmap='coolwarm', extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label("Z-Score Normalized Density")

    # 冰球场参考线
    plt.plot([-100, 100, 100, -100, -100], [-42.5, -42.5, 42.5, 42.5, -42.5], 'k-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='-', linewidth=1.5, label='Center Line (x=0)')
    plt.axvline(x=25, color='b', linestyle='--', linewidth=1.2, label='Blue Line (x=±25)')
    plt.axvline(x=-25, color='b', linestyle='--', linewidth=1.2)
    plt.gca().add_patch(plt.Circle((0, 0), 15, fill=False, color='b', linewidth=1))

    plt.title(f"Possession Loss Heatmap for Style: {style_name}\n(Z-Score Normalized, with Global Average Background)")
    plt.xlabel("X Coordinate (attacking right)")
    plt.ylabel("Y Coordinate")
    plt.xlim(-105, 105)
    plt.ylim(-45, 45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
