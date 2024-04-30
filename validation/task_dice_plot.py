import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

contrast_name = [
    'Emotion - Faces', 'Emotion - Shapes',
    'Gambling - Punish', 'Gambling - Reward',
    'Language - Math', 'Language - Story',
    'Motor - Cue', 'Motor - LF', 'Motor - LH', 'Motor - RF', 'Motor - RH', 'Motor - T',
    'Relational - Match', 'Relational - Rel',
    'Social - Random', 'Social - TOM',
    'WM - 2bk body', 'WM - 2bk face', 'WM - 2bk place', 'WM - 2bk tool',
    'WM - 0bk body', 'WM - 0bk face', 'WM - 0bk place', 'WM - 0bk tool'
]


def plot_curve(sub_list_file, methods, dataset):
    sub_list = np.loadtxt(sub_list_file, dtype=str)

    dice_curves = []
    for method in methods:
        tmp = np.load('result/{}_{}.npy'.format(method, dataset), allow_pickle=True).item()[
            'dice_curve']
        print(tmp.shape)
        dice_curves.append(tmp)
    dice_curves = np.stack(dice_curves, axis=0)  # (#method, #sub, #Perc, #Contrast)
    print(dice_curves.shape)  # (#method, #sub, #Perc, #Contrast)

    np_methods = np.array(methods)
    np_methods = np.tile(np_methods, (dice_curves.shape[1], dice_curves.shape[2], 1)).transpose(2, 0, 1).reshape(-1)
    # np_methods = np.tile(np_methods, (1, dice_curves.shape[2], 1)).transpose(2, 0, 1).reshape(-1)

    arange = np.arange(5, 51) / 100
    arange = np.tile(arange, (dice_curves.shape[0], dice_curves.shape[1], 1)).reshape(-1)
    # arange = np.tile(arange, (dice_curves.shape[0], 1, 1)).reshape(-1)

    df = {
        "Dice": dice_curves.reshape(-1),
        "method": np.tile(np.array(methods)[:, None, None, None],
                          (1, dice_curves.shape[1], dice_curves.shape[2], dice_curves.shape[3])).reshape(-1),
        "percent": np.tile(np.arange(5, 51)[None, None, :, None] / 100,
                           (len(methods), dice_curves.shape[1], 1, dice_curves.shape[3])).reshape(-1),
        "contrast": np.tile(np.array(contrast_name)[None, None, None, :],
                            (len(methods), dice_curves.shape[1], dice_curves.shape[2], 1)).reshape(-1)
    }
    df = pd.DataFrame(df)
    # print(df)
    sns.set_theme(style="ticks", palette='pastel', font_scale=1.8)
    grid = sns.relplot(data=df,
                       x="percent", y="Dice", col="contrast", hue="method",
                       palette="muted", kind="line",
                       col_wrap=4, height=4, sizes=2,
                       # legend=None,
                       facet_kws=dict(margin_titles=False))

    # Adjust the tick positions and labels
    grid.set(xlim=(0., 0.5),
             # ylim=(0.2, 0.8),
             xlabel=None, ylabel=None)
    grid.set_titles("{col_name}")

    # Adjust the arrangement of the plots
    plt.savefig('plot/05_line_plot24_{}.tif'.format(dataset))
    # plt.show()
    plt.close()
    

if __name__ == '__main__':
    plot_curve('../sub_lists/sub_list_drt_rt.txt',
               ['05_task_syn_group', '05_task_syn_Unet_syn_msm_dr'], 'HCP1200')

    plot_curve('../sub_lists/sub_list_drt_rt.txt',
               ['05_task_syn_group', '05_task_syn_Unet_syn_msm_dr'], 'HCP_retest')
