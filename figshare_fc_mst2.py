## convert functional connectivity to maximum spanning tree
## statistical analysis of MST measurements
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from nilearn.plotting import plot_connectome
from scipy import stats
from statannotations.Annotator import Annotator

matplotlib.use("Qt5Agg")


def mst_network(connect):
    """Create maximum spanning tree from fc matrix

    :param connect: functional connectivity matrix
    :return: MST measures: [Diameter, Eccentricity, Leaf number, Tree hierarchy]
    """

    # set node postition and labels
    coords = np.array([[-0.309, 0.95, -0.0349], [0.309, 0.95, -0.0349], [0, 0.719, 0.695], [-0.445, 0.673, 0.5],
                       [0.445, 0.673, 0.5], [-0.809, 0.587, -0.0349], [0.809, 0.587, -0.0349], [0, 0.391, 0.921],
                       [-0.576, 0.36, 0.643], [0.576, 0.36, 0.643], [-0.95, 0.309, 0.0349], [0.95, 0.309, -0.0349],
                       [0, 0, 1], [-0.619, 0, 0.695], [0.619, 0, 0.695], [-0.999, 6.12e-17, -0.0349],
                       [0.999, 6.12e-17, -0.0349], [-0.576, -0.36, 0.643], [0.576, -0.36, 0.643],
                       [-0.95, -0.309, -0.0349],
                       [0.95, -0.309, -0.0349], [0, -0.719, 0.921], [-0.445, -0.673, 0.5], [0.445, -0.673, 0.5],
                       [-0.809, -0.587, -0.0349], [0.809, -0.587, -0.0349], [0, -0.999, 0.695],
                       [-0.309, -0.95, -0.0349],
                       [0.309, -0.95, -0.0349]])
    pos = coords[:, :-1]
    labels = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
              "C4", "T3", "T4", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2"]
    pos_dict = dict()
    labels_dict = dict()
    for i in range(pos.shape[0]):
        labels_dict[i] = labels[i]
        pos_dict[i] = pos[i, :]

    # create graph from adjacent matrix
    G = nx.from_numpy_array(connect)

    # set node color
    r, g, b = [0.9, 0.6, 0.5], [0.83, 0.83, 0.83], [30 / 255, 144 / 255, 255 / 255]
    node_color = []
    edge_color = []
    for i in range(29):
        if i in [0, 3, 5, 8, 10, 13, 15, 17, 19, 22, 24, 27]:
            G.nodes[i]['loc'] = 'l'
            node_color.append(r)
        elif i in [1, 4, 6, 9, 11, 14, 16, 18, 20, 23, 25, 28]:
            G.nodes[i]['loc'] = 'r'
            node_color.append(b)
        else:
            G.nodes[i]['loc'] = 'm'
            node_color.append(g)

    # create minimum spanning tree
    MST = nx.maximum_spanning_tree(G, weight='weight')

    # set edge color
    for e in list(MST.edges):
        loc1 = MST.nodes[e[0]]['loc']
        loc2 = MST.nodes[e[1]]['loc']
        if (loc1 == 'l' and loc2 == 'r') or (loc1 == 'r' and loc2 == 'l'):
            edge_color.append(g)
        elif (loc1 == 'l' and loc2 == 'l') or (loc1 == 'l' and loc2 == 'm') or (loc1 == 'm' and loc2 == 'l'):
            edge_color.append(r)
        elif (loc1 == 'r' and loc2 == 'r') or (loc1 == 'r' and loc2 == 'm') or (loc1 == 'm' and loc2 == 'r'):
            edge_color.append(b)
        else:
            edge_color.append(g)

    # # draw MST
    # plt.figure(figsize=(6, 6))
    # nx.draw_networkx(MST, pos=pos_dict, node_color=node_color, edge_color=edge_color, labels=labels_dict, width=1.5)
    # plt.axis("off")
    # adj = nx.to_numpy_array(MST)
    # adj = np.where(adj == 0, 0, 1)
    # coords[:, 0] = coords[:, 0] * 0.9
    # coords[:, 1] = coords[:, 1] * 1.2
    # coords = coords * 70
    # coords[:, 1] = coords[:, 1] - 15
    # edge_prop = dict()
    # edge_prop['lw'] = 1.5
    # edge_prop['c'] = 'black'
    # edge_prop['alpha'] = 0.6
    # node_degree = MST.degree()
    # tmp = np.array([x[1] for x in node_degree])
    # # cmap = plt.get_cmap('GnBu', np.max(tmp) - np.min(tmp) + 1)
    # node_color = plt.cm.Blues([(x - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) * 0.4 + 0.4 for x in tmp])
    # node_size = tmp * 15
    # fig, ax = plt.subplots(1, 1, figsize=(6.2/2.54, 5.8/2.54), layout='constrained')
    # # fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54))  # for flowchart
    # plot_connectome(adj, coords, node_size=node_size, display_mode='z', edge_kwargs=edge_prop, axes=ax,
    #                 node_color='cornflowerblue', alpha=0.2)

    # calculate MST measure
    diameter = nx.diameter(MST)
    ecc = nx.eccentricity(MST)
    ecc_mean = sum(ecc.values()) / len(ecc)
    leaf_nodes = [x for x in MST.nodes() if MST.degree(x) == 1]
    bet_cen = nx.betweenness_centrality(MST)
    tree_hie = len(leaf_nodes) / (2 * 28 * max(bet_cen.values()))

    return np.array([diameter, ecc_mean, len(leaf_nodes), tree_hie])


# def assump_test_bar(x, y, hue, data, pairs):
#     flag = 0
#     for i, pair in enumerate(pairs):
#         data1 = data[(data[x] == pairs[i][0][0]) & (data[hue] == pairs[i][0][1])]
#         data2 = data[(data[x] == pairs[i][1][0]) & (data[hue] == pairs[i][1][1])]
#         w, p = stats.shapiro(data1[y].values - data2[y].values)
#         if p <= 0.05:
#             flag = 1
#     if flag == 0:
#         return 't-test_paired'
#     else:
#         return 'Wilcoxon'


# def assump_test_violin(x, y, data, pairs):
#     flag = 0
#     for i, pair in enumerate(pairs):
#         data1 = data[data[x] == pairs[i][0]]
#         data2 = data[data[x] == pairs[i][1]]
#         w, p = stats.shapiro(data1[y].values - data2[y].values)
#         if p <= 0.05:
#             flag = 1
#     if flag == 0:
#         return 't-test_paired'
#     else:
#         return 'Wilcoxon'


def r2(data, **kws):
    """calculate personr correlation in selected freq band and MI hand

    :param data: dataframe
    :param kws:
    :return:
    """
    col = data.columns.values.tolist()
    col_mst = [x for x in col if x not in ['Freq', 'MI', 'NIHSS']]
    r, p = scipy.stats.pearsonr(data['NIHSS'], data[col_mst[0]])
    ax = plt.gca()
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)


# def bar_plot(data, x, y, hue, ax, pairs, order):
#     snsFig = sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, order=order, palette='coolwarm', errorbar='se')
#     test_method = assump_test_bar(x, y, hue, data, pairs)
#     annotator = Annotator(ax, pairs, data=data, x=x, y=y, hue=hue)
#     annotator.configure(test=test_method, hide_non_significant=True)
#     annotator.apply_test()
#     ax1, test_results = annotator.annotate()


# def box_plot(data, x, y, hue, ax, title, pairs):
#     snsFig = sns.boxplot(x=x, y=y, hue=hue, data=data, ax=ax, palette='muted')
#     for i, box in enumerate([p for p in snsFig.artists]):
#         color = box.get_facecolor()
#         box.set_edgecolor(color)
#         box.set_facecolor((0, 0, 0, 0))
#         # iterate over whiskers and median lines
#         for j in range(6 * i, 6 * (i + 1)):
#             snsFig.lines[j].set_color(color)
#     handles, labels = snsFig.get_legend_handles_labels()
#     snsFig = sns.stripplot(x=x, y=y, hue=hue, data=data, ax=snsFig, palette='muted', dodge=True, size=6)
#     snsFig.legend(handles, labels)
#     snsFig.set(title=title)
#     annotator = Annotator(ax, pairs, data=data, x=x, y=y, hue=hue)
#     annotator.configure(test='Mann-Whitney', hide_non_significant=True)
#     annotator.apply_test()
#     ax1, test_results = annotator.annotate()


def violin_plot_single(data, x, y, pairs):
    """Violin plot without hue for flowchart

    :param data: dataframe
    :param x: NIHSS group
    :param y: Leaf number
    :param pairs: paired group to statistical analysis
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=(4.4 / 2.54, 4.2 / 2.54), layout='constrained')
    # sns.stripplot(data=data[data.NIHSS_Group == 'Group1'], x=x, y=y, ax=ax, legend=False, size=6, jitter=0.2,
    #               facecolors='none', edgecolor=[0.35, 0.49, 0.75], linewidth=1)
    # sns.stripplot(data=data[data.NIHSS_Group == 'Group2'], x=x, y=y, ax=ax, legend=False, size=6, jitter=0.2,
    #               facecolors='none', edgecolor=[0.85, 0.54, 0.37], linewidth=1)
    # sns.violinplot(data=data, x=x, y=y, ax=ax, inner='box', palette='muted', inner_kws={'box_width':6})
    sns.stripplot(data=data, x=x, y=y, ax=ax, legend=False, size=2.5, jitter=0.13, palette='muted',
                  facecolors='none')
    for dots in ax.collections:  # remove facecolor and set edgecolor
        facecolors = dots.get_facecolors()
        dots.set_edgecolors(facecolors.copy())
        dots.set_facecolors('none')
        dots.set_linewidth(1)

    sns.violinplot(data=data, x=x, y=y, ax=ax, palette='muted', legend='brief', width=0.4)
    ax.set(ylabel='Leaf number', xlabel='NIHSS group')
    ax.spines[['right', 'top']].set_visible(False)
    colors = []
    for collection in ax.collections:  # remove facecolor and set edgecolor
        if isinstance(collection, matplotlib.collections.PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor('none')
    if len(ax.lines) == 2 * len(colors):  # suppose inner=='box'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)

    # t test
    annotator = Annotator(ax, data=data, x=x, y=y, pairs=pairs)
    annotator.configure(test='t-test_ind', hide_non_significant=True, loc='outside')
    annotator.apply_test()
    annotator.annotate()


def violin_plot_ahand(data, x, y, pairs):
    """Violin plot for ahand MI

    :param data: dataframe
    :param x: NIHSS group

    :param y: Leaf number
    :param pairs: paired group to statistical analysis
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    sns.stripplot(data=data, x=x, y=y, ax=ax, legend=False, size=2.5, jitter=0.13, palette='muted',
                  facecolors='none')
    for dots in ax.collections:  # remove facecolor and set edgecolor
        facecolors = dots.get_facecolors()
        dots.set_edgecolors(facecolors.copy())
        dots.set_facecolors('none')
        dots.set_linewidth(1)

    sns.violinplot(data=data, x=x, y=y, ax=ax, palette='muted', legend='brief', width=0.4)
    ax.legend(loc='upper center', ncols=2, bbox_to_anchor=(0.5, 0.98), frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set(xlabel='NIHSS group', ylabel='Leaf number')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    colors = []
    for collection in ax.collections:  # remove facecolor and set edgecolor
        if isinstance(collection, matplotlib.collections.PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor('none')
    if len(ax.lines) == 2 * len(colors):  # suppose inner=='box'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)
    for h in ax.legend_.legendHandles:
        if isinstance(h, matplotlib.patches.Rectangle):
            h.set_edgecolor(h.get_facecolor())
            h.set_facecolor('none')
            h.set_linewidth(1.5)

    # t test
    s1, p1 = scipy.stats.shapiro(data[data[x] == 'Low'][y])
    s2, p2 = scipy.stats.shapiro(data[data[x] == 'High'][y])
    if p1<0.05 or p2<0.05:      # nonnormal distribution
        test_method = 'Mann-Whitney'
    else:
        test_method = 't-test_ind'
    annotator = Annotator(ax, data=data, x=x, y=y, pairs=pairs)
    annotator.configure(test=test_method, hide_non_significant=True, loc='outside')
    annotator.apply_test()
    annotator.annotate()


def mst_measure(index, pli_left, pli_right):
    """

    :param index: subject index
    :param pli_left: imagery coherence matrix in 3 freq bands for left paralysis sub
    :param pli_right: imagery coherence matrix in 3 freq bands for right paralysis sub
    :return: MST measures for left and right paralysis sub
    """
    mst_stat_left = np.zeros((index.shape[0], 3, 4))
    mst_stat_right = np.zeros((index.shape[0], 3, 4))
    for i, s in enumerate(index):
        data_left = pli_left[s, :, :, :]
        data_right = pli_right[s, :, :, :]
        for j in range(3):
            mst_stat_left[i, j, :] = mst_network(np.squeeze(data_left[:, :, j]))
            mst_stat_right[i, j, :] = mst_network(np.squeeze(data_right[:, :, j]))

    return mst_stat_left, mst_stat_right

if __name__ == "__main__":
    plt.ion()
    # adjust figure format
    plt.rc('font', size=9, family='Arial', weight='normal')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.labelweight'] = 'normal'
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['axes.titleweight'] = 'normal'
    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # load imagery coherence matrix: 50 subjects, 29 channels, 3 frequency bands
    pli_left = np.load('data_load/ImCoh_data/alpha_beta12/imcoh_left.npy')
    pli_right = np.load('data_load/ImCoh_data/alpha_beta12/imcoh_right.npy')

    # load subjects info
    sub = pd.read_csv('dataset/subject.csv')

    # plot mst of subject 43,22 (figure 3 right column for paper)
    mst_example1 = mst_network(np.squeeze(pli_left[42, :, :, 1]))  # 1:low beta band
    mst_example2 = mst_network(np.squeeze(pli_left[21, :, :, 1]))

    # calculate MST for all subjects
    mst_stat_left, mst_stat_right = mst_measure(np.arange(50), pli_left, pli_right)
    mst_stat_uhand, mst_stat_ahand = np.zeros((50, 3, 4)), np.zeros((50, 3, 4))     # uhand:unaffected hand, ahand:affected hand
    for i in range(50):
        if sub['ParalysisSide'][i] == 'left':   # for left paralysis sub, MI of right hand is MI of unaffected hand
            mst_stat_uhand[i, :, :] = mst_stat_right[i, :, :]
            mst_stat_ahand[i, :, :] = mst_stat_left[i, :, :]
        else:
            mst_stat_uhand[i, :, :] = mst_stat_left[i, :, :]
            mst_stat_ahand[i, :, :] = mst_stat_right[i, :, :]
    mst_stat_all = np.concatenate((mst_stat_ahand, mst_stat_uhand), axis=1)
    mst_stat_all = np.reshape(mst_stat_all, (300, 4))  # [ahand ahand ahand uhand uhand uhand]*50

    # make all data to one dataframe
    data = pd.DataFrame(np.repeat(sub.values, 6, axis=0), columns=sub.columns)
    data['MI'] = pd.Series(['Ahand', 'Ahand', 'Ahand', 'Uhand', 'Uhand', 'Uhand'] * 50)
    data['Freq'] = pd.Series(['alpha', 'beta1', 'beta2', 'alpha', 'beta1', 'beta2'] * 50)
    data['Diameter'] = pd.Series(mst_stat_all[:, 0])
    data['Ecc'] = pd.Series(mst_stat_all[:, 1])
    data['Leaf'] = pd.Series(mst_stat_all[:, 2])
    data['Tree'] = pd.Series(mst_stat_all[:, 3])
    data['NIHSS'] = data['NIHSS'].astype(float)
    data['Age'] = data['Age'].astype(float)
    data['Duration'] = data['Duration'].astype(float)

    # statistic of subject factors
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    sns.countplot(x='Gender', data=sub, ax=ax[0, 0], palette='Spectral')
    abs_values = sub['Gender'].value_counts(ascending=False).values
    ax[0, 0].bar_label(container=ax[0, 0].containers[0], labels=abs_values, label_type='center', size=15)
    sns.countplot(x='IsFirstTime', data=sub, ax=ax[0, 1], palette='Spectral')
    abs_values = sub['IsFirstTime'].value_counts(ascending=False).values
    ax[0, 1].bar_label(container=ax[0, 1].containers[0], labels=abs_values, label_type='center', size=15)
    sns.countplot(x='ParalysisSide', data=sub, ax=ax[1, 0], order=['left', 'right'], palette='Spectral')
    abs_values = sub['ParalysisSide'].value_counts(ascending=False).values
    ax[1, 0].bar_label(container=ax[1, 0].containers[0], labels=abs_values, label_type='center', size=15)
    sns.histplot(x='NIHSS', data=sub, ax=ax[1, 1], kde=True, edgecolor='none', palette='pastel')

    sub = sub.drop(index=[4, 5, 14, 48])  # delete FFT abnormal and left handness
    data = data[~data['Participant_ID'].isin(['sub-05', 'sub-06', 'sub-15', 'sub-49'])]

    # regression analysis: only "leaf" metric in beta1 band is correlated with NIHSS
    metric = 'Leaf'
    f = 'beta1'
    g = sns.lmplot(data, x='NIHSS', y=metric, row='MI', col='Freq', fit_reg=True, height=4,
                   scatter_kws={'color': 'royalblue'}, line_kws={'color': 'royalblue'})
    g.map_dataframe(r2)

    # comparision analysis: "low NIHSS group" VS "high NIHSS group"
    data_nihss = data.copy()
    data_nihss['NIHSS_Group'] = pd.cut(x=data_nihss.NIHSS, bins=[0, 3, 11], labels=['Low', 'High'])
    data_nihss.loc[data_nihss.Participant_ID == 'sub-16', 'NIHSS_Group'] = 'Low'
    pairs = [(('Low', 'Ahand'), ('Low', 'Uhand')), (('High', 'Ahand'), ('High', 'Uhand')),
             (('Low', 'Ahand'), ('High', 'Ahand')), (('Low', 'Uhand'), ('High', 'Uhand'))]
    pairs = [('Low', 'High')]
    # violin plot visual example for flowchart
    violin_plot_single(data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand')], x='NIHSS_Group', y=metric,
                       pairs=pairs)
    # violin plot of ahand MI for all subs and left-paralysis subs
    violin_plot_ahand(data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand')], x='NIHSS_Group', y=metric,
                      pairs=pairs)
    violin_plot_ahand(
        data=data_nihss[(data_nihss.Freq == f) & (data_nihss.MI == 'Ahand') & (data_nihss.ParalysisSide == 'left')],
        x='NIHSS_Group', y=metric, pairs=pairs)

    # analyze factor of Gender and IsFirstTime
    # data_factor = data[(data.Freq == f) & (data.MI == 'Ahand')]
    # fig, ax = plt.subplots(1,2, figsize=(14,6))
    # sns.lineplot(data=data_factor, x='NIHSS', y='Leaf', hue='Gender', style='Gender', markers=True, dashes=False, errorbar=("sd", 0.5), err_style='band', ax=ax[0])
    # sns.lineplot(data=data_factor, x='NIHSS', y='Leaf', hue='IsFirstTime', style='IsFirstTime', markers=True, dashes=False, errorbar=("sd", 0.5), err_style='band', ax=ax[1])
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # box_plot(data=data_nihss[(data_nihss.MI=='Ahand') & (data_nihss.Freq==f)], x='NIHSS_Group', y=metric, hue='Gender', ax=ax[0], title='Gender Effect',
    #          pairs=[(('Low','male'), ('Low', 'female')), (('High','male'), ('High', 'female'))])
    # box_plot(data=data_nihss[(data_nihss.MI=='Ahand') & (data_nihss.Freq==f)], x='NIHSS_Group', y=metric, hue='IsFirstTime', ax=ax[1], title='IsFirstTime Effect',
    #          pairs=[(('Low','yes'), ('Low', 'no')), (('High','yes'), ('High', 'no'))])


    # correct correlation coefficients using Spearman correlation and permutation test
    metric = 'Leaf'
    f = 'beta1'
    data_factor = data[(data.Freq == f) & (data.MI == 'Ahand')]      # filter dataframe to ahand MI in selected freq band

    s1, p1 = scipy.stats.shapiro(data_factor['NIHSS'])  # normal distribution test: if p<0.05, nonnormal distribution
    s2, p2 = scipy.stats.shapiro(data_factor[metric])

    def statistic(x):  # explore all possible pairings by permuting `x`
        rs = stats.spearmanr(x, data_factor[metric]).statistic  # ignore pvalue
        transformed = rs * np.sqrt((len(x) - 2) / ((rs + 1.0) * (1.0 - rs)))
        return transformed


    fig, ax = plt.subplots(1, 1, figsize=(4.4 / 2.54, 3.8 / 2.54), layout='constrained')
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's': 8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.set(ylabel='Leaf number')
    # ax[0].set_title('All subjects')
    ax.spines[['right', 'top']].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # fig, ax = plt.subplots(1,1, figsize=(6,6))
    # sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color':'royalblue'}, line_kws={'color':'royalblue'})
    # r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    # ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    # p = ref.pvalue
    # ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes)

    # linear regression between Leaf number and NIHSS score(figure 4 second row for paper)
    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    sns.regplot(data=data_factor, x='NIHSS', y=metric, ax=ax, scatter_kws={'color': 'royalblue', 's':8},
                line_kws={'color': 'royalblue'})
    r = scipy.stats.spearmanr(data_factor['NIHSS'], data_factor[metric]).statistic
    ref = stats.permutation_test((data_factor['NIHSS'],), statistic, alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.set(ylabel='Leaf number')
    # ax[0].set_title('All subjects')
    ax.spines[['right', 'top']].set_visible(False)


    def statistic2(x):  # explore all possible pairings by permuting `x` (for left paralysis subs)
        rs = stats.spearmanr(x, data_factor[data_factor.ParalysisSide == 'left'][metric]).statistic  # ignore pvalue
        transformed = rs * np.sqrt((len(x) - 2) / ((rs + 1.0) * (1.0 - rs)))
        return transformed


    fig, ax = plt.subplots(1, 1, figsize=(4.8 / 2.54, 4.4 / 2.54), layout='constrained')
    sns.regplot(data=data_factor[data_factor.ParalysisSide == 'left'], x='NIHSS', y=metric, ax=ax,
                scatter_kws={'color': 'orange', 's':8}, line_kws={'color': 'orange'})
    r = scipy.stats.spearmanr(data_factor[data_factor.ParalysisSide == 'left']['NIHSS'],
                              data_factor[data_factor.ParalysisSide == 'left'][metric]).statistic
    ref = stats.permutation_test((data_factor[data_factor.ParalysisSide == 'left']['NIHSS'],), statistic2,
                                 alternative='less', permutation_type='pairings')
    p = ref.pvalue
    ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes, fontsize=10)
    ax.set(ylabel='Leaf number')
    # ax[1].set_title('Left paralysis subjects')
    ax.spines[['right', 'top']].set_visible(False)
    plt.ioff()
    plt.show()
