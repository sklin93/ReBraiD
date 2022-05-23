import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt


def get_map_17_nets(par_dir=(
    '/content/drive/MyDrive/Colab Notebooks/CRASH_classification' + 
    '/Schaefer2018_200Parcels_17Networks_order.csv')):
    '''used schaefer200plus_2mm_mni_17network.nii.gz during fmri processing,
    it has a slightly differen region definition w.r.t. 7-network 200
    region parcellation'''

    data_17network = np.genfromtxt(par_dir,delimiter=',', usecols=(0, 5))
    region_name_list = pd.read_csv(par_dir, header=None)[1].values.tolist()

    map_17network_region = {}
    map_17network_name = {}
    for i in range(len(data_17network)):
        cur_region = int(data_17network[i][1])
        if cur_region not in map_17network_region:
            map_17network_region[cur_region] = []
            map_17network_name[cur_region] = region_name_list[i]
        map_17network_region[cur_region].append(int(data_17network[i][0])-1)
        map_17network_name[cur_region] = keep_common_prefix(
            map_17network_name[cur_region], region_name_list[i])

    for k, v in map_17network_name.items():
        map_17network_name[k] = v[11:-1] # remove prefix and the '_' at the end
    return map_17network_region, map_17network_name

def keep_common_prefix(a, b):
    ret = ''
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            ret += a[i]
        else:
            break
    return ret

map_17network_region, map_17network_name = get_map_17_nets()
map_label_task = {0: 'rest', 1: 'VWM', 2: 'DYN', 3: 'DOT', 4: 'MOD', 5: 'PVT'}

def plot_individual_AttrA(task_id, num_sample=3, sample_id_max=889,
                          figsize=(12, 20)):
    '''Plot selected samples' A attributions.
    '''
    rows = 1
    cols = num_sample
    plot_id = sample(range(sample_id_max), num_sample)

    print(task_id)
    axes=[]
    fig=plt.figure(figsize=figsize)
    for a in range(rows*cols):
        ax = fig.add_subplot(rows, cols, a+1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        axes.append(ax)
        plt.imshow(task_support_attrs[task_id][plot_id[a]], cmap='Blues')
    plt.show()

def get_task_attr_mean(task_support_attrs, normalize=False):
    '''return a dict storing AttrA's mean, keyed by task id.'''
    # normalize all and check diff
    task_attr_mean = {}
    for k in task_support_attrs:
        cur_task_attr = task_support_attrs[k].mean(0)
        if normalize:
            cur_task_attr = (cur_task_attr - cur_task_attr.min()) / (
                        cur_task_attr.max() - cur_task_attr.min())
        task_attr_mean[k] = cur_task_attr
    return task_attr_mean

def get_test_sc_mean(supports, normalize=True, plot=False):
    '''return N*N SC averaged from test supports'''
    test_sc_mean = supports['test'][0].mean(0)
    if plot:
        print('test_sc_mean')
        plt.imshow(test_sc_mean, cmap='Blues')
    if normalize:
        return (test_sc_mean - test_sc_mean.min()) / (
            test_sc_mean.max() - test_sc_mean.min())
    else:
        return test_sc_mean

def compare_2tasks(target_task1, target_task2, use_normalized_attr=False):
    '''Comparing two tasks'''
    task_attr_mean = get_task_attr_mean(task_support_attrs,
                                        normalize=use_normalized_attr)
    assert target_task1 in task_attr_mean and target_task2 in task_attr_mean, (
        'task id not in data!')
    diff = task_attr_mean[target_task1] - task_attr_mean[target_task2]
    print(diff.min(), diff.max())
    plt.imshow(diff)
    plt.show()

def get_bp_data(data, mean_axis=-1, figsize=(10,6), plot=True):
    ''' Get boxplot data (for ranking), and plot them if desired.

    data: original task_support_attrs or task_signal_attrs, not task averaged!!
    Each data[task_id] has shape [sample_num, N, N] or [sample_num, N, T]
    '''
    task_bp_data = {}
    task_bp_data_no_rl = {}
    for target_task in data:
        print('Task ', target_task, data[target_task].shape)
        bp_data = []
        for k, v in map_17network_region.items():
            bp_data.append(data[target_task].mean(mean_axis)[:, v].mean(-1))
        if plot:
            plt.figure(figsize=figsize)
            plt.boxplot(bp_data)
            plt.show()
        task_bp_data[target_task] = bp_data

        # ignore left and right brain regions (17 networks in total)
        bp_data_no_rl = []
        for i in range(len(bp_data)//2):
            bp_data_no_rl.append(bp_data[i] + bp_data[i + len(bp_data)//2])
        if plot:
            plt.figure(figsize=figsize)
            plt.boxplot(bp_data_no_rl)
            plt.show()
        task_bp_data_no_rl[target_task] = bp_data_no_rl
    return task_bp_data, task_bp_data_no_rl

def get_bp_data_mult(_data, mean_axis=-1, plot=True, figsize=(10,6),
                     whis=1.5, showfliers=False, legend_fz=20,
                     colors=['red','blue','green','purple','tan','pink']):
    ''' Get boxplot data from multiple data sets, and plot them if desired.

    data: a list of task_support_attrs or task_signal_attrs.
    Each data[task_id] has shape [sample_num, N, N] or [sample_num, N, T]
    colors: by default support plot 6 data sets in diff colors. If data set 
            number is larger than 6, overwrite this.
    '''
    task_bp_data = [{} for _ in _data]
    task_bp_data_no_rl = [{} for _ in _data]
    for w in range(len(_data)):
        data = _data[w]
        for target_task in data:
            print('Task ', target_task, data[target_task].shape)
            bp_data = []
            for k, v in map_17network_region.items():
                bp_data.append(data[target_task].mean(mean_axis)[:, v].mean(-1))
            task_bp_data[w][target_task] = bp_data

            # ignore left and right brain regions (17 networks in total)
            bp_data_no_rl = []
            for i in range(len(bp_data)//2):
                bp_data_no_rl.append(bp_data[i] + bp_data[i + len(bp_data)//2])
            task_bp_data_no_rl[w][target_task] = bp_data_no_rl
    if plot:
        num_data_set = len(task_bp_data)
        assert num_data_set > 0, 'empty data'
        for target_task in task_bp_data[0]:
            print(target_task)
            # plot rl boxplot
            plt.figure(figsize=figsize)
            for d_i in range(num_data_set):
                box = plt.boxplot(task_bp_data[d_i][target_task], whis=whis,
                                  showfliers=showfliers,
                                  flierprops=dict(markeredgecolor=colors[d_i]))
                for element in box:
                    plt.setp(box[element], color=colors[d_i])
                box['boxes'][0].set_label('subject %s' % (d_i + 1))
            plt.legend(fontsize=legend_fz)
            plt.show()
            # plot no_rl boxplot
            plt.figure(figsize=figsize)
            for d_i in range(num_data_set):
                box = plt.boxplot(task_bp_data_no_rl[d_i][target_task],
                                  whis=whis, showfliers=showfliers,
                                  flierprops=dict(markeredgecolor=colors[d_i]))
                for element in box:
                    plt.setp(box[element], color=colors[d_i])
                box['boxes'][0].set_label('subject %s' % (d_i + 1))
            plt.legend(fontsize=legend_fz)                   
            plt.show()

    return task_bp_data, task_bp_data_no_rl    

def show_rank(rank_data, keep_LR=True, save_to_csv='',
              map_17network_name=map_17network_name,
              map_label_task=map_label_task):
    ''' Return (and save) rankings dataframe.

    rank_data: dict, {task_id, value/distribution}
    keep_LR: whether to keep the 'LH_', 'RH_' prefix in the df.
    '''
    region_rank = {}
    for target_task in rank_data:
        net_rank = np.argsort(
            [rank_data[target_task][i].mean() for i in range(
                len(rank_data[target_task]))])[::-1]
        # print('Task', target_task, map_label_task[target_task],
        #       net_rank, '\n', [map_17network_name[k] for k in net_rank])
        region_rank[map_label_task[target_task]] = [
            map_17network_name[k] for k in net_rank]
    
    if keep_LR:
        df = pd.DataFrame.from_records(
            [[v[i] for _, v in region_rank.items()] for i in range(
                len(rank_data[target_task])
            )],
            columns=[k for k in region_rank]
        )        
    else:
        df = pd.DataFrame.from_records(
            [[v[i][3:] for _, v in region_rank.items()] for i in range(
                len(rank_data[target_task])
            )],
            columns=[k for k in region_rank]
        )
    df.index += 1
    if save_to_csv:
        df.to_csv(save_to_csv)
        print('df saved to %s' % save_to_csv)
    return df

def show_var(data, scaling=1e8):
    ''' Showing variances'''
    task_vars = []
    for target_task in data:
        task_var = []
        for net_i in range(len(data[target_task])):
            task_var.append(np.var(data[target_task][net_i]))
        task_vars.append(sum(task_var) / len(task_var))
    print([i*scaling for i in task_vars])

def get_single_sess_attr(task_signal_attrs, sess_idx=0):
    ''' returns a dict containing one session's attr per task, keyed by task_id.

        sess_idx: specifies which session to save.
    '''
    # gotten from sliding window cell's output
    task_sess_len = {0: 33, 1: 15, 2: 10, 3:46, 4:53, 5: 48}
    single_sess_attr = {}
    for task_id in task_signal_attrs:
        single_sess_attr[task_id] = (
            task_signal_attrs[task_id][sess_idx*task_sess_len[task_id]:(
                sess_idx+1)*task_sess_len[task_id]])
    return single_sess_attr

def plot_sig(data, supports=None, figsize=(40, 8), mean_axis=None,
             use_diff=False, subnet_map=None, linewidth=4.0):
    ''' Plot figures related to sample averaged attributions.

    - data: should be sample averaged, 2d matrix (N*N)
    mean_axis: if 0, then col avg; if 1, then row avg.
    - supports: a dict, for passing in test supports use use_diff
    - use_diff: if True, compare tasks_attr - test_sc_mean
    - subnet_map: mapping between 17 network to 200 ROI (or similar)
    '''
    rows = 1
    cols = len(data)
    if mean_axis is not None:
        assert mean_axis == 0 or mean_axis == 1
    
    axes=[]
    fig=plt.figure(figsize=figsize)
    for target_task in range(rows*cols):
        assert len(data[target_task].shape) == 2
        ax = fig.add_subplot(rows, cols, target_task+1)
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        axes.append(ax)
        if use_diff:
            assert supports is not None
            diff = data[target_task] - get_test_sc_mean(supports)
            # print(diff.min(), diff.max())
            if mean_axis is None:
                plt.imshow(diff, cmap='Blues')         
            else:
                if subnet_map:
                    subnet_val = []
                    for v in sorted(subnet_map.items()):
                        subnet_val.append(diff.mean(mean_axis)[v[1]].mean())
                    plt.plot(subnet_val, 'k', linewidth=linewidth)
                else:                
                    plt.plot(diff.mean(mean_axis), 'k', linewidth=linewidth)

        else:
            if mean_axis is None:
                plt.imshow(data[target_task], cmap='Blues')
            else:
                if subnet_map:
                    subnet_val = []
                    for v in sorted(subnet_map.items()):
                        # ipdb.set_trace()
                        subnet_val.append(
                            data[target_task].mean(mean_axis)[v[1]].mean())
                    plt.plot(subnet_val, 'k', linewidth=linewidth)

                else:
                    plt.plot(data[target_task].mean(mean_axis), 'k',
                             linewidth=linewidth)
    plt.show()
