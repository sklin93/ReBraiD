'''
After exporting .eeglab files as .mat files
'''
import os
import glob
import scipy.io as spio
import nibabel as nib
import numpy as np
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

base_d = '/content/drive/MyDrive/MRI-EEG data'
eeg_d = os.path.join(base_d, 'eeg')
fmri_d = os.path.join(base_d, 'fmri/matfiles')
fmri_bold_d = os.path.join(base_d, 'fmri/fmri_bold')
sc_d = os.path.join(base_d, 'sc')

###### HELPER FUNCTIONS #####
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

def checkIsAP(arr): 
    '''Check if the given array is arithmetic progression'''
    n = len(arr)
    if (n == 1): return True
    # arr.sort()
    d = arr[1] - arr[0] 
    for i in range(2, n): 
        if (arr[i] - arr[i-1] != d): 
            return False
    return True

def closest_idx(pt, li, k=1):
    '''find the index of the closest point in the li to the pt
    pt: 3d coordinate
    li: list of 3d coordinates
    return the index list of k-nearest neighbors
    '''
    dist = []
    for i in range(len(li)):
        dist.append(np.linalg.norm(pt - li[i]))
    return np.argsort(dist)[:k]
########################

def get_eeg(comn_ids):
    '''
    # eeg_data['loc']
    eeg_data['time_res'] #common time resolution
    eeg_data[subj_id][sess#]
    '''
    print('loading EEG')
    eeg_data = {}
    eeg_data['time_res'] = 1 / 640.0

    for _subj in tqdm(comn_ids):
        eeg_data[_subj] = {}
       
        subj_dir = os.path.join(eeg_d, _subj)
        sess_dirs = sorted([os.path.join(subj_dir, o) for o in os.listdir(
                            subj_dir) if (os.path.isdir(os.path.join(
                                subj_dir, o)) and o[0] == 's')])

        for sess_dir in sess_dirs:
            # cur_sess_num = int(sess_dir.split(os.path.sep)[-1].split('-')[-1])
            cur_sess_num = int(sess_dir.split(os.path.sep)[-1].split(
                '-')[-1].split('_')[0][1:])
            try:
                data = loadmat(os.path.join(
                    sess_dir, 'eeg', 'data.mat'))['data']
                eeg_data[_subj][cur_sess_num] = data
            except FileNotFoundError:
                pass

    return eeg_data

def get_fmri_bold(comn_ids, atlas, task_name='rest'):
    '''
    fMRI time series on the voxel level
    fmri_data['time_res']
    fmri_data[_subj][cur_sess_num]
    '''
    num_roi = atlas.max()
    fmri_data = {}
    fmri_data['time_res'] = 0.910 # each time bin is 0.910s
    for _subj in tqdm(comn_ids):

        fmri_data[_subj] = {}

        subj_dir = os.path.join(fmri_d, 'sub-'+_subj)
        sess_dirs = sorted(
                    [os.path.join(subj_dir, o) for o in os.listdir(subj_dir)
                    if (os.path.isdir(os.path.join(subj_dir, o)) and
                        o[0] == 's')])
        
        for sess_dir in sess_dirs:
            cur_sess_num = int(sess_dir.split(os.path.sep)[-1].split('-')[-1])
            name = glob.glob(os.path.join(sess_dir, 'func',
                        '0_sub-*_'+task_name+'_bold_MNI_3mm.nii.gz'))
            if len(name) != 1:
                print('>> No/ More than one bold file???')
            img = nib.load(name[0])
            ts_img = img.get_fdata() # shape (52, 62, 54, 326), (x, y, z, #ts)
            # x, y, z, _ = ts_img.shape
            # show_slices([ts_img[26, :, :, 0], ts_img[:, 31, :, 0], ts_img[:, :, 27, 0]])
            
            roi_bold = np.zeros((num_roi, ts_img.shape[-1]))
            for region_id in (1, num_roi+1):
                roi_bold[region_id] = ts_img[atlas == region_id].mean()
            fmri_data[_subj][cur_sess_num] = roi_bold

    return fmri_data

def get_fmri(comn_ids, num_region, task_name='rest'):
    '''
    fMRI time series on the region level
    fmri_data['time_res']
    fmri_data[_subj][cur_sess_num]
    '''
    print('loading fMRI')

    fmri = {}
    fmri['time_res'] = 0.910 # each time bin is 0.910s
    for _subj in tqdm(comn_ids):

        fmri[_subj] = {}

        subj_dir = os.path.join(fmri_d, 'sub-'+_subj)
        sess_dirs = sorted([os.path.join(subj_dir, o
                            ) for o in os.listdir(subj_dir) if (os.path.isdir(
                                os.path.join(subj_dir, o)) and o[0] == 's')])

        for sess_dir in sess_dirs:
            cur_sess_num = int(sess_dir.split(os.path.sep)[-1].split('-')[-1])
            name = glob.glob(os.path.join(sess_dir, '*' + task_name + '*' + 
                             str(num_region) + 'plus.mat'))
            if len(name) > 1:
                print(_subj, cur_sess_num,'>> More than one fmri file?')
                continue
            if len(name) == 0:
                print(_subj, cur_sess_num,'>> No fmri file')
                continue

            data = spio.loadmat(name[0])
            # fmri[_subj][cur_sess_num] = data['corrected_bold'][:, :num_region]
            fmri[_subj][cur_sess_num] = data['bold'][:, :num_region]
    return fmri

def get_sc(comn_ids, num_region):
    '''
    sc[_subj][cur_sess_num]
    '''
    print('loading SC')

    sc = {}
    for _subj in tqdm(comn_ids):

        sc[_subj] = {}

        subj_dir = os.path.join(sc_d, 'sub-'+_subj)
        sess_dirs = sorted([os.path.join(subj_dir, o) for o in os.listdir(
                            subj_dir) if (os.path.isdir(os.path.join(
                            subj_dir, o)) and o[0] == 's')])
        
        for sess_dir in sess_dirs:
            cur_sess_num = int(sess_dir.split(os.path.sep)[-1].split('-')[-1])
            name = glob.glob(os.path.join(sess_dir, '*' + str(num_region) +
                            'plus.mat'))
            if len(name) > 1:
                print(_subj, cur_sess_num,'>> More than one sc file?')
                continue
            if len(name) == 0:
                print(_subj, cur_sess_num,'>> No sc file')
                continue

            data = spio.loadmat(name[0])
            '''
            CRASH_schaefer400plus_2mm_mni_17network_lps_count_pass
            CRASH_schaefer400plus_2mm_mni_17network_lps_mean_length_pass
            CRASH_schaefer400plus_2mm_mni_17network_lps_ncount_pass
            CRASH_schaefer400plus_2mm_mni_17network_lps_gfa_pass
            '''
            # TODO: Choose which metric within these four?
            sc[_subj][cur_sess_num] = data['CRASH_schaefer'+str(num_region)+
                            'plus_2mm_mni_17network_lps_ncount_pass'
                            ][:num_region, :num_region]
    return sc

def get_comn_ids(F_only=False):
    # get common subject id (having EEG, fMRI, SC)
    num_li = [str(x) for x in np.arange(10)]
    
    if not F_only:
        eeg_ids = sorted([o for o in os.listdir(eeg_d)
                        if (os.path.isdir(os.path.join(eeg_d, o)) and
                            o[0] in num_li)])
    
    fmri_ids = sorted([o for o in os.listdir(fmri_d)
                    if (os.path.isdir(os.path.join(fmri_d, o)) and
                        o[4] in num_li)])
    fmri_ids = [fmri_id[4:] for fmri_id in fmri_ids]

    sc_ids = sorted([o for o in os.listdir(sc_d)
                    if (os.path.isdir(os.path.join(sc_d, o)) and 
                        len(o) > 4 and o[4] in num_li)])
    sc_ids = [sc_id[4:] for sc_id in sc_ids]

    if F_only:
        comn_ids = sorted([v for v in sc_ids if v in fmri_ids])
    else:
        comn_ids = sorted([v for v in eeg_ids if v in fmri_ids])
        comn_ids = sorted([v for v in sc_ids if v in comn_ids])
    # print(len(comn_ids), 'subjects:', comn_ids)
    return comn_ids

def get_region_assignment(num_region):
    '''Group assignment mapping: need to assign brain regions to closest electrodes
    '''
    coor_mri = np.loadtxt(os.path.join(sc_d, 'Parcellations/MNI',
                          'Schaefer2018_' + str(num_region) +
                          'Parcels_17Networks_order_FSLMNI152_2mm.txt'),
                          usecols=(3,4,5,6))
    coor_eeg = np.loadtxt(os.path.join(base_d, 'utils/eeg_coor_conv/ny_x_z'),
                          usecols=(1,2,3))
    eeg_permute = [1, 0, 2]
    coor_eeg = coor_eeg[:, eeg_permute]

    region_assignment = {}
    for k in range(len(coor_eeg)):
        region_assignment[k] = []

    # for each mri region, assign it to the closest eeg electrode
    for i in range(num_region):
        cur_centroid = coor_mri[coor_mri[:, -1] == (i+1)][:,:3].mean(0)
        closest_eeg = closest_idx(cur_centroid, coor_eeg, k=3)
        for eeg_idx in closest_eeg:
            region_assignment[eeg_idx].append(i)
    return region_assignment

if __name__ == '__main__':

    comn_ids = get_comn_ids()
    num_region = 200 # 200 or 400

    eeg = get_eeg(comn_ids)
    sc = get_sc(comn_ids, num_region)
    fmri = get_fmri(comn_ids, num_region)

    # check and keep only common sessions for each subject
    for subj in comn_ids:
        comn_sess = []
        for k in eeg[subj]:
            if k in sc[subj] and k in fmri[subj]:
                comn_sess.append(k)
        eeg[subj] = {k: v for k, v in eeg[subj].items() if k in comn_sess}
        sc[subj] = {k: v for k, v in sc[subj].items() if k in comn_sess}
        fmri[subj] = {k: v for k, v in fmri[subj].items() if k in comn_sess}
    
    region_assignment = get_region_assignment()

    with open('eeg.pkl', 'wb') as handle:
                pickle.dump(eeg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('sc.pkl', 'wb') as handle:
                pickle.dump(sc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('fmri.pkl', 'wb') as handle:
                pickle.dump(fmri, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('assignment.pkl', 'wb') as handle:
                pickle.dump(region_assignment, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
