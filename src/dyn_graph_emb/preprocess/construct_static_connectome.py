import os
import argparse
from tqdm import tqdm
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
# python -u src/dyn_graph_emb/preprocess/construct_static_connectome.py -d data/ABIDE_pcp/cpac/nofilt_noglobal -a data/aal_SPM12/aal/atlas/AAL.nii -s data/prep_static_aal_batch


def static_connectome_from_timeseries(timeseries):
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([timeseries])[0]

    return np.array(correlation_matrix)


# def binarizer(dyn_corrs, thres=0.2):
#     percentile = 100 - (thres*100)
#     binarized_corrs = []
#     for i in range(dyn_corrs.shape[0]):
#         dyn_corr = dyn_corrs[i]
#         np.fill_diagonal(dyn_corr, np.nan)
#         flat_matrix = dyn_corr[~np.isnan(dyn_corr)]
#         cutoff = np.percentile(flat_matrix, percentile)
#         binary_matrix = (dyn_corr > cutoff).astype(int)
#         np.fill_diagonal(binary_matrix, 0)
#         binarized_corrs.append(binary_matrix)
#     return np.array(binarized_corrs)


# def adjacency_to_dataframe(adj_matrices):
#     time_slices = adj_matrices.shape[0]
#     src = []
#     dst = []
#     times = []
#
#     for t in range(time_slices):
#         matrix = adj_matrices[t]
#         connections = np.argwhere(matrix == 1)
#         for (s, d) in connections:
#             src.append(s)
#             dst.append(d)
#             times.append(t + 1)  # starts at 1
#
#     df = pd.DataFrame({
#         'src': src,
#         'dst': dst,
#         't': times
#     })
#
#     return df


def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    no_ext = os.path.splitext(base_name)[0]
    no_ext = os.path.splitext(no_ext)[0]
    return no_ext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', type=str, default='data/ABIDE_pcp/cpac/nofilt_noglobal')
    parser.add_argument('-s', '--savedir', type=str, default='.')
    parser.add_argument('-a', '--atlas_path', type=str, default='data/aal_SPM12/aal/atlas/AAL.nii')
    # parser.add_argument('--start', type=int, default=0)
    # parser.add_argument('--end', type=int, default=-1)
    # parser.add_argument('--window_size', type=int, default=20)
    # parser.add_argument('--step_size', type=int, default=5)
    # parser.add_argument('--thres', type=int, default=.2)

    args = parser.parse_args()
    opt = vars(args)
    data_dir = opt['datadir']
    save_dir = opt['savedir']
    atlas_path = opt['atlas_path']
    # start = opt['start']
    # end = opt['end']
    # window_size = opt['window_size']
    # step_size = opt['step_size']
    # thres = opt['thres']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'params.txt'), "w") as f:
        for key, value in opt.items():
            print("{}: {}\n".format(key, value))
            f.write("{}: {}\n".format(key, value))

    masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)
    filenames = os.listdir(data_dir)
    filenames = sorted([filename for filename in filenames if filename.endswith('.gz')])

    for i, func_file in tqdm(enumerate(filenames)):
        print(f'preprocessing index: {i}')
        print(f'preprocessing file: {func_file}')
        filepath = os.path.join(data_dir, func_file)
        series = masker.fit_transform(filepath)
        connectome = static_connectome_from_timeseries(series)
        save_name = get_filename_without_extension(filepath) + '.csv'
        save_path = os.path.join(save_dir, save_name)
        np.savetxt(save_path, connectome)
        # bin_dyn_net = binarizer(dyn_net, thres=thres)
        # df = adjacency_to_dataframe(bin_dyn_net)
        # save_name = get_filename_without_extension(filepath) + '.csv'
        # save_path = os.path.join(save_dir, save_name)
        # df.to_csv(save_path, index=False, header=False)


if __name__ == "__main__":
    main()
