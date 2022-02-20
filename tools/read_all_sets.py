import numpy as np

"""Use the tools from OverlapNet"""

def overlap_orientation_npz_file2string_string_nparray(npzfilenames, shuffle=True):

    imgf1_all = []
    imgf2_all = []
    dir1_all = []
    dir2_all = []
    overlap_all = []

    for npzfilename in npzfilenames:
        h = np.load(npzfilename, allow_pickle=True)

        if len(h.files) == 1:
            # old format
            imgf1 = np.char.mod('%06d', h[h.files[0]][:, 0]).tolist()
            imgf2 = np.char.mod('%06d', h[h.files[0]][:, 1]).tolist()
            overlap = h[h.files[0]][:, 2]
            orientation = h[h.files[0]][:, 3]
            n = len(imgf1)
            dir1 = np.array(['' for _ in range(n)]).tolist()
            dir2 = np.array(['' for _ in range(n)]).tolist()
        else:
            imgf1 = np.char.mod('%06d', h['overlaps'][:, 0]).tolist()
            imgf2 = np.char.mod('%06d', h['overlaps'][:, 1]).tolist()
            overlap = h['overlaps'][:, 2]
            dir1 = (h['seq'][:, 0]).tolist()
            dir2 = (h['seq'][:, 1]).tolist()

        if shuffle:
            shuffled_idx = np.random.permutation(overlap.shape[0])
            imgf1 = (np.array(imgf1)[shuffled_idx]).tolist()
            imgf2 = (np.array(imgf2)[shuffled_idx]).tolist()
            dir1 = (np.array(dir1)[shuffled_idx]).tolist()
            dir2 = (np.array(dir2)[shuffled_idx]).tolist()
            overlap = overlap[shuffled_idx]

        imgf1_all.extend(imgf1)
        imgf2_all.extend(imgf2)
        dir1_all.extend(dir1)
        dir2_all.extend(dir2)
        overlap_all.extend(overlap)

    return (imgf1_all, imgf2_all, dir1_all, dir2_all, np.asarray(overlap_all))
