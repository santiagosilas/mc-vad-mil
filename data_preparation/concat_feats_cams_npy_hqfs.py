"""
The following script generate camera concatenations for baseline comparisons:
cd data_preparation
python concat_feats_cams_npy_pets.py
python concat_feats_cams_npy_upfall.py
python concat_feats_cams_npy_hqfs.py
"""

import os
import itertools
import joblib
import numpy as np
from munch import munchify

def concatenate_views(input_folder, filenames):
    views2conc = [
        np.load(os.path.join(input_folder, f), allow_pickle = True) 
        for f in filenames]
    bags_concatenated = list()
    for paired_bags in zip(*views2conc):
        paired_bags = list(paired_bags)
        for i in range(len(paired_bags)):
            paired_bags[i] = munchify(paired_bags[i])
        bags_concatenated.append({
            'name':paired_bags[0].name,
            'X_i':np.hstack([b.X_i for b in paired_bags]),
            'y_i':paired_bags[0].y_i,
            'y_fi':paired_bags[0].y_fi,
        })
    return bags_concatenated

if __name__ == "__main__":
    
    overlapping_cams = 2 # change this to the desired number of overlapping cameras concatenated

    npy_dir = f"/media/dev/LaCie/MC-VAD-MIL-DATA/center-crop/hqfs-relabeled-i3d"
    out_dir = f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-{overlapping_cams}cams"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    get_filename = lambda fmode, split, cam:     f"hqfs-i3d-{fmode}-{split}-cam{cam}.npy"
    get_out_filename = lambda fmode,split,views: f"hqfs-i3d-{fmode}-{split}-views-{views}.npy"



    camera_views = ["1", "2","3", "4", "5"]

    for feat_mode in ["rgb", "flow", "combine"]:
        for split in ["train", "test"]:
            for count_cams in [overlapping_cams,]:
                for views in itertools.combinations(camera_views,count_cams):
                    
                    key=str(views)\
                        .replace("'" , "")\
                        .replace(" " , "")\
                        .replace("(" , "")\
                        .replace(")" , "")\
                        .replace("," , "-") 
                    print(feat_mode, split, count_cams, views, key)
                    np.save(
                        os.path.join(out_dir,get_out_filename(feat_mode,split,key)), 
                        concatenate_views(
                            input_folder = npy_dir, 
                            filenames = [get_filename(feat_mode, split, view) for view in views])
                    )