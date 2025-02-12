
import os
import itertools
import joblib
import numpy as np
from munch import munchify

feat = "i3d"

views = ["c1","c2",]

for view in views:
    for mode in ["rgb", "flow", "combine"]:
        dataset = f"upfall-{feat}-milbkbn-wan"
        
        print(view, mode)
        
        train_path = f'/media/dev/LaCie/MC-VAD-MIL-DATA/center-crop/upfall-i3d/upfall-{mode}-train-single-camera-{view}-seed1.npy'
        test_path = f'/media/dev/LaCie/MC-VAD-MIL-DATA/center-crop/upfall-i3d/upfall-{mode}-test-single-camera-{view}-seed1.npy'
        output = f'/media/dev/LaCie/MC-VAD-MIL-OUT'

        train_bags = np.load(train_path, allow_pickle=True)
        test_bags = np.load(test_path, allow_pickle=True)

        # create a subdir for the dataset
        output_ds = os.path.join(output, dataset)
        if not os.path.exists(output_ds):
            os.mkdir(output_ds)
            
        # Create a subdir for the view
        output_ds_view = os.path.join(output_ds, view)
        if not os.path.exists(output_ds_view):
            os.mkdir(output_ds_view)
            
        # Create a subdir features_video/i3d/mode
        output_ds_view_featvid = os.path.join(output_ds_view, "features_video")
        if not os.path.exists(output_ds_view_featvid):
            os.mkdir(output_ds_view_featvid)
            
        output_ds_view_featvid = os.path.join(output_ds_view_featvid, "i3d")
        if not os.path.exists(output_ds_view_featvid):
            os.mkdir(output_ds_view_featvid)
            
        output_ds_view_featvid = os.path.join(output_ds_view_featvid, mode)
        if not os.path.exists(output_ds_view_featvid):
            os.mkdir(output_ds_view_featvid)
            
        # Create subdirs for each video name in training bags
        for b in train_bags:
            output_ds_view_featvid_i3d_bagname = os.path.join(output_ds_view_featvid, b["name"])
            if not os.path.exists(output_ds_view_featvid_i3d_bagname):
                os.mkdir(output_ds_view_featvid_i3d_bagname)
            output_ds_view_featvid_i3d_bagname_feat = os.path.join(
                output_ds_view_featvid_i3d_bagname, "feature.npy")
            np.save(output_ds_view_featvid_i3d_bagname_feat, b["X_i"])
            
        # Create subdirs for each video name in testing bags
        for b in test_bags:
            bag_features = b["X_i"]
            output_ds_view_featvid_i3d_bagname = os.path.join(output_ds_view_featvid, b["name"])
            if not os.path.exists(output_ds_view_featvid_i3d_bagname):
                os.mkdir(output_ds_view_featvid_i3d_bagname)
            output_ds_view_featvid_i3d_bagname_feat = os.path.join(output_ds_view_featvid_i3d_bagname, "feature.npy")
            np.save(output_ds_view_featvid_i3d_bagname_feat, b["X_i"])
            
            
        train_bags_names = [b["name"] + "\n" for b in train_bags]
        test_bags_names = [b["name"] + "\n" for b in test_bags]

        train_bags_names[-1] = train_bags_names[-1].replace("\n", "")
        test_bags_names[-1] = test_bags_names[-1].replace("\n", "")

        # generate list of train split
        with open(os.path.join(output_ds_view, "train_split.txt"), "w") as f:
            f.writelines(train_bags_names)

        # generate list of test split
        with open(os.path.join(output_ds_view, "test_split.txt"), "w") as f:
            f.writelines(test_bags_names)

        # Create a subdir for the GTs
        output_ds_view_GT = os.path.join(output_ds_view, "GT")
        if not os.path.exists(output_ds_view_GT):
            os.mkdir(output_ds_view_GT)


        # video label dictionary
        video_label = dict()
        frame_label = dict()

        for b in train_bags:
            video_label[b["name"]] = [b["y_i"]]
            frame_label[b["name"]] = b["y_fi"].ravel()

        for b in test_bags:
            video_label[b["name"]] = [b["y_i"]]
            frame_label[b["name"]] = b["y_fi"].ravel()

        np.save(os.path.join(output_ds_view_GT, "frame_label.pickle"), frame_label)
        os.rename(os.path.join(output_ds_view_GT, "frame_label.pickle.npy"), os.path.join(output_ds_view_GT, "frame_label.pickle"))

        np.save(os.path.join(output_ds_view_GT, "video_label.pickle"), video_label)
        os.rename(os.path.join(output_ds_view_GT, "video_label.pickle.npy"), os.path.join(output_ds_view_GT, "video_label.pickle"))