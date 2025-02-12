import os
import time
import options
import itertools
from main import main

if __name__ == "__main__":

    dataset_path = f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-relabeled-i3d-milbkbn-wan"
    num_overlapping_cameras = 3
    combinations = itertools.combinations(["001", "002", "003", "004"], num_overlapping_cameras)
    combinations = [('002', '003')]


    for dataset_names_list in combinations:
        for seed in range(1):
            print(len(dataset_names_list), dataset_names_list, seed)
            start_round = time.time()
            main(options.parser_factory(
                device = "0",
                seed = seed,
                save_every=100,
                dataset_path = dataset_path,
                camera_list = dataset_names_list,
                feature_modal = "rgb",
                feature_size = 1024,
                lambdas = "1_20",
                alpha = 4,
                max_epoch = 100,
                model_name = "model_single",
                sample_size = 2,
                loss_combination = "Max",
                late_fusion = "Max",
                out_foldername = "current",
            ))
            end_round = time.time()
            elapsed_secs = end_round - start_round
            elapsed_mins = elapsed_secs / 60
            print(f"Elapsed time for the round:{elapsed_secs}")
            print(f"Elapsed time for the round:{elapsed_mins:.2f}", )
    