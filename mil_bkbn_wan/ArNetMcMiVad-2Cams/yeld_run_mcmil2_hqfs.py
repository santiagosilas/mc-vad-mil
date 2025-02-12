"""
EXPERIMENTO

"""

import os, sys
import time

def get_command(device, dataset_path, dataset_name_camA, dataset_name_camB, 
                feature_modal,feature_size, lambdas, alpha, max_epoch, loss_combination, late_fusion,
                sample_size, save_every, seed, out_foldername):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """
    command = f"""
    python main.py \
        --device {device} \
        --dataset_path {dataset_path}  \
        --dataset_name_camA {dataset_name_camA}  \
        --dataset_name_camB {dataset_name_camB} \
        --feature_modal {feature_modal} \
        --feature_size {feature_size} \
        --Lambda {lambdas} \
        --k {alpha} \
        --max_epoch {max_epoch} \
        --loss_combination {loss_combination} \
        --late_fusion {late_fusion} \
        --sample_size {sample_size} \
        --model_name model_single_mtv2 \
        --save_every {save_every} \
        --seed {seed}  \
        --out_foldername {out_foldername}
    """
    return command

if __name__ == "__main__":
    new_path = f"/media/dev/Silas-PhD-Backup/MyPhD-ProcData/DS-MC-HQFS/BkBn-ArNet/2024-11-Rerun2GenFigs/Hqfs"
    for cA,cB in [("2","4"),]:
        for seed in range(1):
            start_round = time.time()
            print(f"Run Pair <{cA}, {cB}> Execution {seed + 1}")
            cmd = get_command(
                device="0", dataset_path=new_path, 
                dataset_name_camA=cA, dataset_name_camB=cB,
                feature_modal="rgb",feature_size=1024, lambdas="1_20", alpha=4, max_epoch=10, save_every=1,
                loss_combination = "PairwiseMeanOfClipScoresInBags", late_fusion="Max",
                sample_size=12,  seed=seed, out_foldername = None)
            os.system(cmd)
            end_round = time.time()
            elapsed_secs = end_round - start_round
            elapsed_mins = elapsed_secs / 60
            print(f"Elapsed time for the round:{elapsed_secs}")
            print(f"Elapsed time for the round:{elapsed_mins:.2f}", )
    