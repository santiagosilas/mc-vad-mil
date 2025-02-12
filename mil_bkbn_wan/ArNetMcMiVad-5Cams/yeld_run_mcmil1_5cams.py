"""
EXPERIMENTO

"""

import os, sys

def get_command(device, dataset_path, 
                dataset_name_camA, dataset_name_camB,dataset_name_camC,dataset_name_camD, dataset_name_camE, 
                feature_modal,feature_size, lambdas, alpha, max_epoch, loss_combination, late_fusion,
                sample_size, save_every, seed):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """,
    command = f"""
    python main.py \
        --device {device} \
        --dataset_path {dataset_path}  \
        --dataset_name_camA {dataset_name_camA}  \
        --dataset_name_camB {dataset_name_camB} \
        --dataset_name_camC {dataset_name_camC} \
        --dataset_name_camD {dataset_name_camD} \
        --dataset_name_camE {dataset_name_camE} \
        --feature_modal {feature_modal} \
        --feature_size {feature_size} \
        --Lambda {lambdas} \
        --k {alpha} \
        --max_epoch {max_epoch} \
        --loss_combination {loss_combination} \
        --late_fusion {late_fusion} \
        --sample_size {sample_size} \
        --model_name model_single \
        --save_every {save_every} \
        --seed {seed}
    """
    return command

if __name__ == "__main__":
    for cA,cB, cC, cD, cE in [
        ('1', '2', '3', '4', '5')]:
        for seed in range(5):
            cmd = get_command(
                device="0", 
                dataset_path=f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-arnet-format-i3d", 
                dataset_name_camA=cA, dataset_name_camB=cB, dataset_name_camC=cC,dataset_name_camD=cD,dataset_name_camE=cE,
                feature_modal="rgb",feature_size=1024, lambdas="1_20", alpha=4, max_epoch=10, save_every=1,
                loss_combination = "Max", late_fusion="Max",
                sample_size=12,  seed=seed)
            print(cmd)
            os.system(cmd)
            #break
    