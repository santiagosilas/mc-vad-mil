"""
EXPERIMENTO

"""

import os, sys

def get_command(device, dataset_path, 
                dataset_name_camA, dataset_name_camB,dataset_name_camC,dataset_name_camD, 
                feature_modal,feature_size, lambdas, alpha, max_epoch, loss_combination, late_fusion,
                sample_size, save_every, seed):
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
        --dataset_name_camC {dataset_name_camC} \
        --dataset_name_camD {dataset_name_camD} \
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    for cA,cB, cC, cD in [
            ("001","002", "003", "004"),
        ]:
        for seed in range(5):
            cmd = get_command(
                device="0", 
                dataset_path=f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-arnet-format-i3d", 
                dataset_name_camA=cA, dataset_name_camB=cB, dataset_name_camC=cC,dataset_name_camD=cD,
                feature_modal="rgb",feature_size=1024, lambdas="1_20", alpha=4, max_epoch=100, save_every=100,
                loss_combination = "Max", late_fusion="Max",
                sample_size=2,  seed=seed)
            print(cmd)
            os.system(cmd)
            #break
    