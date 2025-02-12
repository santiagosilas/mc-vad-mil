"""
MODEL EVALUATION
"""

import os, sys

def get_command(device, dataset_path, dataset_name, feature_modal, feature_size, pretrained_ckpt):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """
    command = f"""
    python just_eval.py \
        --device {device} \
        --dataset_path {dataset_path} \
        --dataset_name {dataset_name} \
        --feature_modal {feature_modal} \
        --feature_size {feature_size} \
        --pretrained_ckpt {pretrained_ckpt}
    """
    return command

if __name__ == "__main__":
    base_path = "/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-arnet-format-i3d"
    cmd = get_command(device="0", dataset_path=base_path, dataset_name="002", feature_modal="rgb", feature_size=1024, 
                    pretrained_ckpt = f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-arnet-format-i3d/002/ckpt/model_single/i3d/002/k_4/_Lambda_1_20/rgb/2025-01-08-13-32-36/epoch_100_iter_1300.pkl")
    os.system(cmd)




