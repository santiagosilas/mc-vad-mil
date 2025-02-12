import os, sys

def get_command(device, dataset_path, dataset_name_camA, dataset_name_camB, 
                feature_modal,feature_size, late_fusion, pretrained_ckpt1, pretrained_ckpt2):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """
    command = f"""
    python just_eval_slf_2cams.py \
        --device {device} \
        --dataset_path {dataset_path}  \
        --dataset_name_camA {dataset_name_camA}  \
        --dataset_name_camB {dataset_name_camB} \
        --feature_modal {feature_modal} \
        --feature_size {feature_size} \
        --late_fusion {late_fusion} \
        --model_name model_single \
        --pretrained_ckpt1 {pretrained_ckpt1} \
        --pretrained_ckpt2 {pretrained_ckpt2} \
    """
    return command


import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == "__main__":
    
    



    base_path_ds = f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-relabeled-i3d-milbkbn-wan"
    base_path_model = f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-relabeled-i3d-milbkbn-wan"

    
    for cA,cB in [("002","003"),]: 

        print(f"Pair", cA, cB)

        model1_folders_path = os.path.join(base_path_model, f"{cA}/ckpt/model_single/i3d/{cA}/k_4/_Lambda_1_20/rgb")
        model1_folder_names = os.listdir(model1_folders_path)

        model2_folders_path = os.path.join(base_path_model, f"{cB}/ckpt/model_single/i3d/{cB}/k_4/_Lambda_1_20/rgb")
        model2_folder_names = os.listdir(model2_folders_path)

        for model1_folder_name, model2_folder_name in zip(model1_folder_names, model2_folder_names):
            cmd = get_command(
                device="0",         
                dataset_path = base_path_ds,
                dataset_name_camA=cA, dataset_name_camB=cB,
                feature_modal="rgb",feature_size=1024, late_fusion="Max", 
                pretrained_ckpt1 = os.path.join(model1_folders_path,f"{model1_folder_name}/epoch_100_iter_1300.pkl"),
                pretrained_ckpt2 = os.path.join(model2_folders_path,f"{model2_folder_name}/epoch_100_iter_1300.pkl"),
            )
            #print(cmd)
            os.system(cmd)
            #break
        #break
    