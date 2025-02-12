"""
EXPERIMENTO

"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def get_command(device, dataset_path, 
                dataset_name_camA, dataset_name_camB,
                dataset_name_camC, dataset_name_camD, dataset_name_camE, 
                feature_modal,feature_size, late_fusion, pretrained_ckpt):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """
    command = f"""
    python just_eval.py \
        --device {device} \
        --dataset_path {dataset_path}  \
        --dataset_name_camA {dataset_name_camA}  \
        --dataset_name_camB {dataset_name_camB} \
        --dataset_name_camC {dataset_name_camC} \
        --dataset_name_camD {dataset_name_camD} \
        --dataset_name_camE {dataset_name_camE} \
        --feature_modal {feature_modal} \
        --feature_size {feature_size} \
        --late_fusion {late_fusion} \
        --model_name model_single \
        --pretrained_ckpt {pretrained_ckpt}
    """
    return command

if __name__ == "__main__":
    
    base_path_ds = f"/media/socialab/SILAS-SAND/dataset-arnet-format/HQFS"
    base_path_model = f"/media/socialab/SILAS-SAND/dataset-arnet-format/HQFS"
    
    for cA,cB, cC, cD, cE in [('1', '2', '3', '4',"5"),]:

        print(f"Tuple", cA, cB, cC, cD, cE)

        model_folders_path = os.path.join(base_path_model, f"outputs-{cA}-{cB}-{cC}-{cD}-{cE}/ckpt/model_single/i3d/{cA}-{cB}-{cC}-{cD}-{cE}/k_4/_Lambda_1_20/rgb")
        model_folder_names = os.listdir(model_folders_path)[:5]
        for model_folder_name in model_folder_names:
            cmd = get_command(
                device="0",         
                dataset_path = base_path_ds,
                dataset_name_camA=cA, dataset_name_camB=cB,
                dataset_name_camC=cC, dataset_name_camD=cD, dataset_name_camE=cE,
                feature_modal="rgb",feature_size=1024, late_fusion="Max", 
                pretrained_ckpt = os.path.join(model_folders_path,f"{model_folder_name}/epoch_10_iter_510.pkl")
            )
            #print(cmd)
            os.system(cmd)
    


