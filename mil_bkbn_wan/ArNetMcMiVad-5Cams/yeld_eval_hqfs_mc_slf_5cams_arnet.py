"""
EXPERIMENTO

"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def get_command(device, dataset_path, dataset_name_camA, dataset_name_camB, dataset_name_camC, dataset_name_camD, dataset_name_camE, 
                feature_modal,feature_size, late_fusion, pretrained_ckpt1, pretrained_ckpt2, pretrained_ckpt3, pretrained_ckpt4, pretrained_ckpt5):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """
    command = f"""
    python just_eval_slf_5cams.py \
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
        --pretrained_ckpt1 {pretrained_ckpt1} \
        --pretrained_ckpt2 {pretrained_ckpt2} \
        --pretrained_ckpt3 {pretrained_ckpt3} \
        --pretrained_ckpt4 {pretrained_ckpt4} \
        --pretrained_ckpt5 {pretrained_ckpt5} \
    """
    return command

if __name__ == "__main__":
    
    base_path_ds = f"/media/socialab/SILAS-SAND/dataset-arnet-format/HQFS"
    base_path_model = f"/media/socialab/SILAS-SAND/dataset-arnet-format/HQFS"
    
    for cA,cB, cC, cD, cE in [('1', '2', '3', '4',"5"),]:

        print(f"Tuple", cA, cB, cC, cD, cE)

        model1_folders_path = os.path.join(base_path_model, f"{cA}/ckpt/model_single/i3d/{cA}/k_4/_Lambda_1_20/rgb")
        model1_folder_names = os.listdir(model1_folders_path)

        model2_folders_path = os.path.join(base_path_model, f"{cB}/ckpt/model_single/i3d/{cB}/k_4/_Lambda_1_20/rgb")
        model2_folder_names = os.listdir(model2_folders_path)

        model3_folders_path = os.path.join(base_path_model, f"{cC}/ckpt/model_single/i3d/{cC}/k_4/_Lambda_1_20/rgb")
        model3_folder_names = os.listdir(model3_folders_path)

        model4_folders_path = os.path.join(base_path_model, f"{cD}/ckpt/model_single/i3d/{cD}/k_4/_Lambda_1_20/rgb")
        model4_folder_names = os.listdir(model4_folders_path)

        model5_folders_path = os.path.join(base_path_model, f"{cE}/ckpt/model_single/i3d/{cE}/k_4/_Lambda_1_20/rgb")
        model5_folder_names = os.listdir(model5_folders_path)



        for model1_folder_name, model2_folder_name, model3_folder_name, model4_folder_name, model5_folder_name in zip(
            model1_folder_names, model2_folder_names, 
            model3_folder_names, model4_folder_names, model5_folder_names):
            cmd = get_command(
                device="0",         
                dataset_path = base_path_ds,
                dataset_name_camA=cA, dataset_name_camB=cB, dataset_name_camC=cC, dataset_name_camD=cD, dataset_name_camE=cE,
                feature_modal="rgb",feature_size=1024, late_fusion="Max", 
                pretrained_ckpt1 = os.path.join(model1_folders_path,f"{model1_folder_name}/epoch_10_iter_510.pkl"),
                pretrained_ckpt2 = os.path.join(model2_folders_path,f"{model2_folder_name}/epoch_10_iter_510.pkl"),
                pretrained_ckpt3 = os.path.join(model3_folders_path,f"{model3_folder_name}/epoch_10_iter_510.pkl"),
                pretrained_ckpt4 = os.path.join(model4_folders_path,f"{model4_folder_name}/epoch_10_iter_510.pkl"),
                pretrained_ckpt5 = os.path.join(model5_folders_path,f"{model5_folder_name}/epoch_10_iter_510.pkl"),
            )
            os.system(cmd)
            