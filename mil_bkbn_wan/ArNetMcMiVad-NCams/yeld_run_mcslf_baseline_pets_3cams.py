import os, sys
import options
from just_eval_slf_n_cams import just_eval_slf_n_cams



import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == "__main__":
    
    
    base_path_ds = f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-relabeled-i3d-milbkbn-wan"
    base_path_model = f"/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-relabeled-i3d-milbkbn-wan"

    
    for cA,cB,cC in [("002","003","004"),]: 

        print(f"Pair", cA, cB, cC)

        model1_folders_path = os.path.join(base_path_model, f"{cA}/ckpt/model_single/i3d/{cA}/k_4/_Lambda_1_20/rgb")
        model1_folder_names = os.listdir(model1_folders_path)

        model2_folders_path = os.path.join(base_path_model, f"{cB}/ckpt/model_single/i3d/{cB}/k_4/_Lambda_1_20/rgb")
        model2_folder_names = os.listdir(model2_folders_path)

        model3_folders_path = os.path.join(base_path_model, f"{cC}/ckpt/model_single/i3d/{cB}/k_4/_Lambda_1_20/rgb")
        model3_folder_names = os.listdir(model2_folders_path)

        for model1_folder_name, model2_folder_name, model3_folder_name in zip(model1_folder_names, model2_folder_names, model3_folder_names):
            just_eval_slf_n_cams(options.parser_factory(
                device="0",         
                dataset_path = base_path_ds,
                camera_list = [cA, cB],
                feature_modal="rgb",feature_size=1024, late_fusion="Max", 
                pretrained_ckpts =[
                    os.path.join(model1_folders_path,f"{model1_folder_name}/epoch_100_iter_1300.pkl"),
                    os.path.join(model2_folders_path,f"{model2_folder_name}/epoch_100_iter_1300.pkl"),
                    os.path.join(model3_folders_path,f"{model3_folder_name}/epoch_100_iter_1300.pkl"),
                ],
                out_foldername = "mc-slf-baseline",
                seed = 0,
                save_every=100,
                lambdas = "1_20",
                alpha = 4,
                max_epoch = 100,
                model_name = "model_single",
                sample_size = 2,
                loss_combination = "Max",
                #
            ))
            break
        break
    