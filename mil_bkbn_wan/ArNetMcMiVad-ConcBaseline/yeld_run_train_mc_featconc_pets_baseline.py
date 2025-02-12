"""
EXPERIMENTO

EXECUÇÃO COM PARAMETROS DEFAULT (APHA e LAMBDAS) DO AR-NET
PARA O PETS, O SAMPLE_SIZE EH TAMANHO 4 POR CONTA DO DATASET (NAO PODE SER MAIOR QUE ISSO)
"""

import os, sys, itertools

def get_command(device, dataset_path, dataset_name, feature_modal, feature_size, 
                sample_size, lambdas, alpha, max_epoch, save_every, seed):
    """
    @summary: generate command for the execution
    @param device: cpu, 0
    """
    command = f"""
    python main.py \
        --device {device} \
        --dataset_path {dataset_path} \
        --dataset_name {dataset_name} \
        --feature_modal {feature_modal} \
        --feature_size {feature_size} \
        --sample_size {sample_size} \
        --max_epoch {max_epoch}  \
        --save_every {save_every} \
        --seed {seed}
    """
    return command

length_feature_modals = {
    "rgb": 1024,
    "flow": 1024,
    "conbine": 2048,
}

if __name__ == "__main__": 
    base_path = "/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-relabeled-i3d-featconc-milbkbn-wan/Pets"
    num_overlapping_cameras = 2 # change this to the desired number of overlapping cameras
    selected_feature_modal = "rgb"

    

    for seed in range(1):
        for sample_siz in [2,]:
            for ds_name in itertools.combinations(["001", "002", "003", "004"], num_overlapping_cameras):
                ds_name=str(ds_name).replace("'","").replace("(","").replace(")","").replace(" ","").replace(",","-")
                print("execution", seed, sample_siz, ds_name)
                cmd = get_command(device="0", dataset_path=base_path, dataset_name=ds_name, 
                                feature_modal=selected_feature_modal, 
                                feature_size=length_feature_modals[selected_feature_modal]*num_overlapping_cameras, 
                                sample_size=sample_siz, lambdas=None, 
                                alpha = None, max_epoch=100, save_every=100, seed = seed)
                print(ds_name)
                print(cmd)
                os.system(cmd)
