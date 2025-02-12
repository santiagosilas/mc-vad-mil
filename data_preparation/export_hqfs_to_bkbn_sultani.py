import os
import random
import joblib
import numpy as np
import munch
from collections import Counter
from IPython.display import clear_output, HTML
from collections import Counter


splits_folder = base_path = f"/media/dev/LaCie/MC-VAD-MIL-DATA/center-crop/hqfs-relabeled-i3d/"
output_folder = f'/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-sultani-format'

splits_folder = base_path = f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-2cams"
output_folder = f'/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-2cams-sultani-format'

splits_folder = base_path = f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-3cams"
output_folder = f'/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-3cams-sultani-format'

splits_folder = base_path = f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-4cams"
output_folder = f'/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-4cams-sultani-format'

splits_folder = base_path = f"/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-5cams"
output_folder = f'/media/dev/LaCie/MC-VAD-MIL-OUT/hqfs-relabeled-i3d-featconc-5cams-sultani-format'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


inputs_outputs = list()
for f in os.listdir(splits_folder): 
    if "-train" in f:
        input_file = os.path.join(splits_folder, f)
        output_dir = os.path.join(output_folder, f.replace(".npy", "-ekos-input"))
        inputs_outputs.append([input_file, output_dir])
        print(input_file)
        print(output_dir)
        print()

def generate_ekos_input(input_file, output_dir):
    
    train_bags = np.load(os.path.join(splits_folder, input_file), allow_pickle=True)
    train_bags = munch.munchify(train_bags)
    
    dst_path = output_dir

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        os.mkdir(os.path.join(dst_path, 'Normal'))
        os.mkdir(os.path.join(dst_path, 'Anomaly'))

    # Salva Segmentos nas Pastas Normal e Anomaly
    for count, bag in enumerate(train_bags):
        bag = munch.munchify(bag)
        subfolder = 'Normal' if bag.y_i == 0.0 else 'Anomaly'
        filename = f'{bag.name.replace(".","-")}.txt'
        print(count, filename, end=',')
        with open(os.path.join(dst_path, os.path.join(subfolder, filename)), "w") as fp:
            for x_clip in train_bags[count]["X_i"]:
                fp.write(' '.join([str(x) for x in x_clip]) + "\n")

    clear_output()
    # Gera lista (Arquivo TXT) com os dados de treinamento
    with open(os.path.join(dst_path, 'train-annot.txt'), 'w') as f:
        for i, bag in enumerate(train_bags):
            bag = munch.munchify(bag)
            subfolder = 'Normal' if bag.y_i == 0.0 else 'Anomaly'
            filename = f'{bag.name.replace(".","-")}.mp4'
            if bag.y_i == 0.0:
                content = f'{subfolder+"/"+filename} X'
            else:
                content = f'{subfolder+"/"+filename} X'
            end = '\n' if i < len(train_bags)-1 else ''
            f.write(content  + end)

for input_file, output_dir in inputs_outputs:
    generate_ekos_input(input_file, output_dir)