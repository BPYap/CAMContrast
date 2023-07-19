## Environment setup

```
python -m virtualenv -p 3.9 env
source env/bin/activate

pip install -r requirements.txt
python setup.py install
```

## Datasets

| Dataset      | Task                                                | Link                                                                                                                                                                                       |
|--------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OIA-ODIR     | Fundus pretraining                                  | [original](https://github.com/nkicsl/OIA-ODIR)<br/>[preprocessed](https://drive.google.com/file/d/1g8GlnzAxIjqgZYctMMkbY5uMNoRyKx0M/view?usp=sharing)                                      |
| ChestX-ray14 | X-ray pretraining<br/>Thorax disease classification | [original](https://nihcc.app.box.com/v/ChestXray-NIHCC)<br/>[preprocessed](https://drive.google.com/file/d/1W4LZzGd-IfPkFVgR_6HybwuZxam1w9s-/view?usp=sharing)                             |
| IDRiD        | DR-DME classification<br/>Lesions segmentation      | [source](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)                                                                                            |
| REFUGE       | Glaucoma classification<br/>Disc/cup segmentation   | [source](https://refuge.grand-challenge.org/)                                                                                                                                              |
| Vessel-seg   | Vessel segmentation                                 | [DRIVE](https://drive.grand-challenge.org/DRIVE/)<br/>[STARE](https://cecas.clemson.edu/~ahoover/stare/probing/index.html)<br/>[CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) |
| SIIM-ACR     | Pneumothorax segmentation                           | [source](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation)                                                                                                           |

## Pretrained models
- [CAMContrast-Fundus](https://drive.google.com/drive/folders/1i6UIAl8cc9V-21AfK5Yo5HKco2amKUT8?usp=sharing)
- [CAMContrast-X-ray](https://drive.google.com/drive/folders/1OEwRyjEQNuZDTjtk8ZJesoK6GLi8CPIk?usp=sharing)

## Scripts

- preprocessing and pretraining: `script/upstream`
- transfer learning: `script/downstream`
- shell scripts with commands for reproducing the experimental results: `example`, e.g., execute `example/upstream/chestx-ray14/0_preprocess.sh` to run the preprocessing step

## Acknowledgement

The implementation of contrastive learning loss was adapted from
the [SupContrast repository](https://github.com/HobbitLong/SupContrast). 
