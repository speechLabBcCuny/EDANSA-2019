# EDANSA-2019
The Ecoacoustic Dataset from Arctic North Slope Alaska


## How to run training

```bash
# Clone the repository
mkdir EDANSA
cd EDANSA
git init
git pull https://github.com/speechLabBcCuny/EDANSA-2019
# Install dependencies
python3 -m pip install -e ./ 

# Download the data
wget -O EDANSA-2019.zip  https://zenodo.org/record/6824272/files/EDANSA-2019.zip?download=1
unzip EDANSA-2019.zip -d assets/

# launch training
python runs/augment/run.py
```

## Configs
Configs file is in runs/augment/runconfigs.py, you can change the parameters there.

## Citation
If you use the code from this repository or the dataset,
you are kindly asked to cite the paper for which the information will be
provided here upon publication in November 2022.

```
@inproceedings{Coban2022,
    author = "Çoban, Enis Berk and Perra, Megan and Pir, Dara and Mandel, Michael I",
    title = "EDANSA-2019: The Ecoacoustic Dataset from Arctic North Slope Alaska",
    booktitle = "Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022)",
    address = "Nancy, France",
    month = "November",
    year = "2022",
    abstract = "The arctic is warming at three times the rate of the global average, affecting the habitat and lifecycles of migratory species that reproduce there, like birds and caribou. Ecoacoustic monitoring can help efficiently track changes in animal phenology and behavior over large areas so that the impacts of climate change on these species can be better understood and potentially mitigated. We introduce here the Ecoacoustic Dataset from Arctic North Slope Alaska (EDANSA-2019), a dataset collected by a network of 100 autonomous recording units covering an area of 9000 square miles over the course of the 2019 summer season on the North Slope of Alaska and neighboring regions. We labeled over 27 hours of this dataset according to 28 tags with enough instances of 9 important environmental classes to train baseline convolutional recognizers. We are releasing this dataset and the corresponding baseline to the community to accelerate the recognition of these sounds and facilitate automated analyses of large-scale ecoacoustic databases."
}
```