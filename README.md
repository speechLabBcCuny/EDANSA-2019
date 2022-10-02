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
python setup.py install develop

# Download the data
wget -O EDANSA-2019.zip  https://zenodo.org/record/6824272/files/EDANSA-2019.zip?download=1
unzip assets/EDANSA-2019.zip -d assets/

# launch training
python runs/augment/run.py
```

## Configs
Configs file is in runs/augment/runconfigs.py, you can change the parameters there.