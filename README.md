# Image-Physics-Simulation
This repo implements a simple 2D Ray-Tracing with.


### Installation

```bash
conda create -n img-phy-sim python=3.11 pip -y
conda activate img-phy-sim
pip install numpy matplotlib opencv-python ipython jupyter shapely prime_printer datasets==3.6.0 scikit-image
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```


### Download Example Data

```bash
conda activate img-phy-sim
cd "D:\Informatik\Projekte\Image-Physics-Simulation" && D:
python physgen_dataset.py --output_real_path ./datasets/physgen_train_raw/real --output_osm_path ./datasets/physgen_train_raw/osm --variation sound_reflection --input_type osm --output_type standard --data_mode train
python physgen_dataset.py --output_real_path ./datasets/physgen_test_raw/real --output_osm_path ./datasets/physgen_test_raw/osm --variation sound_reflection --input_type osm --output_type standard --data_mode test
python physgen_dataset.py --output_real_path ./datasets/physgen_val_raw/real --output_osm_path ./datasets/physgen_val_raw/osm --variation sound_reflection --input_type osm --output_type standard --data_mode validation
```

