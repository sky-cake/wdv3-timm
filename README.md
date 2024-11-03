# wdv3-timm

small example thing showing how to use `timm` to run the WD Tagger V3 models.

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/neggles/wdv3-timm.git
cd wd3-timm
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.10 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install PyTorch manually (e.g. if you are not using an nVidia GPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the example script, picking one of the 3 models to use:
```sh
python3.12 wdv3_timm.py --model <vit|swinv2|convnext> /path/to/image/or/image/dir
```

Example output from `python3.12 wdv3_timm.py /path/to/image`:
```sh
Processing /path/to/image...
Image: /path/to/image
Caption: necktie, sunglasses, open_mouth, black_necktie, multiple_boys, black_background, male_focus, no_humans, simple_background, 3boys, shirt, white_shirt, suit, formal, collared_shirt, teeth, jacket, facial_hair, bald, black_jacket
Ratings: {'general': 0.85963, 'sensitive': 0.17064428, 'questionable': 0.0015756089, 'explicit': 0.0007784116}
Character tags: {}
General tags: {'open_mouth': 0.8662723, 'simple_background': 0.7269018, 'shirt': 0.7035996, 'jacket': 0.5113874, 'white_shirt': 0.6908853, 'male_focus': 0.7487514, 'teeth': 0.5238094, 'multiple_boys': 0.853061, 'necktie': 0.9106916, 'collared_shirt': 0.5765164, 'black_jacket': 0.37321326, 'no_humans': 0.7401722, 'facial_hair': 0.44881666, 'sunglasses': 0.8827623, 'formal': 0.5835051, 'suit': 0.65646625, '3boys': 0.7264765, 'black_background': 0.85161805, 'black_necktie': 0.8537366, 'bald': 0.43482396}

```
