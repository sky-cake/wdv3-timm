# wdv3-timm

```bash
# input
git clone https://github.com/neggles/wdv3-timm.git
cd wd3-timm
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

python3.12 wdv3_timm.py --model <vit|swinv2|convnext> /path/to/image/or/image/dir
```

```bash
# output
Processing /path/to/image...
Image: /path/to/image
Caption: necktie, sunglasses, open_mouth, black_necktie, multiple_boys, black_background, male_focus, no_humans, simple_background, 3boys, shirt, white_shirt, suit, formal, collared_shirt, teeth, jacket, facial_hair, bald, black_jacket
Ratings: {'general': 0.85963, 'sensitive': 0.17064428, 'questionable': 0.0015756089, 'explicit': 0.0007784116}
Character tags: {}
General tags: {'open_mouth': 0.8662723, 'simple_background': 0.7269018, 'shirt': 0.7035996, 'jacket': 0.5113874, 'white_shirt': 0.6908853, 'male_focus': 0.7487514, 'teeth': 0.5238094, 'multiple_boys': 0.853061, 'necktie': 0.9106916, 'collared_shirt': 0.5765164, 'black_jacket': 0.37321326, 'no_humans': 0.7401722, 'facial_hair': 0.44881666, 'sunglasses': 0.8827623, 'formal': 0.5835051, 'suit': 0.65646625, '3boys': 0.7264765, 'black_background': 0.85161805, 'black_necktie': 0.8537366, 'bald': 0.43482396}
```
