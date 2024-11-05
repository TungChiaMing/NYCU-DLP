## DataSet

- LAB4_Dataset: https://drive.google.com/file/d/1eDlWSbfkZrUzA-TItgbLPTmvjWc8kXS9/view?usp=sharing
- Demo_Test: https://drive.google.com/file/d/134IUQ4ATMf7UJlR35vgpCb1UegWXqwA1/view?usp=sharing
## Usage

### For training
```
python Trainer.py --DR ./LAB4_Dataset --save_root ./result --fast_train
```
### For testing
```
python Tester.py --DR ./LAB4_Dataset --save_root ./result --ckpt_path ./result/epoch=45.ckpt
```