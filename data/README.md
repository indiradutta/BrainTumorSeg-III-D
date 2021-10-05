## Data Preprocessing

- Loading the pickle file containing the list of image names:

```python
import pickle

with open('imgs.pkl', 'rb') as f:
    directory = pickle.load(f)
print(len(directory))
```
- Calling the Preprocessing Class and creating the Dataloader object:

```python
from preprocessing import Preprocessing
from torch.utils.data import Dataset, DataLoader

root = <'path/to/MICCAI_BraTS2020_TrainingData/'>
data = Preprocessing(root, l1=d)
dataloader = DataLoader(data, batch_size=<batch_size>, shuffle=True)
```

## Data Preprocessing for inference

- Calling the Infer_Preprocess Class and mod_preprocessing the image:

```python
from infer_preprocess import Infer_Preprocess

img = Infer_Preprocess.mod_preprocess(1, <nib loaded image>, <image dimension as a tuple>)
```
