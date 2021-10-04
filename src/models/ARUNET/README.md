## Model Architecture
![arunet](https://user-images.githubusercontent.com/66861243/135914032-7d3c4562-35a3-4bdc-acf5-dbff54cda413.png)



## Model Initialization
The ARUNET3D model may be initialized and the state dict may be viewed by running the following code snippet:

```python
from arunet3D import Block, Attention, ARUNET3D

net = ARUNET3D(Block, Attention, 3, [64,128,256,512])
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
```
