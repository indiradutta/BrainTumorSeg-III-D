## Model Architecture
![pspnet3d_arch](https://user-images.githubusercontent.com/66861243/132329738-550c49a7-df75-447c-ad8d-ff8e6f598ba1.png)

## Model Initialization
The PSPNet model may be initialized and the state dict may be viewed by running the following code snippet:

```python
from pspnet3D import PSPNet

net = PSPNet()
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
```
