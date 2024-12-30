# 分割网络

> 设计用于脑肿瘤分割的网络

- [ ] Unet
- [ ] Uent + EMA
- [ ] transUnet

## Q&A

### 1. 继承nn.Module的作用

> `nn.Module` 是一个非常基本的类, 用于定义神经网络的各个层以及这些层的前向传递方式。几乎所有的神经网络模块都直接或间接地继承自 `nn.Module`.
>
> 主要有以下作用：
>
> - 封装模型层
>
>   > 通过继承 `nn.Module` ，可以方便将模型的各个层封装在一起，形成完整的模型
>   >
> - 参数管理
>
>   > `nn.Module`提供自动管理模型参数的方法，被添加的层的参数会被自动注册吗，通过 `model.parameters()`方法获取到，这对优化器很重要。
>   >
> - 前向传播
>
>   > `nn.Module`要求定义forward方法
>   >
> - 设备转移
>
>   > 方便通过 `model.to(device)`将模型及其参数移动到GPU或CPU上
>   >
> - 保存和加载模型
>
>   > `nn.Module` 提供了方便的方法来保存和加载模型的状态字典（state_dict）
>   >

### 2. 怎么读取NII数据

> 要读取NIfTI（Neuroimaging Informatics Technology Initiative）文件，您可以使用Python中的 `nibabel`库。NIfTI是一种常用的医学图像格式，用于存储MRI、CT等医学图像数据。以下是一个简单的示例代码，展示了如何使用 `nibabel`库读取NIfTI文件


```python
import nibabel as nib

def load_nii_file(file_path):
    """
    读取NIfTI文件并返回图像数据和头信息。

    参数:
    - file_path (str): NIfTI文件的路径。

    返回:
    - image_data (numpy.ndarray): 图像数据。
    - header (nibabel.nifti1.Nifti1Header): NIfTI文件的头信息。
    """
    # 使用nibabel加载NIfTI文件
    nii_image = nib.load(file_path)
    # 获取图像数据
    image_data = nii_image.get_fdata()
    # 获取头信息
    header = nii_image.header
    return image_data, header

# 示例用法
file_path = "path/to/your/nii/file.nii.gz"
image_data, header = load_nii_file(file_path)

print("Image data shape:", image_data.shape)
print("Header information:", header)

```
