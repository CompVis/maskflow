# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE

try:
    from dgm_eval.dgm_eval.representations import get_representations
    from dgm_eval.dgm_eval.metrics.mmd import compute_mmd
    from dgm_eval.dgm_eval.metrics.fd import compute_FD_with_reps
    from dgm_eval.dgm_eval.metrics.inception_score import calculate_score
    from dgm_eval.dgm_eval.metrics.prdc import compute_prdc
    from dgm_eval.dgm_eval.metrics.fls import compute_fls
    from dgm_eval.dgm_eval.models import load_encoder
except:
    from utils.dgm_eval.dgm_eval.representations import get_representations
    from utils.dgm_eval.dgm_eval.metrics.mmd import compute_mmd
    from utils.dgm_eval.dgm_eval.metrics.fd import compute_FD_with_reps
    from utils.dgm_eval.dgm_eval.metrics.inception_score import calculate_score
    from utils.dgm_eval.dgm_eval.metrics.prdc import compute_prdc
    from utils.dgm_eval.dgm_eval.metrics.fls import compute_fls
    from utils.dgm_eval.dgm_eval.models import load_encoder


import torch
from torch.utils.data import DataLoader, TensorDataset

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def remove_transform(
    compose: transforms.Compose, transform_type: type
) -> transforms.Compose:
    """
    Remove a specific transform from a transforms.Compose object.

    Args:
        compose (transforms.Compose): The original Compose object.
        transform_type (type): The type of the transform to remove.

    Returns:
        transforms.Compose: A new Compose object with the specified transform removed.
    """
    # Convert the Compose object to a list of transforms
    transform_list = list(compose.transforms)

    # Remove the transform of the specified type
    transform_list = [t for t in transform_list if not isinstance(t, transform_type)]

    # Convert the list back to a Compose object
    new_compose = transforms.Compose(transform_list)

    return new_compose


class CustomTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image


class DataLoaderWrapper:
    def __init__(self, dataloader: DataLoader, nimages: int):
        self.data_loader = dataloader
        self.nimages = nimages


def create_dataloader(
    images: torch.Tensor, batch_size: int, shuffle: bool = True, transform=None
) -> DataLoaderWrapper:
    """
    Create a DataLoader from a tensor of images.

    Args:
        images (torch.Tensor): Tensor of images with shape (batch_size, c, w, h).
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data. Default is True.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        DataLoaderWrapper: A wrapper object containing the DataLoader.
    """
    # Create a custom dataset from the images
    dataset = CustomTensorDataset(images, transform=transform)

    # Create a DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Wrap the DataLoader in the custom class
    output = DataLoaderWrapper(dataloader, nimages=images.shape[0])
    return output


class DinoV2_Metric(Metric):
    r"""
    Calculates Kernel Inception Distance (KID) which is used to access the quality of generated images. Given by

    .. math::
        KID = MMD(f_{real}, f_{fake})^2

    where :math:`MMD` is the maximum mean discrepancy and :math:`I_{real}, I_{fake}` are extracted features
    from real and fake images, see [1] for more details. In particular, calculating the MMD requires the
    evaluation of a polynomial kernel function :math:`k`

    .. math::
        k(x,y) = (\gamma * x^T y + coef)^{degree}

    which controls the distance between two features. In practise the MMD is calculated over a number of
    subsets to be able to both get the mean and standard deviation of KID.

    Using the default feature extraction (Inception v3 using the original weights from [2]), the input is
    expected to be mini-batches of 3-channel RGB images of shape (3 x H x W) with dtype uint8. All images
    will be resized to 299 x 299 which is the size of the original training data.

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or
        ``pip install torch-fidelity``

    .. note:: the ``forward`` method can be used but ``compute_on_step`` is disabled by default (oppesit of
        all other metrics) as this metric does not really make sense to calculate on a single batch. This
        means that by default ``forward`` will just call ``update`` underneat.

    Args:
        feature: Either an str, integer or ``nn.Module``:

            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``[N,d]`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        subsets: Number of subsets to calculate the mean and standard deviation scores over
        subset_size: Number of randomly picked samples in each subset
        degree: Degree of the polynomial kernel function
        gamma: Scale-length of polynomial kernel. If set to ``None`` will be automatically set to the feature size
        coef: Bias term in the polynomial kernel.
        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.

            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    References:
        [1] Demystifying MMD GANs
        Mikołaj Bińkowski, Danica J. Sutherland, Michael Arbel, Arthur Gretton
        https://arxiv.org/abs/1801.01401

        [2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,
        Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter
        https://arxiv.org/abs/1706.08500

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in ``[64, 192, 768, 2048]``
        ValueError:
            If ``subsets`` is not an integer larger than 0
        ValueError:
            If ``subset_size`` is not an integer larger than 0
        ValueError:
            If ``degree`` is not an integer larger than 0
        ValueError:
            If ``gamma`` is niether ``None`` or a float larger than 0
        ValueError:
            If ``coef`` is not an float larger than 0
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.kid import KernelInceptionDistance
        >>> kid = KernelInceptionDistance(subset_size=50)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> kid.update(imgs_dist1, real=True)
        >>> kid.update(imgs_dist2, real=False)
        >>> kid_mean, kid_std = kid.compute()
        >>> print((kid_mean, kid_std))
        (tensor(0.0337), tensor(0.0023))
    """

    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(
        self,
        feature: Union[str, int, torch.nn.Module] = 2048,
        #### KID related
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,  # type: ignore
        coef: float = 1.0,
        ######
        reset_real_features: bool = True,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `Kernel Inception Distance` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "Kernel Inception Distance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input},"
                    f" but got {feature}."
                )

            self.inception: Module = NoTrainInceptionV3(
                name="inception-v3-compat", features_list=[str(feature)]
            )
        elif isinstance(feature, Module):
            self.inception = feature
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not (isinstance(subsets, int) and subsets > 0):
            raise ValueError("Argument `subsets` expected to be integer larger than 0")
        self.subsets = subsets

        if not (isinstance(subset_size, int) and subset_size > 0):
            raise ValueError(
                "Argument `subset_size` expected to be integer larger than 0"
            )
        self.subset_size = subset_size

        if not (isinstance(degree, int) and degree > 0):
            raise ValueError("Argument `degree` expected to be integer larger than 0")
        self.degree = degree

        if gamma is not None and not (isinstance(gamma, float) and gamma > 0):
            raise ValueError(
                "Argument `gamma` expected to be `None` or float larger than 0"
            )
        self.gamma = gamma

        if not (isinstance(coef, float) and coef > 0):
            raise ValueError("Argument `coef` expected to be float larger than 0")
        self.coef = coef

        if not isinstance(reset_real_features, bool):
            raise ValueError("Arugment `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

        self.model = load_encoder(
            "dinov2",
            self.device,
            ckpt=None,
            arch=None,
            clean_resize=False,
            sinception=False,
            depth=0,
        )

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        """Update the state with extracted features.

        Args:
            imgs: tensor with images feed to the feature extractor
            real: bool indicating if ``imgs`` belong to the real or the fake distribution
        """
        # Create a DataLoader with the appropriate transform
        dl = create_dataloader(
            imgs, batch_size=32, shuffle=False, transform=self.model.transform
        )

        # Extract features using the model
        features = get_representations(self.model, dl, device=self.device)  # [B,C]

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        real_features = np.concatenate(self.real_features, axis=0)
        fake_features = np.concatenate(self.fake_features, axis=0)
        result_dict = dict()
        kid_scores = compute_mmd(
            real_features,
            fake_features,
            n_subsets=self.subsets,
            subset_size=self.subset_size,
        )
        result_dict["kid"] = kid_scores.mean()
        result_dict["kid_std"] = kid_scores.std()
        fid_scores = compute_FD_with_reps(
            real_features,
            fake_features,
        )
        result_dict["fid"] = fid_scores

        is_scores = calculate_score(
            fake_features, splits=10, N=len(fake_features), shuffle=True, rng_seed=2020
        )
        result_dict["is"] = is_scores[0]

        prdc_scores = compute_prdc(
            real_features, fake_features, nearest_k=5, realism=False
        )
        result_dict.update(prdc_scores)

        result_dict = {"dinov2_" + k: v.item() for k, v in result_dict.items()}
        return result_dict

    def reset(self) -> None:
        if not self.reset_real_features:
            # remove temporarily to avoid resetting
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()


if __name__ == "__main__":
    dinov2_metric = DinoV2_Metric(subset_size=50)

    # Generate two slightly overlapping image intensity distributions
    # Here we generate 100 real images and 100 fake images of size 3x299x299 with values in the range [0, 255]
    imgs_real = torch.randint(0, 200, (100, 3, 256, 256), dtype=torch.uint8)
    imgs_fake = torch.randint(100, 255, (100, 3, 256, 256), dtype=torch.uint8)

    # Update the metric with the generated real images
    dinov2_metric.update(imgs_real, real=True)

    # Update the metric with the generated fake images
    dinov2_metric.update(imgs_fake, real=False)
    dinov2_metric.update(imgs_fake, real=False)
    dinov2_metric.update(imgs_real, real=True)
    # Compute the Kernel Inception Distance
    result = dinov2_metric.compute()
    print(result)
