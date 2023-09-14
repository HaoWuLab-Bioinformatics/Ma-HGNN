from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class BlogCatalog(BaseData):
    r"""The BlogCatalog dataset is a social network dataset for vertex classification task. 
    This is a network of social relationships of bloggers from the BlogCatalog website, 
    where nodes' attributes are constructed by keywords, which are generated by users as a short description of their blogs. 
    The labels represent the topic categories provided by the authors.
    
    .. note:: 
        The L1-normalization for the feature is not recommended for this dataset.

    The content of the BlogCatalog dataset includes the following:

    - ``num_classes``: The number of classes: :math:`6`.
    - ``num_vertices``: The number of vertices: :math:`5,196`.
    - ``num_edges``: The number of edges: :math:`343,486`.
    - ``dim_features``: The dimension of features: :math:`8,189`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(5,196 \times 8,189)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(343,486 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(5,196, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("BlogCatalog", data_root)
        self._content = {
            "num_classes": 6,
            "num_vertices": 5196,
            "num_edges": 171743,
            "dim_features": 8189,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "ecdd26c63f483c4d919a156f9c8e92fc"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor],  # partial(norm_ft, ord=1)
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "03ffbc8c9a4d9abeab0f127c717888f0"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "246e7096dd834a631c33fe0c7afb89b4"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
