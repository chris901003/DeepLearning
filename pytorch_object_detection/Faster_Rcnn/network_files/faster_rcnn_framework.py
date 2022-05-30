import torch
from torch import nn
from collections import OrderedDict
from network_files.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from network_files.roi_head import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import torch.nn.functional as F
import warnings
from network_files.transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])

        if self.training and targets is None:
            raise ValueError('In training mode, targets should be passed.')
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError('Expected target boxes to be a tensor of shape [N, 4]')
                else:
                    raise ValueError('Boxes shape error.')
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        features = self.backbone(images, targets)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_size, targets)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
