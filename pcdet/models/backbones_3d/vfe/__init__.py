from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .image_vfe import ImageVFE
from .image_point_vfe import ImagePointVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'ImagePointVFE': ImagePointVFE
}
