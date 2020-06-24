from .affinity import ea, x2p
from .error import error_ee, error_ee_split
from .linearsearch import ls_ee
from .visualize import scatter, scatter_with_colorbar
from .embeddingsLoss import eloss

__all__ = ['ea',
           'x2p',
           'error_ee',
           'error_ee_split',
           'ls_ee',
           'scatter',
           'scatter_with_colorbar',
           'eloss']
