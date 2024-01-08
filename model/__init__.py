from .net import ModelBase
from .cnn import CNNPOS
from .others import Identity, ParamNet
from .code import *

net_dict = {
    'codeposenet': CodePoseNet,
    'codenetkitti': CodeNetKITTI,
    'iden': Identity,
    'cnnpos': CNNPOS,
    'codenet': CodeNet,
    'param': ParamNet,
}
