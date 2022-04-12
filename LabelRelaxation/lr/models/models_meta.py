from enum import Enum
import json
import copy

#from lr.models.densenet import custDenseNet121_2
from lr.models.resnet import resnet_v2, get_resnet18
#from lr.models.simple import get_simple_dense
from lr.models.vgg import get_vgg16, get_vgg3, get_vgg11
from lr.models.convnet import get_convnet, get_deconvnet
from lr.models.mobilenet import get_mobilenet


class StringEnum(Enum):
    def __str__(self):
        return str(self.value)


class ModelType(StringEnum):
    VGG16 = "VGG16"
    VGG3 = "VGG3"
    VGG11 = "VGG11"
    RESNET56_V2 = "ResNet56V2"
    RESNET18 = "ResNet18"
    RESNET50 = "ResNet50"
    CONVNET = "ConvNet"
    DECONVNET = "DeconvNet"
    MOBILENET = "MobileNet"
    DENSENET_BC = "DenseNetBC"
    SIMPLE_DENSE = "simple_dense"


def get_model_type_by_name(model_name):
    if model_name in ["resnet", "ResNet56V2"]:
        return ModelType.RESNET56_V2
    elif model_name in ["ResNet50"]:
        return ModelType.RESNET50
    elif model_name in ["ResNet18"]:
        return ModelType.RESNET18
    elif model_name in ["vgg", "VGG16"]:
        return ModelType.VGG16
    elif model_name in ["VGG3"]:
        return ModelType.VGG3
    elif model_name in ["VGG11"]:
        return ModelType.VGG11
    elif model_name in ["ConvNet"]:
        return ModelType.CONVNET
    elif model_name in ["DeconvNet"]:
        return ModelType.DECONVNET
    elif model_name in ["MobileNet"]:
        return ModelType.MOBILENET
    elif model_name in ["densenet", "DenseNetBC"]:
        return ModelType.DENSENET_BC
    elif model_name == "simple_dense":
        return ModelType.SIMPLE_DENSE
    else:
        raise ValueError("Unknown model name: {}".format(model_name))


def get_backbone_model_fn_by_type(model_type):
    if model_type == ModelType.RESNET56_V2:
        return resnet_v2
    elif model_type == ModelType.RESNET50:
        return resnet_v2
    elif model_type == ModelType.RESNET18:
        return get_resnet18
    elif model_type == ModelType.VGG16:
        return get_vgg16
    elif model_type == ModelType.VGG3:
        return get_vgg3
    elif model_type == ModelType.VGG11:
        return get_vgg11
    elif model_type == ModelType.CONVNET:
        return get_convnet
    elif model_type == ModelType.DECONVNET:
        return get_deconvnet
    elif model_type == ModelType.MOBILENET:
        return get_mobilenet
    elif model_type == ModelType.DENSENET_BC:
        return custDenseNet121_2
    elif model_type == ModelType.SIMPLE_DENSE:
        return get_simple_dense
    else:
        raise ValueError("Unknown model type: {}".format(model_type))


class ModelParameters(object):
    def __init__(self):
        self.parameters = {}

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name, default=None):
        if name in self.parameters:
            return self.parameters[name]
        else:
            return default

    def get_parameter_string(self):
        output_str = ""
        for key in self.parameters:
            if output_str != "":
                output_str += "_"
            output_str += str(key) + "_" + str(self.parameters[key])
        return output_str

    def load_parameters_from_file(self, json_file_path, key, exclude_keys=None):
        with open(json_file_path) as ssfile:
            ext_params = json.load(ssfile)

        if key not in ext_params:
            raise ValueError(
                'Could not find entry for key {} in external parameter file {}.'.format(key, json_file_path))
        else:
            for param_key in ext_params[key]:
                if exclude_keys is not None and param_key in exclude_keys:
                    continue
                value = ext_params[key][param_key]
                if isinstance(value, str):
                    value = value == "True" or value == "true"
                self.set_parameter(param_key, value)

    def duplicate(self):
        result = ModelParameters()
        result.parameters = copy.deepcopy(self.parameters)
        return result
