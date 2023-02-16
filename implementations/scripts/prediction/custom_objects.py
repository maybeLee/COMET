def custom_objects(mode="custom"):

    def no_activation(x):
        return x

    def leakyrelu(x):
        import keras.backend as K
        return K.relu(x, alpha=0.01)

    objects = {}
    objects['no_activation'] = no_activation
    objects['leakyrelu'] = leakyrelu
    if mode == "custom":
        from scripts.generation.custom_layers import CustomPadLayer, CustomCropLayer, CustomDropDimLayer, CustomExpandLayer, CustomCastLayer
        objects['CustomPadLayer'] = CustomPadLayer
        objects['CustomCropLayer'] = CustomCropLayer
        objects['CustomDropDimLayer'] = CustomDropDimLayer
        objects['CustomExpandLayer'] = CustomExpandLayer
        objects['CustomCastLayer'] = CustomCastLayer

    return objects
