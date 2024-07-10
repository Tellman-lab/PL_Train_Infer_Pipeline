from PL_Support_Codes.models.lf_model import LateFusionModel
from PL_Support_Codes.models.ef_model import EarlyFusionModel
from PL_Support_Codes.models.water_seg_model import WaterSegmentationModel

MODELS = {
    'ms_model': WaterSegmentationModel,
    'ef_model': EarlyFusionModel,
    'lf_model': LateFusionModel
}


def build_model(model_name, input_channels, n_classes, lr, log_image_iter,
                to_rgb_fcn, ignore_index, model_used, model_loss_fn_a, model_loss_fn_b,  model_loss_fn_a_ratio, model_loss_fn_b_ratio, optimizer_name, **kwargs):
    try:
        print('!!!!!!!')
        print('!!!!!!!')
        print('!!!!!!!')
        print('!!!!!!!')
        print('optimizer_name:', optimizer_name)
        print('!!!!!!!')
        print('!!!!!!!')
        print('!!!!!!!')
        print(MODELS[model_name])

        print('!!!!!!!')
        print('!!!!!!!')

        model = MODELS[model_name](input_channels, n_classes, lr,
                                   log_image_iter, to_rgb_fcn, ignore_index, model_used, model_loss_fn_a, model_loss_fn_b, model_loss_fn_a_ratio, model_loss_fn_b_ratio,
                                #    model_loss_fn_a, model_loss_fn_b, model_loss_fn_a_ratio,model_loss_fn_b_ratio, 
                                   optimizer_name, **kwargs)
    except KeyError:
        print(f'Could not find model named: {model_name}')

    return model
