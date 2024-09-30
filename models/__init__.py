import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
sys.path.append('.')

import logging
import torch

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def pretrain_load_simple(
        model, 
        resume_checkpoint=None,
        **kwargs
    ):

    if not resume_checkpoint:
        print(f'resume_checkpoint is none, nothing loaded!', flush=True)
        return model

    state_dict = torch.load(resume_checkpoint, map_location='cpu')
    #TODO: check
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'ema' in state_dict:
        state_dict = state_dict['ema']
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if 'x_embedder.proj.weight' in state_dict and state_dict["x_embedder.proj.weight"].ndim == 4:
        state_dict["x_embedder.proj.weight"] = state_dict["x_embedder.proj.weight"].unsqueeze(2)
    
    # [1] load model
    keys_loaded = {}
    try:
        ret = model.load_state_dict(state_dict, strict=True)
        logging.info(f'load a fixed model with {ret}')
    except:
        model_dict = model.state_dict()
        key_list = list(state_dict.keys())
        for skey, item in state_dict.items():
            if skey not in model_dict:
                logging.info(f'Skip {skey}')
                continue
            if item.shape != model_dict[skey].shape:
                logging.info(f'Skip {skey} with different shape {item.shape} {model_dict[skey].shape}')
                continue
            model_dict[skey].copy_(item)
            keys_loaded[skey] = 1
        model.load_state_dict(model_dict)
    
    logging.info(f'Successfully load model from {resume_checkpoint}')
    # print(f'Successfully load model from {resume_checkpoint}')
    return model
    
def get_models(args):
    if 'HawkT2V' in args.model:
        from models.hawk_t2v import HawkT2V

        hawk_model = HawkT2V(**args.HawkT2V)

        if not args.checkpoint and not args.pretrained_model_path:
            print('No pretrained model used!')
            return hawk_model
        if not args.checkpoint:
            pretained_model_path = os.path.join(args.pretrained_model_path, 'transformer_pt/hawk_t2v.pt')
        else:
            pretained_model_path = args.checkpoint
        print("pretrained model path: ", pretained_model_path)

        return pretrain_load_simple(hawk_model, pretained_model_path)
    else:
        raise '{} Model Not Supported!'.format(args.model)
    