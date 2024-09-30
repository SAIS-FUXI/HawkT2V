# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script.
"""


import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
import io
import os
import math
import argparse
import imageio
# import wandb
import html
from typing import Callable, List, Optional, Tuple, Union
import inspect
import re
import urllib.parse as ul
import gc
import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.utils.data.distributed import DistributedSampler

from transformers import T5EncoderModel, T5Tokenizer
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir)
from utils import save_video_grid

from diffusers.utils import (
    BACKENDS_MAPPING,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)

if is_bs4_available():
    from bs4 import BeautifulSoup
if is_ftfy_available():
    import ftfy

from models.video_vae import CausualVAEVideo
from models.text_encoder import T5Encoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

bad_punct_regex = re.compile(
    r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
)  # noqa
import random 
# Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
def _text_preprocessing(text, clean_caption=False):
    if clean_caption and not is_bs4_available():
        logger.warn(BACKENDS_MAPPING["bs4"][-1].format("Setting `clean_caption=True`"))
        logger.warn("Setting `clean_caption` to False...")
        clean_caption = False

    if clean_caption and not is_ftfy_available():
        logger.warn(BACKENDS_MAPPING["ftfy"][-1].format("Setting `clean_caption=True`"))
        logger.warn("Setting `clean_caption` to False...")
        clean_caption = False

    if not isinstance(text, (tuple, list)):
        text = [text]

    def process(text: str):
        if clean_caption:
            text = _clean_caption(text)
            text = _clean_caption(text)
        else:
            text = text.lower().strip()
        return text

    return [process(t) for t in text]


# Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
def _clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    # urls:
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r"@[\w\d]+\b", "", caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
        "-",
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)

    # &quot;
    caption = re.sub(r"&quot;?", "", caption)
    # &amp
    caption = re.sub(r"&amp", "", caption)

    # ip adresses:
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

    # article ids:
    caption = re.sub(r"\d:\d\d\s+$", "", caption)

    # \n
    caption = re.sub(r"\\n", " ", caption)

    # "#123"
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    # "#12345.."
    caption = re.sub(r"#\d{5,}\b", "", caption)
    # "123456.."
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()

# Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
def encode_prompt(
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        clean_caption: bool = False,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
            instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
            PixArt-Alpha, this should be "".
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            whether to use classifier free guidance or not
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            number of images that should be generated per prompt
        device: (`torch.device`, *optional*):
            torch device to place the resulting embeddings on
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
            string.
        clean_caption (bool, defaults to `False`):
            If `True`, the function will preprocess and clean the provided caption before encoding.
        mask_feature: (bool, defaults to `True`):
            If `True`, the function will mask the text embeddings.
    """

    if device is None:
        device = text_encoder.device

    assert prompt is not None and (isinstance(prompt, str) or isinstance(prompt, list))
    batch_size = 1


    # See Section 3.1. of the paper.
    max_length = 200
    prompt = _text_preprocessing(prompt, clean_caption=clean_caption) 
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1: -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {max_length} tokens: {removed_text}"
        )

    attention_mask = text_inputs.attention_mask.to(device)
    prompt_embeds_attention_mask = attention_mask

    #print(f"device: text_encoder:{text_encoder.device}, text_input_ids:{text_input_ids.device}, attention_mask:{attention_mask.device}")

    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]


    dtype = text_encoder.dtype


    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape


    return prompt_embeds,prompt_embeds_attention_mask


def gen_validation_samples(videogen_pipeline, args, samples_dir, train_ts, step):
    video_grids = []
    for prompt in args.validation.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt,
                                   video_length=args.validation.video_length,
                                   height=args.validation.image_size[0],
                                   width=args.validation.image_size[1],
                                   num_inference_steps=args.validation.num_sampling_steps,
                                   guidance_scale=args.validation.guidance_scale,
                                   enable_temporal_attentions=args.validation.enable_temporal_attentions,
                                   num_images_per_prompt=1,
                                   mask_feature=True,
                                   enable_vae_temporal_decoder=args.validation.enable_vae_temporal_decoder
                                   ).video
        try:
            file_prefix = prompt.replace(' ', '_')[:50]
            imageio.mimwrite(samples_dir + '/' + file_prefix + f'{train_ts}_{step}_webv_imageio.mp4',
                             videos[0], fps=8, quality=9)  # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    video_grids = save_video_grid(video_grids)

    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    to_save_path = samples_dir + "/" + f'{train_ts}_{step}_merged_.mp4'
    imageio.mimwrite(to_save_path, video_grids, fps=8,
                     quality=5)
    print('saved in path {}'.format(to_save_path))

    # if args.use_wandb:
    #     wandb.log({"example": wandb.Video(to_save_path)}, step=step)


def init_video_vae(vae_config, vae_model_path):
    videovae_config = OmegaConf.load(vae_config)
    args.fps_ds = videovae_config.fps_ds
    videovae = CausualVAEVideo(ddconfig=videovae_config.ddconfig, embed_dim=videovae_config.embed_dim)
    checkpoint = torch.load(vae_model_path)
    msg = videovae.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("vae load from {}".format(vae_model_path))
    print(msg)
    for p in videovae.parameters():
        p.requires_grad = False
    t_downsampling = 2**(sum([s==2 for s in videovae_config.ddconfig.temporal_stride]))
    assert t_downsampling in [4, 8]
    videovae.patch_size = (t_downsampling, 8, 8)    # TODO: vae spatial downsampling is fixed to 8
    videovae.scaling_factor = getattr(videovae_config, f"scaling_factor_video")
    return videovae

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    global logger

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")


    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_models(args)

    # Note that parameter initialization is done within the constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    diffusion = create_diffusion(timestep_respacing="", device=device)  # default: 1000 steps, linear noise schedule

    target_dtype = torch.float32
    if args.mixed_precision == 'bf16':
        target_dtype=torch.bfloat16
    if args.mixed_precision == 'fp16':
        target_dtype=torch.half

    vae = init_video_vae(args.vae_config, args.vae_model_path).eval().to(target_dtype) 

    vae = vae.to(device) if not args.vae_cpu else vae.to('cpu') # it is strongly recomended to encode video offline before starting train the t2v model, when video is larger than 256x256

    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    if args.fixed_spatial:
        trainable_modules = (
        "attn_temp",
        )
        model.requires_grad_(False)
        for name, module in model.named_modules():
            if name.endswith(tuple(trainable_modules)):
                for params in module.parameters():
                    logger.info("WARNING: Only train {} parametes!".format(name))
                    params.requires_grad = True
        logger.info("WARNING: Only train {} parametes!".format(trainable_modules))

    # set distributed training
    if args.use_fsdp:
        config = {}
        model = FSDP(model, **config, device_id=device)
    else:
        model = DDP(model.to(device), device_ids=[local_rank])
    
    if args.dataset == 't2v':
        t5_encoder = T5Encoder(from_pretrained="/cpfs01/projects-HDD/cfff-01ff502a0784_HDD/public/pretrain_models/",
                                    model_max_length=200,
                                    shardformer=True)
        text_encoder = t5_encoder.t5.model
        text_encoder = text_encoder.to(device) if not args.text_encoder_cpu else text_encoder.to('cpu')
        text_encoder.eval()
        tokenizer = t5_encoder.t5.tokenizer

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    # import ipdb; ipdb.set_trace();
    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name=args.scheduler_type,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module if hasattr(model, 'module') else model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    train_ts = int(start_time)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(train_steps, args.max_train_steps), disable=rank!=0)
    progress_bar.set_description("Steps")


    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            videos_data = batch['videos'].to(device, non_blocking=True) if not args.vae_cpu else batch['videos'] # [1, 4, 16, 3, 512, 512]
            images_data = batch['images'].to(device, non_blocking=True) if not args.vae_cpu else batch['images'] # [1, 4, 3, 512, 512]
            video_captions = batch['video_captions']
            image_captions = batch['image_captions'] 

            if random.random() <= args.cfg_random_null_text_ratio:
                video_captions = ['']

            ## it is strongly recomended to encode video offline before we start the training of the t2v model, when target video size is larger than 256x256
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=target_dtype):
                    if videos_data.ndim == 6:
                        vdata_list = []
                        for idx in range(videos_data.shape[1]):
                            vdata = videos_data[0][idx].unsqueeze(0)  # [1, 16, 3, 512, 512]
                            #  # [b, 3, t, h, w]
                            vdata = rearrange(vdata, 'b f c h w -> b c f h w').contiguous()
                            if args.vae_cpu:
                                vdata = vae.encode(vdata.cpu()).sample().mul_(args.vae_scale_factor).to(device, non_blocking=True) # [1, z, t, h, w]
                            else:
                                vdata = vae.encode(vdata).sample().mul_(args.vae_scale_factor) # [1, z, t, h, w]
                            vdata_list.append(vdata)
                        videos_latent = torch.cat(vdata_list, dim=2) # [1, z, 5*b, h, w]

                    if images_data.ndim == 5:
                        vdata_list = [] 
                        for idx in range(images_data.shape[1]):
                            vdata = images_data[0][idx].unsqueeze(1).unsqueeze(0).contiguous()
                            if args.vae_cpu:
                                vdata = vae.encode(vdata.cpu()).sample().mul_(args.vae_scale_factor).to(device, non_blocking=True)
                            else:
                                vdata = vae.encode(vdata).sample().mul_(args.vae_scale_factor)
                            vdata_list.append(vdata)
                        images_latent = torch.cat(vdata_list, dim=2) # [1, z, 1*b, h, w]
                    x = torch.cat([videos_latent, images_latent], dim=2)

                    #  Encode input prompt
                    if 't2v' in args.dataset:
                        prompt_embeds_list = []

                        merge_captions = video_captions + image_captions
                        merge_captions = [item if isinstance(item, str) else ' '.join(item) for item in merge_captions]
                        prompt_embeds, prompt_embeds_attention_mask = encode_prompt(
                                tokenizer,
                                text_encoder,
                                merge_captions
                        )
                        prompt_embeds = prompt_embeds.unsqueeze(0).to(device, non_blocking=True)
                        prompt_embeds_attention_mask = prompt_embeds_attention_mask.unsqueeze(0).to(device, non_blocking=True)

            model_kwargs = dict(encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_embeds_attention_mask, 
                                use_image_num=args.use_image_num,
                                use_video_num=args.use_video_num) # tav unet

            torch.cuda.empty_cache()
            gc.collect()

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            if args.use_fsdp:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss = loss / args.gradient_accumulation_steps
            else:
                with torch.autocast(device_type='cuda', dtype=target_dtype):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                    loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            else:
                gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            # ref: https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation
            if train_steps % args.gradient_accumulation_steps == 0:
                torch.cuda.empty_cache()
                gc.collect()
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
                update_ema(ema, model.module if hasattr(model, 'module') else model)

            # Log loss values:
            running_loss += loss.item() * args.gradient_accumulation_steps #### 
            log_steps += 1
            train_steps += 1
            progress_bar.update(1)
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if False and train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:

                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args
                    }

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clean_fragment_interval", type=int, default=10000000)
    parser.add_argument("--scheduler_type", type=str, default='constant')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.update({"checkpoint": args.checkpoint})
    main(config)
