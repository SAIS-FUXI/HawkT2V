# HawkT2V

<!-- ![figure1](source/logo.png "figure1") -->
<div align=center>
<img src='source/logo.png' width="150" height="150" >
</div>


HawkT2V, a diffusion transformer model designed for generating videos from textual inputs. To adeptly handle the intricacies of video data, we incorporate 3D Variational Autoencoder (VAE) that effectively compresses video across spatial and temporal dimensions. Additionally, we enhance the diffusion transformer by integrating a window-based attention mechanism, specifically tailored for video generation tasks. Unfortunately, training text-to-video generative model from scratch demands significant computational resources and data. To overcome these obstacles, we implement a multi-stage training pipeline that optimizes training efficiency and effectiveness. Employing progressive training methodologies, HawkT2V is proficient at crafting coherent, long-duration videos with prominent motion dynamics.  
Currently, HawkT2V is be able to generate 2-4s 512x512 video!  
As soon as possible we will update the code to support the generation of longer and large resolution videos.


## Preparation
### Installation
1) install from pip
```bash
pip install -r requirements.txt
```
To enable xformers which may save the memory cost of GPU, you also need to install xformers and flash-attn first.
For xformers installation, please refer to more specific instruction on [Xformers](https://github.com/facebookresearch/xformers).
If it is not needed, make sure the xformers setting in the config file is False, which can be closed as the setting below:
```
enable_xformers_memory_efficient_attention: False
```

## Quick Start
### Inference
(1) Download the Model  
The 3B model can be downloaded through Huggingface [Fudan-FUXI/HawkT2V_1.0_3B](https://huggingface.co/Fudan-FUXI/HawkT2V_1.0_3B)  
And to run our provided inference example successfully, you can create a sub-directory named 'pretrained_models' in current working directory, then move the downloaded model to the sub-directory.

(2) Run inference  
We prepare some example prompts for video generation in 'samples/test_prompt.json', it is easy to do video generation by just using the command below:
```
bash scripts/inference.sh
```
Currently the provided chekpoint can generate 2s 512x512 video, we will update the pretrained model as soon as possible to support the generation of up to 8s video.

### Finetune
Currently, to finetune on your own dataset, the command below is an example:
```
bash scripts/train.sh
```
This script will finetune the 3B model on custom datasets, finnaly after enough training it will be able generate 512x512 videos.
To run the training of 3B model smoothly, the memory of GPU should be equal or larger than 80G.  

## Demos
Here are some 512x512 video examples generated by our 3B model.

<table>
<div>
<tr>
<video src=source/1-Standing_atop_a_hill_overlooking_a_battlefield,_a_gorgeous_female_samurai_stares_into_the_distance._She_turns_her_head_s1727253636_0_webv_imageio.mp4 width="45%" controls autoplay loop></video>
</tr>
<tr>
<video src=source/1-the_video_shows_a_bunny_nodding_head_confidently,_high_ornamented_light_armor,_fluffy_fur,_foggy,_wet,_stormy,_70mm,_cin1727258644_0_webv_imageio.mp4 width="45%" controls autoplay loop></video>
</tr>
<tr>
<video src=source/2-A_young_woman_with_sun-kissed_blonde_wavy_hair_and_light_hazel_eyes_walks_along_the_shoreline_of_a_tranquil_lake_at_suns1727249093_0_webv_imageio.mp4 width="45%" controls autoplay loop></video>
</tr>
<tr>
<video src=source/2-Produce_a_video_portrait_of_an_orange_tabby_cat_with_a_charming_smile,_dressed_in_a_Renaissance-inspired_costume._The_ca1727256594_0_webv_imageio.mp4 width="45%" controls autoplay loop></video>
</tr>
<tr>
<video src=source/3-the_video_shows_an_easter_bunny_blinking_eyes,_high_ornamented_light_armor,_fluffy_fur,_basket_full_of_painted_eggs,_70m1727237343_0_webv_imageio.mp4 width="45%" controls autoplay loop></video>
</tr>
<tr>
<video src=source/3-Under_the_dim_blue_night_sky,_a_woman_steps_out_of_the_house,_her_eyes_open_and_reflecting_the_warm_glow_of_the_windows.1727253636_0_webv_imageio.mp4 width="45%" controls autoplay loop></video>
</tr>
</div>
</table>

## Acknowledgement
HawkT2V is built upon many wonderful previous works, included but not limmited to [Latte](https://github.com/Vchitect/Latte), [Pixart](https://github.com/PixArt-alpha/PixArt-alpha), [HD-VILA](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md) and [LAION](https://laion.ai/blog/laion-400-open-dataset/)

## License Agreement
The code in this repository is released under the Apache 2.0 License.
