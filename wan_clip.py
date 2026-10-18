"""One Wan 2.2 TI2V-5B video clip on a Tesla V100 (Volta, sm_70, 32 GB) -- text-to-video or image-to-video.

The V100 has no fast bf16 and no FlashAttention, and the machine this was written on has only 16 GB of system RAM, so:
- the video model (transformer) runs in fp16, loaded straight onto the GPU;
- the umT5 text encoder runs in fp32 (umT5/T5 can overflow in fp16), encodes the prompt and the negative prompt, and is
  deleted before the video model loads (the prompt is embedded the way diffusers' WanPipeline does it: padded to 512,
  zeros after the real tokens -- done by hand so the optional `ftfy` package isn't needed);
- the VAE decodes in fp16 with --vae-fp16 (about 4x faster than fp32 here, no visible change), tiled with --tile;
- frames are piped to ffmpeg (no imageio/opencv needed).
--fast DIR swaps in FastWan 2.2 (FastVideo's 3-step distilled transformer) with its DMD sampler and no CFG.

Examples:
  python wan_clip.py --model Wan2.2-TI2V-5B-Diffusers --prompt "..." --out robot.mp4 --width 1280 --height 704 --tile --vae-fp16
  python wan_clip.py --model Wan2.2-TI2V-5B-Diffusers --fast FastWan2.2-TI2V-5B-FullAttn-Diffusers --steps 3 \
      --image dog.jpg --prompt "..." --out dog.mp4 --width 704 --height 1056 --tile --vae-fp16
Run with CUDA_VISIBLE_DEVICES set to the V100. Tested: diffusers 0.41.0, transformers 5.18.0, torch 2.8.0+cu126.
"""
import argparse
import gc
import subprocess
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
from transformers import AutoTokenizer, UMT5EncoderModel

# Wan's own default negative prompt (from the Wan 2.2 repository)
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走")


class DMD:
    """FastWan's 3-step sampler, as FastVideo runs it: at each timestep the model's flow prediction gives the clean
    video (x0 = x_t - sigma * v, with sigma = t / 1000), which is noised again to the next timestep; the last step's
    clean video is the result. Plugs into the diffusers Wan pipelines in place of their scheduler."""
    order = 1

    def __init__(self, timesteps=(1000, 757, 522), seed=0):
        self.ts = list(timesteps)
        self.g = torch.Generator("cuda").manual_seed(seed + 1)

    def set_timesteps(self, n, device=None, **kw):
        self.timesteps = torch.tensor(self.ts, dtype=torch.float32, device=device)
        self.sigmas = torch.tensor([t / 1000 for t in self.ts] + [0.0])

    def set_begin_index(self, i=0):
        pass

    def step(self, v, t, x, return_dict=False, **kw):
        x0 = x.float() - float(t) / 1000 * v.float()
        i = self.ts.index(int(round(float(t))))
        if i == len(self.ts) - 1:
            return (x0.to(x.dtype),)
        s_next = self.ts[i + 1] / 1000
        noise = torch.randn(x0.shape, generator=self.g, device=x0.device, dtype=torch.float32)
        return (((1 - s_next) * x0 + s_next * noise).to(x.dtype),)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Wan2.2-TI2V-5B-Diffusers folder (text encoder, VAE, config)")
    ap.add_argument("--fast", help="FastWan2.2-TI2V-5B-FullAttn-Diffusers folder: its transformer, 3 steps, no CFG")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", help="start from this picture (image-to-video); it is resized to --width x --height")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--frames", type=int, default=121, help="4n+1; 121 = 5 s at 24 fps")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0, help="ignored with --fast (1.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", action="store_true", help="decode in tiles (needed at 720p x 121 frames)")
    ap.add_argument("--vae-fp16", action="store_true", help="decode in fp16 (V100 tensor cores; ~4x faster than fp32)")
    a = ap.parse_args()
    W, H = a.width, a.height
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(a.model, subfolder="text_encoder", dtype=torch.float32, device_map={"": 0},
                                          low_cpu_mem_usage=True).eval()

    def embed(text):
        t = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        with torch.no_grad():
            e = te(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state[0, :n]
        e = torch.cat([e, e.new_zeros(512 - n, e.size(1))])[None]
        assert torch.isfinite(e).all(), "the text encoder gave inf/nan"
        return e.to(torch.float16)

    pe, ne = embed(a.prompt), embed(NEGATIVE)
    del te
    gc.collect()
    torch.cuda.empty_cache()
    t1 = time.time()

    transformer = WanTransformer3DModel.from_pretrained(a.fast or a.model, subfolder="transformer", torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True, device_map="cuda")
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae",
                                           torch_dtype=torch.float16 if a.vae_fp16 else torch.float32).to("cuda")
    if a.tile:
        vae.enable_tiling()
    image = None
    if a.image:
        from PIL import Image
        image = Image.open(a.image).convert("RGB").resize((W, H), Image.LANCZOS)
        pipe = WanImageToVideoPipeline.from_pretrained(a.model, transformer=transformer, vae=vae, text_encoder=None,
                                                       tokenizer=None, image_encoder=None, image_processor=None,
                                                       torch_dtype=torch.float16)
    else:
        pipe = WanPipeline.from_pretrained(a.model, transformer=transformer, vae=vae, text_encoder=None, tokenizer=None,
                                           torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    if a.fast:
        pipe.scheduler = DMD(seed=a.seed)
    t2 = time.time()

    g = torch.Generator("cuda").manual_seed(a.seed)
    frames = pipe(**({"image": image} if image is not None else {}), prompt_embeds=pe, negative_prompt_embeds=ne,
                  height=H, width=W, num_frames=a.frames, guidance_scale=1.0 if a.fast else a.guidance,
                  num_inference_steps=a.steps, generator=g, output_type="np").frames[0]
    t3 = time.time()
    ok = bool(np.isfinite(frames).all())
    frames = (np.clip(np.nan_to_num(frames), 0, 1) * 255).astype(np.uint8)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], stdin=subprocess.PIPE)
    ff.stdin.write(frames.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{'FastWan ' if a.fast else ''}clip {W}x{H}, {a.frames} frames, {a.steps} steps, seed {a.seed}: "
          f"text {t1 - t0:.0f}s, load {t2 - t1:.0f}s, generate+decode {t3 - t2:.0f}s, finite {ok}, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
