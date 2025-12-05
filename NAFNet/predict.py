import torch
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt
from cog import BasePredictor, Path, Input, BaseModel

import argparse
import os

from tqdm import tqdm

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


class Predictor(BasePredictor):
    def setup(self, task_type: str = None):
        self.task_type = task_type
        if task_type == "Image Denoising":
            opt_path_denoise = "options/test/SIDD/NAFNet-width64.yml"
            opt_denoise = parse(opt_path_denoise, is_train=False)
            opt_denoise["dist"] = False
            self.model = create_model(opt_denoise)
        elif task_type == "Image Debluring":
            opt_path_deblur = "options/test/GoPro/NAFNet-width64.yml"
            opt_deblur = parse(opt_path_deblur, is_train=False)
            opt_deblur["dist"] = False
            self.model = create_model(opt_deblur)
        elif task_type == "Stereo Image Super-Resolution":
            opt_path_stereo = "options/test/NAFSSR/NAFSSR-L_4x.yml"
            opt_stereo = parse(opt_path_stereo, is_train=False)
            opt_stereo["dist"] = False
            self.model = create_model(opt_stereo)

    def predict(
        self,
        image: Path = Input(
            description="Input image. Stereo Image Super-Resolution, upload the left image here.",
        ),
        image_r: Path = Input(
            default=None,
            description="Right Input image for Stereo Image Super-Resolution. Optional, only valid for Stereo"
            " Image Super-Resolution task.",
        ),
        output_path: Path = Input(default=None, description="Output path.")
    ) -> Path:

        out_path = output_path if output_path else Path(tempfile.mkdtemp()) / "output.png"

        model = self.model
        if self.task_type == "Stereo Image Super-Resolution":
            assert image_r is not None, (
                "Please provide both left and right input image for "
                "Stereo Image Super-Resolution task."
            )

            img_l = imread(str(image))
            inp_l = img2tensor(img_l)
            img_r = imread(str(image_r))
            inp_r = img2tensor(img_r)
            stereo_image_inference(model, inp_l, inp_r, str(out_path))

        else:

            img_input = imread(str(image))
            # If too large, resize for faster inference
            h, w = img_input.shape[:2]
            scale = 1
            if max(h, w) > 1500:
                scale = 2
                img_input = cv2.resize(img_input, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)
            inp = img2tensor(img_input)
            single_image_inference(model, inp, str(out_path), original_size=(h, w) if scale != 1 else None)

        return out_path


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
    
    # Calculate Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()

    return variance > 100

def single_image_inference(model, img, save_path, original_size=None):
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals["result"]])
    
    if laplacian_variance(sr_img):
        # Drop the image if it's too sharp (likely has artifacts)
        sr_img = tensor2img([img])

    if original_size is not None:
        h, w = original_size
        sr_img = cv2.resize(
            sr_img,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    
    imwrite(sr_img, save_path)


def stereo_image_inference(model, img_l, img_r, out_path):
    img = torch.cat([img_l, img_r], dim=0)
    model.feed_data(data={"lq": img.unsqueeze(dim=0)})

    if model.opt["val"].get("grids", False):
        model.grids()

    model.test()

    if model.opt["val"].get("grids", False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    img_L = visuals["result"][:, :3]
    img_R = visuals["result"][:, 3:]
    img_L, img_R = tensor2img([img_L, img_R], rgb2bgr=False)

    # save_stereo_image
    h, w = img_L.shape[:2]
    fig = plt.figure(figsize=(w // 40, h // 40))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.title("NAFSSR output (Left)", fontsize=14)
    ax1.axis("off")
    ax1.imshow(img_L)

    ax2 = fig.add_subplot(2, 1, 2)
    plt.title("NAFSSR output (Right)", fontsize=14)
    ax2.axis("off")
    ax2.imshow(img_R)

    plt.subplots_adjust(hspace=0.08)
    plt.savefig(str(out_path), bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAFNet Predictor")
    parser.add_argument("-i", "--input_dir", type=Path, required=True, help="Input image directory.")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Output path.")
    args = parser.parse_args()

    predictor = Predictor()
    predictor.setup(task_type="Image Denoising")  # Change task type as needed

    input_images = [f for f in os.listdir(args.input_dir) if f.endswith(".png")]  # Adjust the glob pattern as needed
    os.makedirs(args.output_dir, exist_ok=True)

    for image in tqdm(input_images):
        # The same name but saved in output directory
        output_path = args.output_dir / image
        predictor.predict(
            image=os.path.join(args.input_dir, image),
            output_path=output_path
        )

    print("Inference completed.")
