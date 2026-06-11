# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import argparse
import os

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_masks


def main(args):
    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(args.image_path)
    image_name = os.path.basename(os.path.dirname(args.image_path)).split(".")[0]
    # mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
    masks = load_masks(os.path.dirname(args.image_path), extension=".png")

    # run model
    # output = inference(image, mask, seed=42)
    outputs = [inference(image, mask, seed=42) for mask in masks]

    for i, out in enumerate(outputs):
        mesh = out["glb"]
        outpath = os.path.join(args.output_path, f"{image_name}", f"object{i}.ply")
        mesh.export(outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    parser.add_argument("--masks_path", type=str, required=True, help="Path to masks directory")
    parser.add_argument("--output_path", type=str, default="output", help="Path to output directory")
    args = parser.parse_args()
    
    main(args)