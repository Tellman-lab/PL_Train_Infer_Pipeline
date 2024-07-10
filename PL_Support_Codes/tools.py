import omegaconf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_gif(
    image_list,
    save_path,
    fps=1,
    image_text=None,
    fontpct=5,
    overlay_images=None,
    optimize=False,
):
    """Create a gif image from a collection of numpy arrays.

    Args:
        image_list (list[numpy array]): A list of images in numpy format of type uint8.
        save_path (str): Path to save gif file.
        fps (float, optional): Frames per second. Defaults to 1.
        image_text (list[str], optional): A list of text to add to each frame of the gif.
            Must be the same length as image_list.
    """

    # Check dtype of images in image list.
    assert type(image_list) is list
    assert all([img.dtype == "uint8" for img in image_list])

    if len(image_list) < 2:
        print(
            f"Cannot create a GIF with less than 2 images, only {len(image_list)} provided."
        )
        return None
    elif len(image_list) == 2:
        img, imgs = Image.fromarray(
            image_list[0]), [Image.fromarray(image_list[1])]
    else:
        img, *imgs = [Image.fromarray(img) for img in image_list]

    if overlay_images is not None:
        assert len(overlay_images) == len(image_list)

        # Overlay images together
        images = [img]
        images.extend(imgs)

        images_comb = []
        for image_1, image_2 in zip(images, overlay_images):
            # Make sure images have alpha channel
            image_1.putalpha(1)
            image_2.putalpha(1)

            # Overlay images
            try:
                image_comb = Image.alpha_composite(image_1, image_2)
            except:
                breakpoint()
                pass
            images_comb.append(image_comb)

        img, *imgs = [img for img in images_comb]

    if image_text is not None:
        assert len(image_text) == len(image_list)

        # Have an issue loading larger font
        H = image_list[0].shape[0]
        if fontpct is None:
            font = ImageFont.load_default()
        else:
            if H < 200:
                font = ImageFont.load_default()
            else:
                fontsize = int(H * fontpct / 100)
                # Find fonts via "locate .ttf"
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                        fontsize)
                except:
                    print(
                        "Cannot find font at: /usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
                    )
                    font = ImageFont.load_default()

        images = [img]
        images.extend(imgs)
        for i, (img, text) in enumerate(zip(images, image_text)):
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, (255, 0, 0), font=font)
            images[i] = img

        img, *imgs = images

    # Convert the images to higher quality
    images = [img]
    images.extend(imgs)
    img, *imgs = [img.quantize(dither=Image.NONE) for img in images]

    duration = int(1000 / fps)
    img.save(
        fp=save_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=duration,
        loop=0,
        optimize=optimize,
    )


def load_cfg_file(path):
    with open(path, "r") as fp:
        cfg = omegaconf.OmegaConf.load(fp.name)
    return cfg


def create_conf_matrix_pred_image(pred, target):
    # Pred: numpy
    # target: numpy
    out_image = np.zeros([pred.shape[0], pred.shape[1], 3], dtype='uint8')

    # TP - White
    x, y = np.where((pred == 1) & (target == 1))
    out_image[x, y, :] = np.array([255, 255, 255])

    # FP - Teal
    x, y = np.where((pred == 1) & (target == 0))
    out_image[x, y, :] = np.array([0, 255, 255])

    # FN - Red
    x, y = np.where((pred == 0) & (target == 1))
    out_image[x, y, :] = np.array([255, 0, 0])

    return out_image
