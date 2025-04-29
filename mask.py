import torch

def get_masked_fn(config):
    if config.eval.mask_type == "box_center":
        return mask_box_center
    elif config.eval.mask_type == "box_checkerboard":
        return mask_box_checkerboard
    else:
        raise NotImplementedError(f"does not support mask type: {config.eval.mask_type} yet")
    
def mask_box_center(config):
    ## Returns: mask tensor of shape [H, W] where 1.0 = masked; 0.0 = unmasked
    # NOTE: assumes that img W == H
    img_size = config.data.image_size
    mask_box_size = config.eval.mask_box_size # int

    mask = torch.zeros((img_size, img_size), device=config.device).float()
    img_center = img_size//2
    
    mask_start = img_center - mask_box_size // 2
    mask_end = mask_start + mask_box_size

    mask[mask_start:mask_end, mask_start:mask_end] = 1.0

    return mask

def mask_box_checkerboard(config):
    print("USING CHECKERBOARD MASK")
    img_size = config.data.image_size
    mask_box_size = config.eval.mask_box_size # int
    mask = torch.zeros((img_size, img_size), device=config.device).float()
    for row in range(0,img_size,mask_box_size*2):
        for col in range(0, img_size, mask_box_size*2):
            box_row_end = min(img_size, row+mask_box_size)
            box_col_end = min(img_size, col+mask_box_size)
            mask[row:box_row_end, col:box_col_end] = 1.0

    return mask