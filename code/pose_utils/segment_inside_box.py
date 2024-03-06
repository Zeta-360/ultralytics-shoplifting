
import sys
import os

from imports import *





class Segment_box:
    def __init__(self, device="0"):
        self.device = device
        self.Fast_SAM_frames = []

        self.SAM_model= FastSAM(os.path.join("../weights/FastSAM.pt"))

    def overlay_mask_with_image(self,frame, mask):
        
        mask = repeat(mask, 'h w -> h w c', c=3)
        mask = mask.astype(np.uint8)
        
        
        
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask = cv2.addWeighted(frame, 0.3, mask, 0.7, 0)
        #covert image to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return mask

    def segment_own_code(self,frame,masks):
        mask_np = masks[0].masks.data.cpu().numpy()
        img_ei = rearrange(mask_np,'c h w-> h w c')

        prediction = np.argmax(img_ei, axis=2)
        final_mask = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlayed_image = self.overlay_mask_with_image(frame, final_mask)
        return overlayed_image

    def crop_image(self,frame, bbox):
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    def stitch_back_to_original(self,original_image,masked_part,bbox):
        original_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = masked_part
        return original_image

    def segment_inside_box(self,frame,bbox):
        original_image = frame.copy()
        frame = self.crop_image(frame, bbox)
        # box = [x,y,x2,y2]
        # float_box = [tensor.item() for tensor in box]
        DEVICE = 0
        everything_results = self.SAM_model(frame, device=DEVICE, retina_masks=True, conf=0.3, iou=0.9)
        prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
        #masks = prompt_process.box_prompt(bbox = float_box)
        masks = everything_results
        masked_part = self.segment_own_code(frame,masks)
        return self.stitch_back_to_original(original_image,masked_part,bbox)