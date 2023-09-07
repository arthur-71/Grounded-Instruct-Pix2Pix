import cv2
import torch
import spacy

import numpy as np
from PIL import Image

from huggingface_hub import hf_hub_download
from torchmetrics.multimodal.clip_score import CLIPScore


# Grounding DINO + Segment Anything
from segment_anything import build_sam, SamPredictor

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import load_image, predict


# This downloads and builds a GroundingDINO model
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda:0'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    _ = model.eval().to(device)
    return model


def join_bboxes(bboxes_xyxy):
    min_x, min_y, _, _ = torch.min(bboxes_xyxy, dim=0)[0]
    _, _, max_x, max_y = torch.max(bboxes_xyxy, dim=0)[0]
    
    return torch.tensor([min_x.item(), min_y.item(), max_x.item(), max_y.item()])



class ExternalMaskExtractor():
    def __init__(self, device, clip_score_threshold=0., mask_dilation_size=11) -> None:
        self.device = device
        self.mask_dilation_size = mask_dilation_size

        self.nlp = spacy.load("en_core_web_sm")
        self.clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)

        # External Segmentation method = Grounded-SAM (which is GroundingDINO + SAM with bounding box input)
        # First let's load GroundingDINO
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
        
        # Next, load Segment-Anything
        sam_path = '/home/artur.shagidanov/text-guided-image-editing/Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_path).to(device)
        self.sam_predictor = SamPredictor(sam)


    def _dino_predict(self, image, prompt):
        # preprocessing
        transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_transformed, _ = transform(image, None)
        boxes, _, _ = predict(model=self.groundingdino_model, image=image_transformed, caption=prompt,
                              box_threshold=0.3, text_threshold=0.25, device='cuda:0')
        
        # account for no bboxes detected case
        return boxes if boxes.shape[0] != 0 else None

    def _grounded_sam_predict(self, image, text_prompt):
        image_orig = np.asarray(image)
        boxes = self._dino_predict(image, text_prompt)        
        
        # if no boxes were detected
        if boxes is None:
            return torch.zeros(image_orig.shape[:2], dtype=torch.float32).to('cuda:0')
        # else, proceed with SAM with bbox
        else:
            self.sam_predictor.set_image(image_orig)
            H, W, _ = image_orig.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_orig.shape[:2]).to('cuda:0')
            masks, _, _ = self.sam_predictor.predict_torch(point_coords=None, point_labels=None,
                                                           boxes=transformed_boxes, multimask_output=False)

            masks_sum = sum(masks)[0]
            masks_sum[masks_sum > 1] = 1
            
            return masks_sum

        
    def _get_noun_phrases(self, prompt, verbose=False):
        parsed_edit = self.nlp(prompt)
        edit_noun_phrases = [chunk.text for chunk in parsed_edit.noun_chunks]
        if verbose:
            print("Edit Instruction Noun-Phrases:", edit_noun_phrases, '\n')
        return edit_noun_phrases

    def _get_clip_scores(self, image, prompt_noun_phrases, verbose=False):
        width, height = image.size

        clip_scores = dict()
        for noun_phrase in prompt_noun_phrases:
            # get the encompassing bbox for the noun-phrase
            bboxes = self._dino_predict(image, noun_phrase)
            if bboxes is None:
                print(noun_phrase, ': not found, clip_score = 0')
                clip_scores[noun_phrase] = 0.0
                continue
            bboxes_xyxy = box_ops.box_cxcywh_to_xyxy(bboxes) * torch.Tensor([width, height, width, height])
            encompassing_bbox = join_bboxes(bboxes_xyxy)
            
            # crop the image with the bbox, resize to max_size==224 and compute CLIP_Score(img_crop, noun_phrase)
            image_cropped = image.crop(encompassing_bbox.data.cpu().tolist())
            image_cropped.thumbnail((224, 224), Image.Resampling.LANCZOS)
            if verbose:
                image_cropped.show()
            image_cropped = np.array(image_cropped)[None, :].transpose(0, 3, 1, 2)
            image_cropped = torch.from_numpy(image_cropped).to(self.device)
            
            clip_score = self.clip_metric(image_cropped, noun_phrase).cpu().item()
            clip_scores[noun_phrase] = clip_score
            if verbose:
                print(f'CLIP Score(^ above image; "{noun_phrase}"):', clip_score)
        
        return clip_scores

    @torch.no_grad()
    def get_external_mask(self, image, prompt, exclude_noun_phrases=None, verbose=False):
        # Extract all noun-phrases
        prompt_noun_phrases = self._get_noun_phrases(prompt, verbose=verbose)
        if len(prompt_noun_phrases) == 0:
            print('No noun-phrases detected, falling back to Instruct-Pix2Pix.')
            chosen_noun_phrase = 'IP2P_FALLBACK_MODE'
            width, height = image.size
            external_mask = np.ones((height, width), dtype=np.uint8)
            external_mask = Image.fromarray((255*external_mask).astype(np.uint8))
            return external_mask, chosen_noun_phrase, None

        # Compute CLIP-Score for each one
        clip_scores = self._get_clip_scores(image, prompt_noun_phrases, verbose=verbose)
        if exclude_noun_phrases is not None:
            for exclude_noun_phrase in exclude_noun_phrases:
                clip_scores[exclude_noun_phrase] = -1
                
        # Select the one with the highest score
        chosen_noun_phrase = max(clip_scores, key=clip_scores.get)

        # Extract its mask
        external_mask = self._grounded_sam_predict(image, chosen_noun_phrase)
        external_mask = cv2.dilate(external_mask.data.cpu().numpy().astype(np.uint8),
                                   kernel=(np.ones((self.mask_dilation_size, self.mask_dilation_size), np.uint8)))
        external_mask = Image.fromarray((255*external_mask).astype(np.uint8))
        if verbose:
            print(f'Chose "{chosen_noun_phrase}" as input to G-SAM.')
            external_mask.show()
            
        return external_mask, chosen_noun_phrase, clip_scores