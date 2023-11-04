import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

class GroundedSAM:
    def __init__(self, 
                 grounding_dino_config_path: str = "data/checkpoints/GroundingDINO_SwinT_OGC.cfg.py",
                 grounding_dino_checkpoint_path: str = "data/checkpoints/groundingdino_swint_ogc.pth",
                 sam_encoder_version: str = "vit_b",
                 sam_checkpoint_path: str = "data/checkpoints/sam_vit_b_01ec64.pth",
                 classes: list = ["cloth"],
                 box_threshold: float = 0.25,
                 text_threshold: float  = 0.25,
                 nms_threshold: float = 0.8,
                 vis: bool = False) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=grounding_dino_config_path, 
                                          model_checkpoint_path=grounding_dino_checkpoint_path,
                                          device=device)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path).to(torch.device(device))
        self.sam_predictor = SamPredictor(sam)

        # params
        self.classes = list(classes)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.vis = vis

    def predict(self, image: np.ndarray) -> np.ndarray:
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        if self.vis:
            # visualize the annotated grounding dino image
            cv2.imshow("groundingdino_annotated_image.jpg", annotated_frame)
            cv2.waitKey()
            # visualize the annotated grounded-sam image
            cv2.imshow("grounded_sam_annotated_image.jpg", annotated_image)
            cv2.waitKey()

        return detections.mask 

    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        # Prompting SAM with detected boxes
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
if __name__ == '__main__':
    SOURCE_IMAGE_PATH = '/home/xuehan/Desktop/PhoXiCameraCPP/ExternalCamera/Data/test0.png'

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    grounded_sam_model = GroundedSAM(vis=True)

    masks = grounded_sam_model.predict(image)
