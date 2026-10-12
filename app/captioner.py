import re

import cv2
import torch
from PIL import Image
from pydantic import BaseModel, Field

from .config import settings
from .models import get_caption_model


class Caption(BaseModel):
    caption: str
    activity: str = ""
    scene: str = ""
    tags: list[str] = Field(default_factory=list)
    people_visible: list[str] = Field(default_factory=list)


def caption_image(image, labels):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    processor, model, device = get_caption_model()
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.caption_max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    caption = processor.decode(output[0], skip_special_tokens=True).strip()
    caption = re.sub(r"\s+", " ", caption).strip(" .,:;\"'")

    # Avoid turning appearance guesses into identity metadata.
    for pattern, replacement in (
        (r"\bmen\b", "people"), (r"\bman\b", "person"),
        (r"\bwomen\b", "people"), (r"\bwoman\b", "person"),
        (r"\bboys?\b", "person"), (r"\bgirls?\b", "person"),
        (r"\bhe\b|\bshe\b", "they"), (r"\bhis\b|\bher\b", "their"),
    ):
        caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
    if len(caption.split()) < 3:
        caption = "People are visible in this photo." if labels else "A photo."

    activity = caption
    if labels:
        # Make captions templates with the detected people as the grammatical
        # subject. `P1` tokens are replaced with stable face/person tokens by
        # ingestion, and with assigned names when the API renders the caption.
        action = re.sub(r"\s+", " ", caption).strip(" .,:;")
        # Captions from BLIP often start with group descriptions such as
        # "a couple and their children". Keep the observable action/context,
        # and discard those competing subject guesses so the person tokens
        # remain the only subjects in the sentence.
        activity_verbs = (
            "swimming", "reading", "walking", "running", "riding", "cycling",
            "playing", "sitting", "standing", "posing", "holding", "eating",
            "drinking", "talking", "looking", "smiling", "dancing", "working",
            "cooking", "painting", "writing", "leaning", "taking", "wearing",
        )
        lower = action.lower()
        positions = [lower.find(verb) for verb in activity_verbs if lower.find(verb) >= 0]
        if positions:
            action = action[min(positions):]
            action = re.sub(r"^(?:is|are|was|were)\s+", "", action, flags=re.IGNORECASE)
            action = action[:1].lower() + action[1:]
            subject = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
            verb = "is" if len(labels) == 1 else "are"
            caption = f"{subject} {verb} {action.rstrip('.')} .".replace(" .", ".")
        else:
            context = re.search(r"\b(in|at|on)\s+(?:(a|an|the)\s+)?([\w-]+(?:\s+[\w-]+){0,3})", action, re.IGNORECASE)
            article = f"{context.group(2)} " if context and context.group(2) else ""
            location = f"{context.group(1).lower()} {article}{context.group(3).strip()}" if context else "in the photo"
            subject = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
            verb = "is" if len(labels) == 1 else "are"
            caption = f"{subject} {verb} {location.rstrip('.')} .".replace(" .", ".")
    else:
        caption = caption[:1].upper() + caption[1:] if caption else ""
    return Caption(caption=caption, activity=activity, people_visible=list(labels))
