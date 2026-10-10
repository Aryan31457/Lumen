import base64
import cv2
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from .models import get_vlm
from . import prompts


class Caption(BaseModel):
    caption: str
    activity: str = ""
    scene: str = ""
    tags: list[str] = Field(default_factory=list)
    people_visible: list[str] = Field(default_factory=list)


def to_b64_jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok: raise ValueError('Could not encode image')
    return base64.b64encode(encoded).decode()


def caption_image(image, labels):
    user = (prompts.CAPTION_USER.replace('<<LABELS>>', ', '.join(labels)) if labels else prompts.CAPTION_USER_NO_PEOPLE)
    message=HumanMessage(content=[{"type":"text","text":user},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+to_b64_jpeg(image)}}])
    result=get_vlm().with_structured_output(Caption).invoke([SystemMessage(content=prompts.CAPTION_SYSTEM),message])
    allowed=set(labels)
    if not set(result.people_visible).issubset(allowed): raise ValueError('Caption included invalid labels')
    return result
