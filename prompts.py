"""VLM prompt templates (Triage + Direct / Depth-assisted answer)."""

SYSTEM_PROMPT = """\
You are a 3D spatial reasoning expert. Answer spatial reasoning multiple-choice questions based on an image."""


TRIAGE_USER_TEMPLATE = """\
Carefully observe this image and assess the difficulty of the following spatial reasoning question.

Question: {question}
Options:
{options_text}

Important: Judging 3D depth relationships from a 2D image is error-prone. Be honest about your ability and do not be overconfident.
Only consider depth assistance unnecessary when the depth relationship is very obvious (e.g., one object is clearly in the foreground and the other is clearly in the distant background).

Output strictly in the following format:
CONFIDENCE: <an integer from 1 to 10, where 10 means completely certain and 1 means completely uncertain>
NEED_DEPTH_MAP: <YES or NO>
REASON: <one sentence explanation>"""


DIRECT_ANSWER_TEMPLATE = """\
Carefully observe this image and answer the following spatial reasoning question.

Question: {question}

Options:
{options_text}

First, briefly analyze the spatial positions of the objects in the image, then give your answer.
You must end with exactly this format:
Answer: <a single letter A, B, C, or D>"""


DEPTH_ASSISTED_ANSWER_TEMPLATE = """\
Answer the following spatial reasoning question using the two images below.

The first image is the [original image]. The second image is the [depth map of the original image].
In the depth map, brighter areas are closer to the camera and darker areas are farther away.

Use the brightness differences in the depth map to determine the relative distances of the objects.

Question: {question}

Options:
{options_text}

First, briefly analyze the brightness differences in the depth map and the relative distances of the objects, then give your answer.
You must end with exactly this format:
Answer: <a single letter A, B, C, or D>"""


DEPTH_MAP_TEMPLATE = "Generate a depth map of this image"
