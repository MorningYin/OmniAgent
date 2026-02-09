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

Think step by step in 1-2 sentences, then give your answer.
You must end with exactly this format:
Answer: <a single letter A, B, C, or D>"""


DEPTH_ASSISTED_ANSWER_TEMPLATE = """\
Here is a depth map of the same scene. Your previous answer may be wrong. Use the depth map to verify or correct it.

Reminder — the question is:
{question}
Options:
{options_text}

How to read the depth map:
- BRIGHTER (whiter) pixels = CLOSER to the camera.
- DARKER (blacker) pixels = FARTHER from the camera.

You MUST follow these steps:
1. Identify the objects mentioned in the question. For EACH object, find its EXACT region in the depth map and describe its brightness (bright/dark/medium). Do NOT describe irrelevant objects.
2. Compare the brightness of the relevant objects to determine their relative depth.
3. Give your final answer based ONLY on this depth comparison. If it contradicts your previous answer, you MUST change it.

Answer: <a single letter A, B, C, or D>"""


DEPTH_MAP_TEMPLATE = "生成这张图的深度图"
