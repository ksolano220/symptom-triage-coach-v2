"""Multi-modal symptom triage.

Takes an image of an affected area + a plain-language symptom description,
returns a schema-valid triage JSON: image-grounded visual findings, body
systems involved, ranked possible causes, red flags, and questions the
doctor will likely ask.

Run from the command line:
    python inference.py path/to/image.jpg "I have an itchy red rash on my forearm"
"""

import base64
import json
import os
import sys
from anthropic import Anthropic
from dotenv import load_dotenv
from jsonschema import validate

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "visual_findings",
        "systems",
        "possible_causes",
        "red_flags",
        "questions_to_prepare_for",
    ],
    "properties": {
        "visual_findings": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "systems": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "possible_causes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "likelihood", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "likelihood": {
                        "type": "string",
                        "enum": ["common", "less_common", "rare", "serious"],
                    },
                    "description": {"type": "string"},
                },
            },
            "minItems": 2,
            "maxItems": 8,
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "questions_to_prepare_for": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
        },
    },
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are a pre-visit triage assistant. A patient describes a symptom in plain language and uploads an image of the affected area. Your job is to help them prepare for their doctor's appointment, not diagnose.

Return a JSON object with five fields:

1. "visual_findings": 2-5 short strings describing what is VISIBLE in the image. Use clinical observation language ("raised erythematous papules on the volar forearm", "ill-defined plaque with silvery scale"). Do NOT infer findings that aren't visible. If image quality is poor, say so.

2. "systems": 1-5 body systems involved, lowercase. Choose from: cardiovascular, respiratory, gastrointestinal, neurological, musculoskeletal, dermatological, mental_health, endocrine, urological, reproductive, hematologic, immunologic, ophthalmologic, otolaryngologic.

3. "possible_causes": 2-8 objects with {name, likelihood, description}. likelihood is one of: common, less_common, rare, serious.

4. "red_flags": up to 6 strings, each a feature warranting immediate care.

5. "questions_to_prepare_for": 2-6 questions the doctor will likely ask.

Ground findings in what's visible AND consistent with the description. If the image and description conflict, surface the conflict.

Return ONLY the JSON object. No prose before or after."""


_client = Anthropic()  # reads ANTHROPIC_API_KEY from env


def triage(image_bytes: bytes, symptom_text: str) -> dict:
    """Run multi-modal triage on an image + symptom description.

    Raises ValueError if the model's response can't be parsed.
    Raises jsonschema.ValidationError if the parsed JSON doesn't match the schema.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")
    if not symptom_text.strip():
        raise ValueError("symptom_text is empty")

    media_type = _detect_media_type(image_bytes)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Patient description: {symptom_text}\n\nReturn the triage JSON.",
                    },
                ],
            }
        ],
    )

    text = response.content[0].text
    result = _extract_json(text)
    validate(instance=result, schema=OUTPUT_SCHEMA)
    return result


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect PNG / JPEG / GIF / WEBP from magic bytes."""
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unsupported image format. Use PNG, JPEG, GIF, or WEBP.")


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's response.

    The prompt asks for JSON only, but models occasionally wrap output in
    markdown fences or add prose. This is the safety net.
    """
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"Could not parse JSON from model response: {text[:200]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <image_path> \"<symptom description>\"")
        sys.exit(1)

    image_path, symptom_text = sys.argv[1], sys.argv[2]

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = triage(image_bytes, symptom_text)
    print(json.dumps(result, indent=2))
