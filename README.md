## Symptom Triage Coach v2 (Multi-modal)

Pre-visit prep assistant that takes an **image of the affected area** plus a **plain-language symptom description** and returns a structured triage JSON: image-grounded visual findings, body systems involved, ranked possible causes, red flags, and questions the doctor will likely ask.

This is the multi-modal extension of [`symptom-triage-coach`](https://github.com/ksolano220/symptom-triage-coach). v1 was a LoRA fine-tune of Qwen2.5-1.5B for text-only triage. v2 takes the same output schema and adds a vision input via Claude's vision API.

This is not a diagnostic tool. It is a pre-visit prep assistant that helps patients walk into their appointment informed.

### What's different vs v1

| | v1 | v2 |
|---|---|---|
| Input | Symptom text only | Image + symptom text |
| Model | Qwen2.5-1.5B-Instruct + LoRA adapter | Claude Sonnet 4.6 (vision) via API |
| New schema field | (none) | `visual_findings`: clinical observations grounded in the image |
| Hosting | Hugging Face Spaces (Gradio, free CPU) | Streamlit Cloud (planned) |

The new `visual_findings` field is the image-grounding signal. It forces the model to surface what is *visible* in the image as a separate output, which lets us prove (and evaluate) that the image actually informed the answer rather than the model ignoring it and reasoning from text alone.

### Output shape

```json
{
  "visual_findings": [
    "raised erythematous papules on the volar surface of the forearm",
    "linear excoriations consistent with scratching",
    "no overlying scale or vesicles"
  ],
  "systems": ["dermatological", "immunologic"],
  "possible_causes": [
    {"name": "contact dermatitis", "likelihood": "common", "description": "Skin reaction to an allergen or irritant in contact with the area."},
    {"name": "atopic dermatitis (eczema)", "likelihood": "common", "description": "Chronic itchy skin condition often flaring in flexural areas."}
  ],
  "red_flags": [
    "rapidly spreading rash with fever",
    "swelling of lips, tongue, or throat",
    "blistering or skin breakdown"
  ],
  "questions_to_prepare_for": [
    "When did the rash first appear?",
    "Have you used any new soaps, lotions, or detergents recently?",
    "Does it itch, burn, or hurt?",
    "Any recent insect bites or outdoor exposure?"
  ]
}
```

### How to Run

```bash
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
python inference.py path/to/image.jpg "I have an itchy red rash on my forearm"
```

The CLI reads the image bytes, calls Claude Sonnet 4.6 with vision, validates the response against [`OUTPUT_SCHEMA`](inference.py), and prints the JSON.

### Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Vision-language model | Claude Sonnet 4.6 via Anthropic API | Strong vision quality, sane cost, no GPU or training pipeline needed |
| Output validation | jsonschema at inference time | Hallucination containment. Invalid outputs raise, never reach the user |
| Hallucination guard | Prompt instruction: visual_findings must be observable | Defends against the model fabricating findings from the symptom text alone |
| Image format detection | Magic-byte sniffing (not file extension) | Robust to mislabeled files and Streamlit uploads with no extension |
| Auth | `.env` + python-dotenv | Standard pattern; works locally and on Streamlit Cloud secrets |

### Disclaimer

This is a research and portfolio project. It is not a medical device, is not intended for clinical use, and does not provide medical advice. Always consult a licensed healthcare provider for any medical concern.
