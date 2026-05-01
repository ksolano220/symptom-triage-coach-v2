## Symptom Triage Coach v2 (Multi-modal)

Live demo: https://symptom-triage-coach-v2.streamlit.app/

Pre-visit prep assistant. Takes an image of the affected area plus a plain-language symptom description, returns a structured triage JSON: image-grounded visual findings, body systems involved, ranked possible causes, red flags, and questions the doctor will likely ask.

This is the multi-modal extension of [`symptom-triage-coach`](https://github.com/ksolano220/symptom-triage-coach). v1 was a LoRA fine-tune of Qwen2.5-1.5B for text-only triage. v2 keeps the same output schema and adds an image input via Claude's vision API.

Not a diagnostic tool. Pre-visit prep only.

### What's different vs v1

| | v1 | v2 |
|---|---|---|
| Input | Symptom text only | Image + symptom text |
| Model | Qwen2.5-1.5B-Instruct + LoRA adapter | Claude Sonnet 4.6 (vision) via API |
| New schema field | (none) | `visual_findings`: clinical observations grounded in the image |
| UI | Gradio on Hugging Face Spaces | Streamlit Cloud |

The new `visual_findings` field is the image-grounding signal. It forces the model to surface what's actually visible in the image as a separate output, which makes it possible to evaluate whether the image informed the answer or whether the model ignored it and reasoned from text alone.

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

### Run it

Install and add your API key:

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add ANTHROPIC_API_KEY=sk-ant-...
```

CLI:

```bash
python inference.py path/to/image.jpg "I have an itchy red rash on my forearm"
```

Streamlit app:

```bash
streamlit run app.py
```

### Eval

Three cases at `eval/cases.json`, all using the same dermatology image with three different symptom descriptions to probe specific behaviors. Each case runs through both the multi-modal path (`triage`) and a text-only baseline (`triage_text_only`) so the contribution of the image is isolated.

```bash
python eval/run.py
```

Sample run at [`eval/sample_results.json`](eval/sample_results.json):

| case | image+text top causes | text-only top causes |
|---|---|---|
| `derm-consistent` (image + text agree) | Allergic Contact Dermatitis, Irritant Contact Dermatitis, Urticaria | Contact Dermatitis (Irritant), Allergic Contact Dermatitis, Poison Ivy / Oak / Sumac |
| `derm-vague` (vague text, image-grounding test) | Urticaria, Dermatographism, Allergic Contact Dermatitis | Soft tissue injury, Fracture, Skin infection / cellulitis |
| `derm-conflict` (text describes cardiac emergency, image shows rash) | Acute MI, Anaphylaxis with cardiac involvement, Unstable Angina | Acute MI, Unstable Angina, Aortic Dissection |

Two readings:

- `derm-vague`: when the patient text is uninformative ("something is wrong with my arm, it looks bad"), the text-only path drifts to fractures and bruising. The multi-modal path correctly anchors on dermatologic causes from the image.
- `derm-conflict`: text describes a cardiac emergency, image shows a rash. Both paths correctly prioritize the cardiac differential. Only the multi-modal path also surfaces the dermatologic finding ("Anaphylaxis with cardiac involvement", `dermatological` listed as a system), which is the actually-safe behavior. The text-only path is blind to the rash.

Three cases is illustrative, not statistically convincing. Adding more cases is a matter of dropping images into `eval/cases/` and rows into `cases.json`.

### Architecture

| Component | Choice |
|---|---|
| Vision-language model | Claude Sonnet 4.6 via Anthropic API |
| Output validation | jsonschema, called at inference time. Invalid outputs raise. |
| Image format detection | Magic-byte sniffing, not file extension |
| Auth | `.env` locally, `st.secrets` on Streamlit Cloud |

### Disclaimer

Research and portfolio project. Not a medical device, not for clinical use, does not provide medical advice. Consult a licensed healthcare provider for any medical concern.
