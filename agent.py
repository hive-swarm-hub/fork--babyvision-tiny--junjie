"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import base64
import re
import io
from collections import Counter

from openai import OpenAI
from PIL import Image


def load_image_b64(image_path: str, min_size: int = 768) -> str:
    """Load image, upscale if too small, return base64."""
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) < min_size:
        scale = min_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def extract_choice(raw_output):
    """Extract choice answer: letter -> 0-indexed."""
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    answer_line = lines[-1] if lines else raw_output
    letter_map = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}
    m = re.search(r'\b([A-D])\b', answer_line)
    if m and m.group(1) in letter_map:
        return letter_map[m.group(1)]
    m = re.search(r'\b([0-3])\b', answer_line)
    if m:
        return m.group(1)
    for line in reversed(lines):
        m = re.search(r'\b([A-D])\b', line)
        if m and m.group(1) in letter_map:
            return letter_map[m.group(1)]
    return answer_line


def extract_blank(raw_output):
    """Extract blank answer from last line."""
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    answer = lines[-1] if lines else raw_output
    answer = re.sub(r'\s*,\s*', ',', answer)
    answer = answer.rstrip('.')
    return answer


def api_call(client, model, messages, temperature=0, max_tokens=1024):
    """API call with retry on empty."""
    for _ in range(2):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_completion_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        if content and content.strip():
            return content.strip()
    return ""


def solve(question: str, image_path: str, ans_type: str, options: list) -> str:
    client = OpenAI()
    img_b64 = load_image_b64(image_path)
    img_url = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
    hi_url = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")

    # Step 1: Describe image (multi-turn: this becomes conversation context)
    desc_messages = [{"role": "user", "content": [hi_url,
        {"type": "text", "text": "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise."}
    ]}]
    description = api_call(client, model, desc_messages, temperature=0, max_tokens=2048)
    if not description:
        # Retry without detail:high
        desc_messages = [{"role": "user", "content": [img_url,
            {"type": "text", "text": "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise."}
        ]}]
        description = api_call(client, model, desc_messages, temperature=0, max_tokens=2048)
    if not description:
        description = "(no description available)"

    # Step 2: Multi-turn answer with image + description context
    if ans_type == "choice" and options:
        answer, raw_output = solve_choice(client, model, question, options, description, img_url, hi_url, desc_messages)
    else:
        answer, raw_output = solve_blank(client, model, question, description, img_url, desc_messages)

    # Save trajectory
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump({
                "index": int(idx), "model": model, "description": description,
                "question": question, "image_path": image_path,
                "ans_type": ans_type, "options": options,
                "raw_response": raw_output, "parsed_answer": answer,
            }, f, indent=2)

    return answer


def solve_choice(client, model, question, options, description, img_url, hi_url, desc_messages):
    """Solve choice using multi-turn conversation."""
    n = len(options)
    labels = ['A', 'B', 'C', 'D'][:n]
    all_letters = all(len(o) == 1 and o in 'ABCD' for o in options)

    # Build multi-turn: description as first turn, answer as second
    messages = list(desc_messages)
    messages.append({"role": "assistant", "content": description})

    if all_letters:
        answer_prompt = f"""Now answer this question about the image:
{question}

The options are shown in the image as {', '.join(labels)}.

First, describe what you see in EACH option ({', '.join(labels)}) separately and in detail.
Then, explain step by step which option is correct and why, comparing each option against the requirements.
Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""
    else:
        opts = "\n".join(f"{labels[i]}. {o}" for i, o in enumerate(options))
        answer_prompt = f"""Now answer this question about the image:
{question}

Options:
{opts}

First, describe what you see for each option in detail.
Then, explain step by step which option is correct and why.
Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""

    messages.append({"role": "user", "content": [img_url, {"type": "text", "text": answer_prompt}]})

    raw = api_call(client, model, messages, temperature=0, max_tokens=1500)
    answer = extract_choice(raw)
    return answer, raw


def solve_blank(client, model, question, description, img_url, desc_messages):
    """Solve blank with 2 prompts + multi-turn approach."""
    q_lower = question.lower()
    is_counting = any(w in q_lower for w in ["how many", "count", "pass through", "total"])

    # Approach 1: Multi-turn (description context + answer)
    messages = list(desc_messages)
    messages.append({"role": "assistant", "content": description})

    if is_counting:
        answer_prompt = f"""{question}

Count methodically:
1. Identify exactly what needs to be counted
2. Go row by row (or section by section), listing each item with its position
3. Sum up the total
4. Double-check by counting again

Put ONLY the final count number on the last line."""
    else:
        answer_prompt = f"""{question}

Think step by step. Pay close attention to the exact format requested in the question.
Give your final answer in the exact format requested. Put ONLY the answer value on the last line."""

    messages.append({"role": "user", "content": [img_url, {"type": "text", "text": answer_prompt}]})
    raw_a = api_call(client, model, messages, temperature=0, max_tokens=1024)
    answer_a = extract_blank(raw_a)

    # Approach 2: Single-turn with different framing
    prompt_b = f"""Question: {question}

Image analysis notes:
{description}

Look at the image carefully. Think step by step. Give your final answer in the exact format requested. Put ONLY the answer value on the last line."""

    raw_b = api_call(client, model,
        [{"role": "user", "content": [img_url, {"type": "text", "text": prompt_b}]}],
        temperature=0, max_tokens=1024)
    answer_b = extract_blank(raw_b)

    if answer_a == answer_b:
        return answer_a, raw_a

    # Tiebreak: for counting prefer higher (models undercount), otherwise prefer multi-turn
    if is_counting:
        try:
            va, vb = int(answer_a), int(answer_b)
            if va >= vb:
                return answer_a, f"A={answer_a} B={answer_b} PICKED=A(multi-turn)"
            else:
                return answer_b, f"A={answer_a} B={answer_b} PICKED=B(higher)"
        except ValueError:
            pass

    return answer_a, f"A={answer_a} B={answer_b} PICKED=A(multi-turn)"


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
