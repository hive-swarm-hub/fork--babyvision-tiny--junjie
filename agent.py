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


def extract_answer(raw_output, ans_type):
    """Extract and clean the answer from model output."""
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    answer = lines[-1] if lines else raw_output
    answer = re.sub(r'\s*,\s*', ',', answer)
    answer = answer.rstrip('.')
    if ans_type == "choice":
        m = re.search(r'\b([0-4])\b', answer)
        if m:
            answer = m.group(1)
    return answer


def solve(question: str, image_path: str, ans_type: str, options: list) -> str:
    client = OpenAI()

    img_b64 = load_image_b64(image_path)
    img_url = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}

    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")

    # Step 1: Describe the image (high detail for better perception)
    hi_url = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
    desc_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            hi_url,
            {"type": "text", "text": "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise."},
        ]}],
        temperature=0,
        max_completion_tokens=512,
    )
    description = desc_response.choices[0].message.content
    if not description or not description.strip():
        desc_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                hi_url,
                {"type": "text", "text": "Describe this image in detail. Focus on layout, elements, positions, differences."},
            ]}],
            temperature=0,
            max_completion_tokens=300,
        )
        description = desc_response.choices[0].message.content
    description = description.strip() if description else "(no description available)"

    # Step 2: Two answer attempts with different prompt styles
    if ans_type == "choice" and options:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        n = len(options) - 1
        prompt_a = f"""Here is a detailed description of the image:
{description}

Now answer this question about the image:
{question}

Options:
{opts}

Think step by step, then give your final answer as ONLY the option number (0, 1, 2, or {n}). Put your final answer on the last line."""
        # For choice, just use one attempt (model bias is consistent)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [img_url, {"type": "text", "text": prompt_a}]}],
            temperature=0,
            max_completion_tokens=1024,
        )
        raw_output = response.choices[0].message.content.strip()
        answer = extract_answer(raw_output, ans_type)
    else:
        # For blank questions: try two different framings
        prompt_a = f"""Question: {question}

Image analysis notes:
{description}

Look at the image carefully. Think step by step. Give your final answer in the exact format requested. Put ONLY the answer value on the last line."""

        q_lower = question.lower()
        if any(w in q_lower for w in ["how many", "count"]):
            prompt_b = f"""Image description: {description}

{question}

IMPORTANT: Before giving your count, list each item you're counting with its approximate position (e.g., "row 1: item at col 2, item at col 5"). Then total them up.
Put ONLY the final count number on the last line."""
        else:
            prompt_b = f"""Here is a detailed description of the image:
{description}

Now answer this question about the image:
{question}

Think step by step, then give your final answer in the exact format requested. Put your final answer on the last line, with ONLY the answer value and nothing else."""

        resp_a = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [img_url, {"type": "text", "text": prompt_a}]}],
            temperature=0.1,
            max_completion_tokens=1024,
        )
        answer_a = extract_answer(resp_a.choices[0].message.content.strip(), ans_type)

        resp_b = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [img_url, {"type": "text", "text": prompt_b}]}],
            temperature=0.1,
            max_completion_tokens=1024,
        )
        answer_b = extract_answer(resp_b.choices[0].message.content.strip(), ans_type)

        if answer_a == answer_b:
            answer = answer_a
            raw_output = resp_a.choices[0].message.content.strip()
        else:
            # When answers disagree, prefer prompt A (question-first, better for counting)
            answer = answer_a
            raw_output = f"A={answer_a} B={answer_b} PICKED=A"

    # Save trajectory
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        trajectory = {
            "index": int(idx),
            "model": model,
            "description": description,
            "question": question,
            "image_path": image_path,
            "ans_type": ans_type,
            "options": options,
            "raw_response": raw_output,
            "parsed_answer": answer,
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)

    return answer


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
