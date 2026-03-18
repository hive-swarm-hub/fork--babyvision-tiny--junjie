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
    # Check last line for letter
    m = re.search(r'\b([A-D])\b', answer_line)
    if m and m.group(1) in letter_map:
        return letter_map[m.group(1)]
    # Check last line for digit
    m = re.search(r'\b([0-3])\b', answer_line)
    if m:
        return m.group(1)
    # Search from end of full output
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

    # Step 1: Describe image
    description = api_call(client, model,
        [{"role": "user", "content": [hi_url,
            {"type": "text", "text": "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise."}
        ]}], temperature=0, max_tokens=512)
    if not description:
        description = api_call(client, model,
            [{"role": "user", "content": [img_url,
                {"type": "text", "text": "Describe this image in detail. Focus on layout, elements, positions, differences."}
            ]}], temperature=0, max_tokens=300)
    if not description:
        description = "(no description available)"

    # Step 2: Answer
    if ans_type == "choice" and options:
        answer, raw_output = solve_choice(client, model, question, options, description, img_url)
    else:
        answer, raw_output = solve_blank(client, model, question, description, img_url)

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


def solve_choice(client, model, question, options, description, img_url):
    """Solve choice with describe-each-option approach."""
    n = len(options)
    labels = ['A', 'B', 'C', 'D'][:n]
    all_letters = all(len(o) == 1 and o in 'ABCD' for o in options)

    if all_letters:
        prompt = f"""Here is a detailed description of the image:
{description}

{question}

The options are shown in the image as {', '.join(labels)}.

First, describe what you see in EACH option ({', '.join(labels)}) separately and in detail.
Then, explain step by step which option is correct and why, comparing each option against the requirements.
Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""
    else:
        opts = "\n".join(f"{labels[i]}. {o}" for i, o in enumerate(options))
        prompt = f"""Here is a detailed description of the image:
{description}

{question}

Options:
{opts}

First, describe what you see for each option in detail.
Then, explain step by step which option is correct and why.
Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""

    raw = api_call(client, model,
        [{"role": "user", "content": [img_url, {"type": "text", "text": prompt}]}],
        temperature=0, max_tokens=1500)
    answer = extract_choice(raw)
    return answer, raw


def solve_blank(client, model, question, description, img_url):
    """Solve blank with 3-prompt voting."""
    q_lower = question.lower()
    is_counting = any(w in q_lower for w in ["how many", "count", "pass through", "total"])

    # Prompt A: question-first
    prompt_a = f"""Question: {question}

Image analysis notes:
{description}

Look at the image carefully. Think step by step. Give your final answer in the exact format requested. Put ONLY the answer value on the last line."""

    # Prompt B: counting-specific or description-first
    if is_counting:
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

    # Prompt C: direct with image emphasis
    prompt_c = f"""{question}

I have analyzed the image and here are my notes:
{description}

Now, looking at the image again very carefully, I need to answer the question above.
Let me work through this step by step, being very precise about what I see.

My final answer (in the exact format requested, ONLY the answer value on the last line):"""

    answers = []
    raws = []
    for prompt in [prompt_a, prompt_b, prompt_c]:
        raw = api_call(client, model,
            [{"role": "user", "content": [img_url, {"type": "text", "text": prompt}]}],
            temperature=0, max_tokens=1024)
        ans = extract_blank(raw)
        answers.append(ans)
        raws.append(raw)

    # Majority vote
    counts = Counter(answers)
    winner, count = counts.most_common(1)[0]
    if count >= 2:
        answer = winner
    else:
        # No majority — prefer prompt A
        answer = answers[0]

    raw_output = f"votes={answers} winner={answer}\n{raws[0]}"
    return answer, raw_output


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
