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


def extract_choice_letter(raw_output):
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


def api_call(client, model, messages, temperature=0, max_tokens=1024):
    """API call with retry on empty."""
    for _ in range(2):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_completion_tokens=max_tokens,
            seed=42,
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

    # Step 1: Description (returns empty with detail:high + 512 tokens, acts as conversation seed)
    desc_messages = [{"role": "user", "content": [
        hi_url,
        {"type": "text", "text": "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise."},
    ]}]
    description = api_call(client, model, desc_messages, temperature=0, max_tokens=512)
    if not description:
        description = "(no description available)"

    if ans_type == "choice" and options:
        # Choice: multi-turn with letter-based answers (junjie's approach)
        n = len(options)
        labels = ['A', 'B', 'C', 'D'][:n]
        all_letters = all(len(o) == 1 and o in 'ABCD' for o in options)

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
        raw_output = api_call(client, model, messages, temperature=0, max_tokens=1500)
        answer = extract_choice_letter(raw_output)

    else:
        # Blank questions
        prompt_a = f"""Question: {question}

Image analysis notes:
{description}

Look at the image carefully. Think step by step. Give your final answer in the exact format requested. Put ONLY the answer value on the last line."""

        q_lower = question.lower()
        is_counting = any(w in q_lower for w in ["how many", "count"])
        # Grid counting: 2D grids with squares/patterns, or dot grids with lines
        is_grid = is_counting and any(w in q_lower for w in ["square", "pattern", "pass through", "point"]) and not any(w in q_lower for w in ["3d", "block", "cube"])

        if is_grid:
            # Grid transcription: model marks grid cells/points, Python counts
            if any(w in q_lower for w in ["pass through", "point"]):
                grid_prompt = f"""Look at this image carefully. The question is: {question}

The image shows dots arranged in a grid with lines connecting some of them. Your task: for EACH dot in the grid, write 'X' if the line passes through it, or '.' if it doesn't.

Write the grid of dots row by row from top to bottom, left to right. Use 'X' for dots the line passes through, '.' for dots it doesn't.
One row per line. Separate with spaces.

Be very precise — trace the line carefully through each dot."""
            else:
                grid_prompt = f"""Look at this image carefully. The question is: {question}

Your task: Transcribe the image as a grid/matrix. For EACH element in the image, write 'X' if it matches what needs to be counted, or '.' if it doesn't.

Write the grid row by row. One row per line. Use only 'X' and '.' characters separated by spaces.
Example format:
. X . X .
X X . . X
. . X . .

Be very precise — examine each cell/element carefully."""

            # Run grid transcription twice with slightly different prompts, take max
            grid_text1 = api_call(client, model, [{"role": "user", "content": [img_url, {"type": "text", "text": grid_prompt}]}], temperature=0, max_tokens=2048)
            grid_count1 = grid_text1.count('X') if grid_text1 else 0

            grid_prompt2 = grid_prompt + "\n\nIMPORTANT: Make sure you don't miss any — trace every line segment carefully."
            grid_text2 = api_call(client, model, [{"role": "user", "content": [img_url, {"type": "text", "text": grid_prompt2}]}], temperature=0.1, max_tokens=2048)
            grid_count2 = grid_text2.count('X') if grid_text2 else 0

            # Take max (model tends to under-mark)
            grid_count = max(grid_count1, grid_count2)
            grid_text = f"COUNT1={grid_count1} COUNT2={grid_count2} MAX={grid_count}\n---\n{grid_text1}\n---\n{grid_text2}"

            if grid_count > 0:
                answer = str(grid_count)
            else:
                # Fallback to baseline
                resp_a = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [hi_url, {"type": "text", "text": prompt_a}]}],
                    temperature=0.1,
                    max_completion_tokens=1024,
                )
                answer = extract_answer(resp_a.choices[0].message.content.strip(), ans_type)
            raw_output = f"GRID_COUNT={grid_count} PICKED={answer}\n---\n{grid_text}"

        elif is_counting:
            # Non-grid counting: combine multi-turn analysis + direct prompts, majority vote
            all_answers = []

            # Approach 1: Multi-turn systematic counting (3 samples)
            count_msgs = [
                {"role": "user", "content": [
                    img_url,
                    {"type": "text", "text": f"Look at this image carefully.\n\n{question}\n\nFirst, systematically locate and list each item you need to count, with its position (e.g., row and column). Be thorough — scan every row and column."},
                ]},
            ]
            analysis = api_call(client, model, count_msgs, temperature=0, max_tokens=1024)
            count_msgs.append({"role": "assistant", "content": analysis})
            count_msgs.append({"role": "user", "content": "Now count your list carefully and give the total. Put ONLY the number on the last line."})

            for _ in range(3):
                resp = client.chat.completions.create(
                    model=model, messages=count_msgs, temperature=0.3, max_completion_tokens=256,
                    seed=42,
                )
                all_answers.append(extract_answer(resp.choices[0].message.content.strip(), ans_type))

            # Approach 2: Direct prompts with detail:high (2 samples)
            resp_a = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [hi_url, {"type": "text", "text": prompt_a}]}],
                temperature=0.1,
                max_completion_tokens=1024,
                seed=42,
            )
            all_answers.append(extract_answer(resp_a.choices[0].message.content.strip(), ans_type))

            prompt_count = f"""Image description: {description}

{question}

IMPORTANT: Before giving your count, list each item you're counting with its approximate position (e.g., "row 1: item at col 2, item at col 5"). Then total them up.
Put ONLY the final count number on the last line."""
            resp_b = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [hi_url, {"type": "text", "text": prompt_count}]}],
                temperature=0.1,
                max_completion_tokens=1024,
                seed=42,
            )
            all_answers.append(extract_answer(resp_b.choices[0].message.content.strip(), ans_type))

            counts = Counter(all_answers)
            answer = counts.most_common(1)[0][0]
            raw_output = f"samples={all_answers} picked={answer}"
        else:
            # Non-counting blank: dual prompts, prefer A
            prompt_b = f"""Here is a detailed description of the image:
{description}

Now answer this question about the image:
{question}

Think step by step, then give your final answer in the exact format requested. Put your final answer on the last line, with ONLY the answer value and nothing else."""

            resp_a = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [hi_url, {"type": "text", "text": prompt_a}]}],
                temperature=0.1,
                max_completion_tokens=1024,
                seed=42,
            )
            answer_a = extract_answer(resp_a.choices[0].message.content.strip(), ans_type)

            resp_b = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [hi_url, {"type": "text", "text": prompt_b}]}],
                temperature=0.1,
                max_completion_tokens=1024,
                seed=42,
            )
            answer_b = extract_answer(resp_b.choices[0].message.content.strip(), ans_type)

            if answer_a == answer_b:
                answer = answer_a
                raw_output = resp_a.choices[0].message.content.strip()
            else:
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




