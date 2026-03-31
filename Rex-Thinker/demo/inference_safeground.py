import argparse
import json
import os
import torch
from collections import Counter

from groundingdino.util.inference import load_model
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tools.inference_tools import (
    convert_boxes_from_absolute_to_qwen25_format,
    inference_gdino,
    parse_json,
    find_best_matched_index,
    visualize
)

from discrete_uq import compute_ucom, compute_ta, compute_ie, compute_cd

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for Qwen-2.5-VL with SafeGround UQ")
    parser.add_argument("--image_path", type=str, default="demo/example_images/demo_helmet.png")
    parser.add_argument("--cate_name", type=str, default="helmet")
    parser.add_argument("--ref_exp", type=str, default="the forth helmet from left")
    parser.add_argument("--vis_path", type=str, default="vis/example_output_sg.jpg")
    parser.add_argument("--model_path", type=str, default="IDEA-Research/Rex-Thinker-GRPO-7B")
    parser.add_argument("--gdino_config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino_weights", type=str, default="GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of multi-samples for UQ")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for stochastic decoding")
    parser.add_argument("--tau", type=float, default=None, help="Rejection threshold (if UCOM > tau, reject)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    gdino_model = load_model(args.gdino_config, args.gdino_weights)
    gdino_model.eval()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=16 * 28 * 28, max_pixels=1280 * 28 * 28
    )

    image = Image.open(args.image_path).convert("RGB")
    prompts = [args.cate_name]

    gdino_boxes = inference_gdino(image, prompts, gdino_model)
    proposed_box = convert_boxes_from_absolute_to_qwen25_format(
        gdino_boxes, image.width, image.height
    )
    hint = json.dumps({f"{args.cate_name}": proposed_box})
    question = f"Hint: Object and its coordinates in this image: {hint}\nPlease detect {args.ref_exp} in the image."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    print(f"Running {args.num_samples} samples with temperature {args.temperature}...")
    sampled_indices = []
    
    for i in range(args.num_samples):
        generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=True, temperature=args.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse output box and find matched DINO index
        w, h = image.size
        try:
            json_output = parse_json(output_text)
            if len(json_output) > 0:
                box = json_output[0] # Take first box if multiple
                x0, y0, x1, y1 = box
                x0 = x0 / input_width.item() * w
                y0 = y0 / input_height.item() * h
                x1 = x1 / input_width.item() * w
                y1 = y1 / input_height.item() * h
                
                final_boxes = [[x0, y0, x1, y1]]
                ref_labels = find_best_matched_index(final_boxes, gdino_boxes)
                matched_idx = ref_labels[0] # 1-based index (or None if no match)
                sampled_indices.append(matched_idx)
            else:
                sampled_indices.append(-1) # Fallback for unparseable
        except Exception as e:
            print(f"Sample {i} parsing failed: {e}")
            sampled_indices.append(-1)
            
    # Calculate Probabilities
    counts = Counter(sampled_indices)
    probs = [count / args.num_samples for idx, count in counts.items()]
    
    # Calculate UQ
    ta = compute_ta(probs)
    ie = compute_ie(probs)
    cd = compute_cd(probs)
    ucom = compute_ucom(probs, w_ta=0.4, w_ie=0.3, w_cd=0.3)
    
    print("\n--- Uncertainty Metrics ---")
    print(f"Top-Candidate Ambiguity (TA): {ta:.4f}")
    print(f"Informational Dispersion (IE): {ie:.4f}")
    print(f"Concentration Deficit (CD): {cd:.4f}")
    print(f"Combined Uncertainty (UCOM): {ucom:.4f}")
    
    # Decision
    most_common_idx = counts.most_common(1)[0][0]
    is_rejected = False
    
    if args.tau is not None:
        if ucom > args.tau:
            print(f">>> DECISION: REJECTED (UCOM {ucom:.4f} > tau {args.tau})")
            is_rejected = True
        else:
            print(f">>> DECISION: ACCEPTED (UCOM {ucom:.4f} <= tau {args.tau})")
    
    if not is_rejected and most_common_idx != -1 and most_common_idx is not None:
        print(f"Final chosen GDINO box index: {most_common_idx}")
        # Visualize the chosen box
        chosen_box = gdino_boxes[most_common_idx - 1]
        vis_image = visualize(
            image.copy(),
            [chosen_box],
            [1.0],
            labels=[str(most_common_idx)],
            font_size=20,
            draw_width=10,
        )
        os.makedirs(os.path.dirname(args.vis_path), exist_ok=True)
        vis_image.save(args.vis_path)
        print(f"Saved visualization to {args.vis_path}")
    else:
        print("Model rejected the prediction or no valid box was found. Not saving visualization.")
