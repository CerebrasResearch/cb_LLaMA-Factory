import pandas as pd
import argparse
import os
import logging
import json
import glob
from typing import List, Dict, Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def build_combined_df(folder_path):
    
    # Initialize an empty dictionary to hold the value counts
    value_counts_dict = {}
    max_reward = float('-inf')

    reward_files = []
    for filename in os.listdir(folder_path):
        if filename.startswith("reward_") and filename.endswith(".jsonl"):
            reward_files.append(filename)
    
    reward_files = sorted(reward_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Loop through the files in the specified folder
    for filename in reward_files:
        # Read the JSON file
        b = pd.read_json(os.path.join(folder_path, filename), lines=True)
        # Get the value counts of "problem"
        problem_counts = b["problem"].value_counts()
        # Extract the number from the filename
        number = filename.split("_")[1].split(".")[0]
        max_reward = max(max_reward, int(number))
        # Store the counts in the dictionary
        value_counts_dict[int(number)] = problem_counts
        value_counts_dict[f"{number}_plans"] = b.groupby("problem")["plan_id"].agg(list)

    # Create a new dataframe from the dictionary
    problem_counts_df = pd.DataFrame(dict([(k, v) for k, v in value_counts_dict.items()])).fillna(0)
    _vkeys = list(value_counts_dict.keys())
    _vkeys = [_vkey for _vkey in _vkeys if not str(_vkey).endswith('_plans')] + [_vkey for _vkey in _vkeys if str(_vkey).endswith('_plans')]
    problem_counts_df = problem_counts_df[_vkeys]

    for col in problem_counts_df.columns:
        if str(col).endswith('plans'):
            problem_counts_df[col] = problem_counts_df[col].replace(0.0, None)

    def create_value_list(row, max_reward):
        value_list = []
        for i in range(0, max_reward + 1):
            value_list.append(row.get(i, 0))  # Use get to avoid KeyError if key doesn't exist
        return value_list

    del b
    problem_counts_df['value_counts_list'] = problem_counts_df.apply(lambda row: create_value_list(row, max_reward), axis=1)
    
    return problem_counts_df, max_reward


def add_sample_plan_filter(problem_counts_df, max_reward, atleast_atmost_cond1, atleast_atmost_cond2, num_plans_cond1, num_plans_cond2, and_or_op, num_rewards_1, num_rewards_2):

    def choose_fcn(row, max_reward=max_reward):
        value_counts_list = var1 = row.value_counts_list
        op_dict = {"atleast": ">=", "atmost": "<="}
        op1 = op_dict[atleast_atmost_cond1]
        op2 = op_dict[atleast_atmost_cond2]

        # process condition 1:
        bool_cond1_str  = f"sum(var1[0:{num_rewards_1}]) {op1} {num_plans_cond1}"
        bool_cond1 = eval(bool_cond1_str)

        # process condition 2:
        bool_cond2_str  = f"sum(var1[{num_rewards_2}:{max_reward+1}]) {op2} {num_plans_cond2}"
        bool_cond2 = eval(bool_cond2_str)

        final_bool_str = f"{bool_cond1}, {bool_cond1_str} {and_or_op} {bool_cond2_str} {bool_cond2}"
        final_bool = eval(f"{bool_cond1} {and_or_op} {bool_cond2}")

        return final_bool_str, final_bool


    problem_counts_df[["choose_str", "choose"]] = problem_counts_df.apply(lambda row: choose_fcn(row, max_reward), axis=1, result_type='expand')

    return problem_counts_df


def convert_json_format_to_sharegpt(input_data):
    """
    Converts JSON from the original format to the conversation format.
    
    Args:
        input_data (dict): The input JSON data with the original structure
        
    Returns:
        dict: The converted JSON in the new conversation format
    """
    conversations = []
    images = []
    
    # Extract conversation data
    conversation = input_data.get("conversation", {})
    
    # Process messages in the order they appear in the dictionary
    for key, data in conversation.items():
        
        if key.startswith("user"):
            # Process user message
            user_text_parts = []
            
            # Handle different formats: list of dicts or just string
            if isinstance(data, list):
                # List of message objects (all dicts)
                for msg in data:
                    if msg.get("type") == "image_url":
                        # Extract image URL/data
                        image_url = msg.get("image_url", {}).get("url", "")
                        if image_url:
                            images.append(image_url)
                            user_text_parts.append("<image>")
                    elif msg.get("type") == "text":
                        user_text_parts.append(msg.get("text", ""))
            elif isinstance(data, str):
                # Just a string
                user_text_parts.append(data)
            
            # Combine user text
            user_text = "".join(user_text_parts)
            if user_text:
                conversations.append({
                    "role": "user",
                    "content": user_text
                })
        
        elif key.startswith("assistant"):
            # Process assistant message
            if isinstance(data, str) and data:
                conversations.append({
                    "role": "assistant",
                    "content": data
                })
            elif isinstance(data, dict):
                # Handle dict format for assistant (extract text if needed)
                text = data.get("text", str(data))
                if text:
                    conversations.append({
                        "role": "assistant",
                        "content": text
                    })
    
    # Generate loss mask based on the last assistant message
    loss_mask = []
    last_assistant_message = None
    
    # Find the last assistant message
    for conv in reversed(conversations):
        if conv["role"] == "assistant":
            last_assistant_message = conv["content"]
            break
    
    # Determine loss mask logic
    if last_assistant_message and last_assistant_message.startswith("No the plan is not good"):
        # Only apply loss to the last assistant message
        for i, conv in enumerate(conversations):
            if conv["role"] == "assistant" and i == len(conversations) - 1:
                loss_mask.append(1)  # Last assistant message gets loss
            else:
                loss_mask.append(0)  # All others get no loss
    else:
        # Apply loss to all assistant messages
        for conv in conversations:
            if conv["role"] == "assistant":
                loss_mask.append(1)  # Assistant messages get loss
            else:
                loss_mask.append(0)  # User messages get no loss
    
    # Build the result
    result = {
        "messages": conversations,
        "loss_mask": loss_mask
    }
    
    # Only add images array if there are images
    if images:
        result["images"] = images
    
    return result


def process_jsonl_files(csv_file_path: str, jsonl_folder_path: str, output_file_path: str = None) -> List[Dict[Any, Any]]:
    """
    Process JSONL files and filter based on problems present in CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        jsonl_folder_path (str): Path to folder containing JSONL files (reward_*.jsonl)
        output_file_path (str, optional): Path to save the processed results as JSONL
        
    Returns:
        List[Dict]: List of converted JSON objects
    """
    
    # Read CSV file and get the set of problems
    try:
        df = pd.read_csv(csv_file_path)
        csv_problems = set(df['problem'].tolist())
        print(f"Found {len(csv_problems)} problems in CSV file")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    # Find all JSONL files matching the pattern
    jsonl_pattern = os.path.join(jsonl_folder_path, "reward_*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    if not jsonl_files:
        print(f"No JSONL files found matching pattern: {jsonl_pattern}")
        return []
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    converted_results = []
    total_processed = 0
    total_filtered = 0
    
    # Process each JSONL file
    for jsonl_file in jsonl_files:
        print(f"Processing: {jsonl_file}")
        
        try:
            content = pd.read_json(jsonl_file, lines=True)
            content = content.to_json(orient='records')
            

            if content:
                # Parse as JSON array
                json_data = json.loads(content)
                
                # Process each item in the array
                for item in json_data:
                    total_processed += 1
                    
                    # Check if problem exists in CSV
                    problem = item.get("problem", "")
                    if problem in csv_problems:
                        total_filtered += 1
                        
                        # Convert the format
                        converted_item = convert_json_format_to_sharegpt(item)
                        
                        # Add original metadata
                        converted_item["problem"] = problem
                        converted_item["plan_id"] = item.get("plan_id", "")
                        converted_item["answer"] = item.get("answer", "")
                        converted_item["reward_file"] = jsonl_file
                        
                        converted_results.append(converted_item)

                        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {jsonl_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {jsonl_file}: {e}")
            continue
    
    print(f"Total items processed: {total_processed}")
    print(f"Items matching CSV problems: {total_filtered}")
    print(f"Converted results: {len(converted_results)}")
    
    # Save results if output path is provided
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=4)
            print(f"Results saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    return converted_results




def parse_arguments():
    parser = argparse.ArgumentParser(description="Process problem counts and apply sample plan filters.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing reward JSON files.")
    parser.add_argument("--atleast_atmost_cond1", type=str, choices=["atleast", "atmost"], required=False, help="Condition 1: 'atleast' or 'atmost'., default= 'atleast'", default="atleast")
    parser.add_argument("--atleast_atmost_cond2", type=str, choices=["atleast", "atmost"], required=False, help="Condition 2: 'atleast' or 'atmost' default= 'atleast'", default="atleast")
    parser.add_argument("--num_plans_cond1", type=int, required=False, help="Number of plans for condition 1.", default=1)
    parser.add_argument("--num_plans_cond2", type=int, required=False, help="Number of plans for condition 2.", default=1)
    parser.add_argument("--and_or_op", type=str, choices=["and", "or"], required=False, help="Logical operator: 'and' or 'or'.", default="and")
    parser.add_argument("--num_rewards_1", type=int, required=False, help="Number of rewards for condition 1.", default=6)
    parser.add_argument("--num_rewards_2", type=int, required=False, help="Number of rewards for condition 2.", default=8)
    parser.add_argument("--output_folder", type=str, required=False, help="Output folder to save the files.", default="output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    problem_counts_df, max_reward = build_combined_df(args.folder_path)
    logging.info(f"Max reward found: {max_reward}")
    logging.info(f"Problem counts DataFrame shape: {problem_counts_df.shape}")
    logging.info(f"Running filtering with conditions: Choose samples with {args.atleast_atmost_cond1} {args.num_plans_cond1} <= {args.num_rewards_1} {args.and_or_op} {args.atleast_atmost_cond2} {args.num_plans_cond2} >={args.num_rewards_2}")
    problem_counts_df = add_sample_plan_filter(
        problem_counts_df,
        max_reward,
        args.atleast_atmost_cond1,
        args.atleast_atmost_cond2,
        args.num_plans_cond1,
        args.num_plans_cond2,
        args.and_or_op,
        args.num_rewards_1,
        args.num_rewards_2
    )
    chosen_problems = problem_counts_df[problem_counts_df['choose']]
    logging.info(f"Samples {len(problem_counts_df[problem_counts_df['choose']])}/{len(problem_counts_df)} selected based on the filter criteria.")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    chosen_csv_path = os.path.join(args.output_folder, "chosen_problems.csv")
    chosen_problems.to_csv(chosen_csv_path)

    logging.info(f"Chosen problems saved to {chosen_csv_path}")
    # Convert to ShareGPT format and save
    
    output_file_path = os.path.join(args.output_folder, "converted_results.json")
    process_jsonl_files(chosen_csv_path, args.folder_path, output_file_path)

    













    
