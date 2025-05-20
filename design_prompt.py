import json
import re
from pathlib import Path

iemocap_path = Path("")

meld_path = Path("")

system_prompt = """You are an expert sentiment analysis. Please determine if the target utterance needs to rely on image and audio caption for emotion categorization according to the following steps:

Decision Flow:
1. Text Self-Sufficiency: Can emotion be clearly determined from text alone?
2. Modality Supplementation: Do audio features or visual cues provide additional/clarifying evidence?
3. Conflict Check: Do multimodal signals contradict textual content?

Conversation:
{conversation}
Target Utterance: <{target_utterance}>

Output Rules:
Keep Captions if ANY applies:
1. Text is emotionally ambiguous
2. Modalities enhance/override text interpretation

Discard Captions ONLY if:
1. Text contains strong emotional markers (explicit words/punctuation)
2. Modalities add no diagnostic value

Output Format:
1. Is it recommended that the Caption be retained? (Yes, the captions should remain/No, the captions should be removed)
2. Brief reasons (in 50 words or less):
"""


def extract_content_between_hashtags(text):
    pattern = r"###\s*(.*?)\s*###"
    match = re.findall(pattern, text, re.DOTALL)
    if match:
        return match[1].strip()
    else:
        text_split = text.split("###")
        return text_split[3].strip()


def extract_content_between_angle_brackets(text):
    pattern = r"<(.*?)>"
    matches = re.findall(pattern, text)
    return matches[0].strip()


def add_audio_caption(text):
    # 查找引号后紧跟的括号及其内容
    pattern = r'"(.*?)"(\([^)]*\))'

    def replace_function(match):
        quote_content = match.group(1)
        bracket_content = match.group(2)
        # 在括号内容前添加"Audio caption:"
        modified_bracket = bracket_content.replace("(", "(Audio caption: ", 1)
        return f'"{quote_content}"{modified_bracket}'

    return re.sub(pattern, replace_function, text)


def iemocap_judge_prompt(path, dataset_type):
    path = path / (dataset_type + ".json")
    messages = []
    with open(path, "r") as f:
        data = json.load(f)

    for d in data:
        instruction = d["instruction"]
        target_emotion = d["output"]
        # conversation 在 ### ### 之间
        conversation = extract_content_between_hashtags(instruction)
        conversation = add_audio_caption(conversation)
        target_utterance = extract_content_between_angle_brackets(instruction)
        # 将 conversation 和 target_utterance 替换到 system_prompt 中
        prompt = system_prompt.format(
            conversation=conversation, target_utterance=target_utterance
        )
        user = {"role": "user", "content": prompt}
        assistant = {"role": "assistant", "content": target_emotion}
        message = [user, assistant]
        messages.append({"messages": message, "solution": target_emotion})

    with open(dataset_type + ".json", "w") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


def meld_judge_prompt(path, dataset_type):
    path = path / (dataset_type + "_MELD.json")
    messages = []
    with open(path, "r") as f:
        data = json.load(f)

    for d in data:
        instruction = d["messages"]
        content = instruction[0]["content"]
        target_emotion = instruction[1]["content"]

        visual_description = (
            content.split("###")[-1]
            .split("Please select the emotional label of")[0]
            .strip()
        )

        conversation = extract_content_between_hashtags(content)

        conversation = (
            add_audio_caption(conversation) + "\nImage caption: " + visual_description
        )
        target_utterance = extract_content_between_angle_brackets(content)
        prompt = system_prompt.format(
            conversation=conversation, target_utterance=target_utterance
        )
        user = {"role": "user", "content": prompt}
        assistant = {"role": "assistant", "content": target_emotion}
        message = [user, assistant]
        messages.append({"messages": message, "solution": target_emotion})

    with open(dataset_type + ".json", "w") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


def iemocap_emotion_prompt(path, dataset_type):
    path = path / (dataset_type + ".json")
    messages = []
    with open(path, "r") as f:
        data = json.load(f)

    for d in data:
        instruction = d["instruction"]
        target_emotion = d["output"]
        user = {"role": "user", "content": instruction}
        assistant = {"role": "assistant", "content": target_emotion}
        message = [user, assistant]
        messages.append({"messages": message, "solution": target_emotion})

    with open(dataset_type + ".json", "w") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


def meld_emotion_prompt(path, dataset_type):
    path = path / (dataset_type + "_MELD_multimodal.json")
    messages = []
    with open(path, "r") as f:
        data = json.load(f)

    for d in data:
        instruction = d["messages"]
        content = instruction[0]["content"]
        target_emotion = instruction[1]["content"]

        user = {"role": "user", "content": content}
        assistant = {"role": "assistant", "content": target_emotion}
        message = [user, assistant]

        messages.append({"messages": message, "solution": target_emotion})

    with open(dataset_type + ".json", "w") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

