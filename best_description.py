import json

from utils import (
    MELD_DEV_JSON_PATH,
    MELD_TEST_JSON_PATH,
    MELD_TRAIN_JSON_PATH,
    read_json,
)


def generate_train_dev_test_multimodal_description_MELD(train_or_dev_or_test):
    if train_or_dev_or_test == "train":
        video_description_file = "video_description_train.json"
        audio_description_file = "audio_description_train.json"
        data_file = MELD_TRAIN_JSON_PATH
    elif train_or_dev_or_test == "dev":
        video_description_file = "video_description_dev.json"
        audio_description_file = "audio_description_dev.json"
        data_file = MELD_DEV_JSON_PATH
    elif train_or_dev_or_test == "test":
        video_description_file = "video_description_test.json"
        audio_description_file = "audio_description_test.json"
        data_file = MELD_TEST_JSON_PATH

    video_description = read_json(video_description_file)
    audio_description = read_json(audio_description_file)

    results = []

    FORWARD_WINDOW = 12
    BACKWARD_WINDOW = 1

    for data in read_json(data_file):
        conversation = data["conversation"]
        conversation_id = data["conversation_ID"]
        conversation_length = len(conversation)

        system_prompt = "Now you are an expert in sentiment and emotional analysis. The conversation enclosed between '### ###' involves multiple speakers and includes  multimodal descriptions. Your task is to thoroughly analyze the content of the dialogue, focusing on the speakers' underlying emotions. Do not rely solely on the multimodal descriptions; instead, critically evaluate the text to provide a nuanced and accurate analysis."

        for idx, utt in enumerate(conversation):
            forward_text = ""
            backward_text = ""

            start_idx = max(0, idx - FORWARD_WINDOW)

            for i in range(start_idx, idx):
                utt_id = conversation[i]["utterance_ID"]
                name = f"dia{conversation_id}_utt{utt_id}"
                speaker = conversation[i]["speaker"]
                text = conversation[i]["text"]
                tone_description = (
                    audio_description[name]
                    if len(audio_description[name].split(" ")) != 1
                    else "The speaker's tone is " + audio_description[name].lower()
                )

                forward_text += f'{speaker}: "{text}"({tone_description})\t'
            end_idx = min(conversation_length, idx + BACKWARD_WINDOW)

            for i in range(idx, end_idx):
                utt_id = conversation[i]["utterance_ID"]
                name = f"dia{conversation_id}_utt{utt_id}"
                speaker = conversation[i]["speaker"]
                text = conversation[i]["text"]
                tone_description = (
                    audio_description[name]
                    if len(audio_description[name].split(" ")) != 1
                    else "The speaker's tone is " + audio_description[name].lower()
                )

                backward_text += f'{speaker}: "{text}"({tone_description})\t'

            user_input = "### " + forward_text + backward_text.strip() + " ### "

            speaker = utt["speaker"]
            text = utt["text"]
            emotion = utt["emotion"]
            visual_description = (
                "And this video scene describes: "
                + video_description[f"dia{conversation_id}_utt{utt_id}"]
            )

            question = f'\nPlease select the emotional label of <{speaker}: "{text}"> from <neutral, surprise, fear, sadness, joy, disgust, anger>:'
            result = {
                "messages": [
                    {
                        "role": "user",
                        "content": system_prompt
                        + user_input
                        + visual_description
                        + question,
                    },
                    {"role": "assistant", "content": emotion},
                ]
            }
            results.append(result)
    with open(
        f"{train_or_dev_or_test}_MELD_multimodal.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def generate_train_dev_test_description_MELD(train_or_dev_or_test):
    if train_or_dev_or_test == "train":
        data_file = MELD_TRAIN_JSON_PATH
    elif train_or_dev_or_test == "dev":
        data_file = MELD_DEV_JSON_PATH
    elif train_or_dev_or_test == "test":
        data_file = MELD_TEST_JSON_PATH

    results = []

    FORWARD_WINDOW = 12
    BACKWARD_WINDOW = 1

    for data in read_json(data_file):
        conversation = data["conversation"]
        conversation_length = len(conversation)

        system_prompt = "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. "

        for idx, utt in enumerate(conversation):
            forward_text = ""
            backward_text = ""

            start_idx = max(0, idx - FORWARD_WINDOW)

            for i in range(start_idx, idx):
                speaker = conversation[i]["speaker"]
                text = conversation[i]["text"]

                forward_text += f'{speaker}: "{text}"\t'
            end_idx = min(conversation_length, idx + BACKWARD_WINDOW)

            for i in range(idx, end_idx):
                speaker = conversation[i]["speaker"]
                text = conversation[i]["text"]

                backward_text += f'{speaker}: "{text}"\t'

            user_input = "### " + forward_text + backward_text.strip() + " ### "

            speaker = utt["speaker"]
            text = utt["text"]
            emotion = utt["emotion"]

            question = f'\nPlease select the emotional label of <{speaker}: "{text}"> from <neutral, surprise, fear, sadness, joy, disgust, anger>:'
            result = {
                "messages": [
                    {
                        "role": "user",
                        "content": system_prompt + user_input + question,
                    },
                    {"role": "assistant", "content": emotion},
                ]
            }
            results.append(result)
    with open(f"{train_or_dev_or_test}_MELD.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
