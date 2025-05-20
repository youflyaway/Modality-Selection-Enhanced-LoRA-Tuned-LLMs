import argparse
import json
import pickle

multimodal_description = json.load(
    open(
        "multimodal_description.json"
    )
)


def process_dataset(dataset, window=110):
    """
    dataset: parameter that define the evaluated dataset.
    window:       parameter that control the historical context window
    speaker_mode: parameter that control the speaker identification task whether to be added in the main task

    data_path:    parameter that record the processed dataset's filepath
    """
    label_set = {
        "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
        "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
        "EmoryNLP": [
            "Joyful",
            "Mad",
            "Peaceful",
            "Neutral",
            "Sad",
            "Powerful",
            "Scared",
        ],
    }
    label_text_set = {
        "iemocap": "happy, sad, neutral, angry, excited, frustrated",
        "meld": "neutral, surprise, fear, sad, joyful, disgust, angry",
        "EmoryNLP": "Joyful, Mad, Peaceful, Neutral, Sad, Powerful, Scared",
    }
    speaker_label_text_set = {
        "iemocap": "Speaker_0, Speaker_1",
        "meld": "Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8",
        "EmoryNLP": "Speaker_0, Speaker_1, Speaker_2, Speaker_3, Speaker_4, Speaker_5, Speaker_6, Speaker_7, Speaker_8",
    }

    speaker_label_dict = {}
    content_target_dict = {}
    speaker_target_dict = {}
    content_task_dict = {}
    speaker_task_dict = {}
    sentence_dict = {}
    num_2speaker = {0: "M", 1: "F"}
    data = pickle.load(
        open(
            f"{dataset}.pkl",
            "rb",
        )
    )

    # 不同的数据集有不同的speaker_label的处理方式
    # Different datasets have different ways of handling speaker_label
    if dataset == "iemocap":
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                if speaker_label == "M":
                    temp_speaker_list.append(0)
                else:
                    temp_speaker_list.append(1)
            speaker_label_dict[conv_id] = temp_speaker_list

    # 对conversation的utterance进行处理，其中index_w用于处理窗口大小设置下面的起始index
    # Process the utterances in the conversation, where 'index_w' is used to handle the starting index under the window size setting.
    for conv_id in all_conv_id:
        man = 0
        woman = 0
        descriptions = []

        for speaker_label in speaker_label_dict[conv_id]:
            gender = num_2speaker[speaker_label]
            if gender == "M":
                utt_name = conv_id + "_" + gender + f"{man:>03d}"
                man += 1
            else:
                utt_name = conv_id + "_" + gender + f"{woman:>03d}"
                woman += 1
            descriptions.append(multimodal_description[utt_name])

        for conv_turn in range(len(sentence_dict[conv_id])):
            temp_content_str = "Now you are an expert in sentiment and emotional analysis. The conversation enclosed between '### ###' involves multiple speakers and includes  multimodal descriptions. Your task is to thoroughly analyze the content of the dialogue, focusing on the speakers' underlying emotions. Do not rely solely on the multimodal descriptions; instead, critically evaluate the text to provide a nuanced and accurate analysis.### "

            index_w = max(conv_turn - window, 0)

            for speaker_label, sub_sent, description in zip(
                speaker_label_dict[conv_id][index_w : conv_turn + 1],
                sentence_dict[conv_id][index_w : conv_turn + 1],
                descriptions[index_w : conv_turn + 1],
            ):
                temp_content_str += (
                    f'Speaker_{speaker_label}:"{sub_sent}"({description})\t'
                )
                # temp_id_task_str += (f'\t Speaker_?:"{sub_sent}"')
                target_utterance = f'Speaker_{speaker_label}:"{sub_sent}"'

            content_target_dict[f"{conv_id}_{conv_turn}"] = label_set[dataset][
                data[1][conv_id][conv_turn]
            ]

            temp_content_str += " ### "
            temp_content_str += f"Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset]}>:"
            # demon
            # temp_content_str += f'Here is a demonstration. {demon}'
            content_task_dict[f"{conv_id}_{conv_turn}"] = temp_content_str

    for conv_id in all_conv_id:
        for conv_turn in range(len(data[2][conv_id])):
            temp_speaker_task_str = f"Now you are expert of sentiment and emotional analysis. Please select the Speaker label of the utterance <Speaker: {sentence_dict[conv_id][conv_turn]}> from <{speaker_label_text_set[dataset]}>:"
            speaker_target_dict[f"Speaker_{conv_id}_{conv_turn}"] = (
                f"Speaker_{speaker_label_dict[conv_id][conv_turn]}"
            )
            speaker_task_dict[f"Speaker_{conv_id}_{conv_turn}"] = temp_speaker_task_str

    if dataset == "iemocap":
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]

    new_train_id, new_test_id, new_valid_id = [], [], []
    # new_train_target, new_test_target, new_valid_target = [], [], []
    for train_id in train_ids:
        for conv_turn in range(len(sentence_dict[train_id])):
            new_train_id.append(f"{train_id}_{conv_turn}")

    for test_id in test_ids:
        for conv_turn in range(len(sentence_dict[test_id])):
            new_test_id.append(f"{test_id}_{conv_turn}")

    for valid_id in valid_ids:
        for conv_turn in range(len(sentence_dict[valid_id])):
            new_valid_id.append(f"{valid_id}_{conv_turn}")

    train_dataset = []
    test_dataset = []
    valid_dataset = []

    for train_id in new_train_id:
        train_dataset.append(
            {
                "instruction": content_task_dict[train_id],
                "input": "",
                "output": content_target_dict[train_id],
            }
        )

    for test_id in new_test_id:
        test_dataset.append(
            {
                "instruction": content_task_dict[test_id],
                "input": "",
                "output": content_target_dict[test_id],
            }
        )

    for valid_id in new_valid_id:
        valid_dataset.append(
            {
                "instruction": content_task_dict[valid_id],
                "input": "",
                "output": content_target_dict[valid_id],
            }
        )

    with open("train.json", "w") as f:
        json.dump(train_dataset, f, indent=4, ensure_ascii=False)

    with open("test.json", "w") as f:
        json.dump(test_dataset, f, indent=4, ensure_ascii=False)

    with open("valid.json", "w") as f:
        json.dump(valid_dataset, f, indent=4, ensure_ascii=False)


parser = argparse.ArgumentParser(description="Data processing script")
parser.add_argument(
    "--dataset", type=str, default="iemocap", help="Dataset name or path"
)
parser.add_argument(
    "--historical_window", type=int, default=32, help="Historical window size"
)
args = parser.parse_args()


# Process data
processed_data_path = process_dataset(
    dataset=args.dataset,
    window=args.historical_window,
)

print(processed_data_path)
