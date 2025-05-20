import argparse
import pickle


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
    data = pickle.load(
        open(
            "pkl",
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
        for conv_turn in range(len(sentence_dict[conv_id])):
            temp_content_str = (
                "Now you are expert of sentiment and emotional analysis. "
            )
            temp_content_str += "The following conversation noted between '### ###' involves several speakers. ### "

            index_w = max(conv_turn - window, 0)
            for speaker_label, sub_sent in zip(
                speaker_label_dict[conv_id][index_w : conv_turn + 1],
                sentence_dict[conv_id][index_w : conv_turn + 1],
            ):
                temp_content_str += f'\t Speaker_{speaker_label}:"{sub_sent}"'
                # temp_id_task_str += (f'\t Speaker_?:"{sub_sent}"')

            content_target_dict[f"{conv_id}_{conv_turn}"] = label_set[dataset][
                data[1][conv_id][conv_turn]
            ]
            target_utterance = temp_content_str.split("\t")[-1]
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


parser = argparse.ArgumentParser(description="Data processing script")
parser.add_argument(
    "--dataset", type=str, default="iemocap", help="Dataset name or path"
)
parser.add_argument(
    "--historical_window", type=int, default=12, help="Historical window size"
)
args = parser.parse_args()


# Process data
processed_data_path = process_dataset(
    dataset=args.dataset,
    window=args.historical_window,
)

print(processed_data_path)
