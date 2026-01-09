import json


def load_processor_and_model(video_or_audio="video"):
    if video_or_audio == "video":
        model_name = QWEN2VL_PATH
        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif video_or_audio == "audio":
        model_name = QWEN2_AUDIO_PATH
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(model_name)

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    return processor, model


def audio_tone_inference(model, processor, messages):
    # "text": "Be concise and analyze the tone of the speaker in the audio."
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    audios = []
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            ele["audio_url"],
                            sr=processor.feature_extractor.sampling_rate,
                        )[0]
                    )

    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )
    inputs = inputs.to("cuda")

    generate_ids = model.generate(**inputs, max_length=128, top_p=0.1, temperature=0.1)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def vision_inference(model, processor, messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


def qwen2VL_inference(train_or_test="train"):
    # 获取 VIDEO_PATH 下的所有视频文件
    file = TRAIN_FILE if train_or_test == "train" else TEST_FILE
    read_data = get_data(file)
    processor, model = load_processor_and_model()

    results = []
    for data in tqdm(read_data):
        conversation = data["conversation"]
        conversation_list = []
        for utt in conversation:
            video_file = VIDEO_PATH / utt["video_name"]

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_file.as_posix(),
                            "max_pixels": 320 * 180,
                            "fps": 2.0,
                        },
                        {
                            "type": "text",
                            # Background
                            # "text": "Be concise and provide a brief description of the background of the video. (Note: provide an answer within 50 words or less)",
                            # character action
                            "text": "Describe the actions of the person currently speaking in the video in detail, and provide a brief description of the actions of others and limit to 50 words or less.",
                            # Speaker's Facial Expressions
                            # "text": "Describe the facial expressions of each person in the current video, with a focus on the facial expression of the person currently speaking and limit to 50 words or less.",
                        },
                    ],
                },
            ]
            output = vision_inference(model, processor, messages)
            conversation_list.append(
                {"utterance_ID": utt["utterance_ID"], "action": output}
            )
        results.append(
            {
                "conversation_ID": data["conversation_ID"],
                "conversation": conversation_list,
            }
        )
    with open(f"action_{train_or_test}.json", "w") as f:
        json.dump(results, f, indent=4)


def audio_inference(train_or_test="train"):
    file = TRAIN_FILE if train_or_test == "train" else TEST_FILE
    read_data = get_data(file)
    processor, model = load_processor_and_model("audio")
    results = []

    for data in tqdm(read_data):
        conversation_ID = data["conversation_ID"]
        conversation = data["conversation"]
        conversation_list = []
        for utt in conversation:
            video_name = utt["video_name"]
            utt_id = utt["utterance_ID"]
            audio_name = video_name.split(".")[0] + ".wav"
            audio_file = AUDIO_PATH / audio_name
            messages = [
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio_url": audio_file.as_posix(),
                        },
                        {
                            "type": "text",
                            "text": "Be concise and analyze the tone of the speaker in the audio.",
                        },
                    ],
                },
            ]
            response = audio_tone_inference(model, processor, messages)

            conversation_list.append({"utterance_ID": utt_id, "tone": response})
        results.append(
            {
                "conversation_ID": conversation_ID,
                "conversation": conversation_list,
            }
        )
    with open(f"audio_tone_{train_or_test}.json", "w") as f:
        json.dump(results, f, indent=4)
