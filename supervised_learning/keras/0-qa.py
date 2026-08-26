#!/usr/bin/env python3
"""
Module to perform Question Answering using TensorFlow Hub and Transformers.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Parameters:
    - question: string containing the question to answer
    - reference: string containing the reference document

    Returns:
    - string containing the answer, or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize input question and reference text
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # Prepare input tokens and segment IDs for BERT ([CLS] Q [SEP] R [SEP])
    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)

    # 0 for question tokens + [CLS] + [SEP], 1 for reference tokens + [SEP]
    input_type_ids = ([0] * (len(question_tokens) + 2)) + ([1] * (len(reference_tokens) + 1))

    # Convert inputs to tensors with batch dimension
    input_word_ids, input_mask, input_type_ids = map(
        lambda x: tf.expand_dims(tf.constant(x, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    # Run inputs through model
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # Extract start and end logits for answer span
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    # Return None if start index comes after end index or invalid range
    if short_start > short_end:
        return None

    # Reconstruct answer string from tokens
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer or answer.strip() == "":
        return None

    return answer
