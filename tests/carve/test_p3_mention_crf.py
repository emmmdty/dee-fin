import unittest

import torch

from carve.p3_mention_crf import MentionCRF, build_bio_labels
from carve.text_segmentation import Sentence
from carve.tokenization import tokenize_with_offsets, token_span_to_char_span


class P3MentionCrfTests(unittest.TestCase):
    def test_bio_labels_single_role(self) -> None:
        sentence = Sentence(text="甲公司质押100万股。", char_start=0, char_end=10)
        tokens = tokenize_with_offsets(sentence.text, base_offset=sentence.char_start)

        labels = build_bio_labels(sentence, tokens, [("质押", "质押方", "甲公司")])

        self.assertEqual(labels[:3], [1, 2, 2])
        self.assertTrue(all(label == 0 for label in labels[3:]))

    def test_bio_labels_multi_role_priority(self) -> None:
        sentence = Sentence(text="甲公司质押100万股。", char_start=0, char_end=10)
        tokens = tokenize_with_offsets(sentence.text, base_offset=sentence.char_start)

        labels = build_bio_labels(
            sentence,
            tokens,
            [("质押", "第一角色", "甲公司"), ("质押", "第二角色", "甲公司质押")],
        )

        self.assertEqual(labels[:3], [1, 2, 2])
        self.assertEqual(labels[3], 0)

    def test_bio_labels_normalize_aligned(self) -> None:
        sentence = Sentence(text="甲公司质押１００万股。", char_start=5, char_end=16)
        tokens = tokenize_with_offsets(sentence.text, base_offset=sentence.char_start)

        labels = build_bio_labels(sentence, tokens, [("质押", "质押股票/股份数量", "100万股")])

        self.assertEqual(labels[5:10], [1, 2, 2, 2, 2])

    def test_crf_viterbi_recovers_simple_pattern(self) -> None:
        crf = MentionCRF(hidden_size=3, num_tags=3)
        with torch.no_grad():
            crf.emission.weight.copy_(torch.eye(3))
            crf.emission.bias.zero_()
            crf.transitions.fill_(-5.0)
            crf.transitions[0, 0] = 0.0
            crf.transitions[0, 1] = 0.0
            crf.transitions[1, 2] = 0.0
            crf.transitions[2, 2] = 0.0
            crf.transitions[2, 0] = 0.0
        token_repr = torch.tensor([[[0.0, 6.0, 0.0], [0.0, 0.0, 6.0], [6.0, 0.0, 0.0]]])
        mask = torch.tensor([[True, True, True]])

        spans = crf.decode(token_repr, mask)

        self.assertEqual(spans, [[(0, 2)]])

    def test_token_span_to_char_span_roundtrip(self) -> None:
        text = "甲公司质押100万股。"
        tokens = tokenize_with_offsets(text)

        start, end = token_span_to_char_span(tokens, 0, 3)

        self.assertEqual(text[start:end], "甲公司")


if __name__ == "__main__":
    unittest.main()
