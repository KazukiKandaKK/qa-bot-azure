import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add repository root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_qa_chatbot import QAChatbot, KnowledgeBaseManager


class TestQAChatbotNoContext(unittest.TestCase):
    """QAChatbot.answer のコンテキストなし時の挙動をテスト"""

    def setUp(self):
        self.mock_kb = Mock(spec=KnowledgeBaseManager)
        self.mock_kb.search.return_value = []

        patcher_tok = patch('rag_qa_chatbot.AutoTokenizer')
        patcher_model = patch('rag_qa_chatbot.AutoModelForCausalLM')
        patcher_pipeline = patch('rag_qa_chatbot.pipeline')
        self.addCleanup(patcher_tok.stop)
        self.addCleanup(patcher_model.stop)
        self.addCleanup(patcher_pipeline.stop)
        patcher_tok.start()
        patcher_model.start()
        patcher_pipeline.start()

        self.chatbot = QAChatbot(self.mock_kb)
        self.chatbot.generator = None  # _simple_answer を強制

    def test_answer_without_context(self):
        question = "会社の歴史は？"
        answer = self.chatbot.answer(question)
        self.mock_kb.search.assert_called_once_with(question, top_k=3)
        self.assertEqual(answer, "申し訳ありませんが、関連する情報が見つかりませんでした。")


if __name__ == '__main__':
    unittest.main()
