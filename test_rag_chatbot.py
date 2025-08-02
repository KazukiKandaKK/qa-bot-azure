import unittest
import sys
import os
from unittest.mock import Mock, patch
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_qa_chatbot import KnowledgeBaseManager, QAChatbot


class TestKnowledgeBaseManager(unittest.TestCase):
    """KnowledgeBaseManagerクラスのテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        # モックのembedding modelを使用してテストを高速化
        with patch('rag_qa_chatbot.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(3, 384).astype('float32')
            mock_transformer.return_value = mock_model
            
            self.kb_manager = KnowledgeBaseManager()
            self.test_documents = [
                "これは経費精算に関するドキュメントです。上限は月額50,000円です。",
                "有給休暇は入社から6ヶ月後に付与されます。年間10日間取得可能です。",
                "システムの利用方法について説明します。ログインには社員番号が必要です。"
            ]
    
    def test_document_splitting(self):
        """ドキュメント分割機能のテスト"""
        document = "これは最初の文です。これは二番目の文です。これは三番目の文です。"
        chunks = self.kb_manager._split_document(document)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)
    
    @patch('rag_qa_chatbot.faiss')
    def test_build_index(self, mock_faiss):
        """インデックス構築のテスト"""
        # FAISSのモック設定
        mock_index = Mock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        self.kb_manager.build_index(self.test_documents)
        
        # チャンクが作成されていることを確認
        self.assertGreater(len(self.kb_manager.chunks), 0)
        
        # FAISSインデックスが作成されていることを確認
        self.assertIsNotNone(self.kb_manager.index)
        mock_faiss.IndexFlatIP.assert_called_once()
    
    def test_search_empty_index(self):
        """空のインデックスでの検索テスト"""
        result = self.kb_manager.search("テスト質問")
        self.assertEqual(result, [])
    
    @patch('rag_qa_chatbot.faiss')
    def test_search_with_results(self, mock_faiss):
        """検索結果があるケースのテスト"""
        # インデックスを構築
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        self.kb_manager.build_index(self.test_documents)
        self.kb_manager.index = mock_index
        
        results = self.kb_manager.search("経費精算", top_k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)


class TestQAChatbot(unittest.TestCase):
    """QAChatbotクラスのテスト"""
    
    def setUp(self):
        """テスト前の準備"""
        # モックのKnowledgeBaseManagerを作成
        self.mock_kb = Mock(spec=KnowledgeBaseManager)
        
        # LLMの初期化をスキップしてシンプルな回答システムを使用
        with patch('rag_qa_chatbot.AutoTokenizer'), \
             patch('rag_qa_chatbot.AutoModelForCausalLM'), \
             patch('rag_qa_chatbot.pipeline'):
            self.chatbot = QAChatbot(self.mock_kb)
            # LLMを無効にしてテンプレートベース回答を強制
            self.chatbot.generator = None
    
    def test_build_prompt(self):
        """プロンプト構築のテスト"""
        context = ["経費精算の上限は50,000円です。", "申請は月末までに行ってください。"]
        question = "経費精算の上限はいくらですか？"
        
        prompt = self.chatbot._build_prompt(context, question)
        
        self.assertIn("株式会社○○", prompt)
        self.assertIn(question, prompt)
        self.assertIn(context[0], prompt)
        self.assertIn(context[1], prompt)
    
    def test_simple_answer_with_context(self):
        """コンテキストありのシンプル回答テスト"""
        context = ["経費精算の上限は月額50,000円です。"]
        question = "経費精算の上限はいくらですか？"
        
        answer = self.chatbot._simple_answer(context, question)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertNotEqual(answer, "申し訳ありませんが、関連する情報が見つかりませんでした。")
    
    def test_simple_answer_no_context(self):
        """コンテキストなしのシンプル回答テスト"""
        context = []
        question = "存在しない情報について教えてください"
        
        answer = self.chatbot._simple_answer(context, question)
        
        self.assertEqual(answer, "申し訳ありませんが、関連する情報が見つかりませんでした。")
    
    def test_answer_integration(self):
        """統合的な回答生成テスト"""
        # モックの検索結果を設定
        self.mock_kb.search.return_value = ["経費精算の上限は月額50,000円です。"]
        
        question = "経費精算の上限はいくらですか？"
        answer = self.chatbot.answer(question)
        
        # KnowledgeBaseManagerの検索が呼ばれたことを確認
        self.mock_kb.search.assert_called_once_with(question, top_k=3)
        
        # 回答が生成されたことを確認
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


class TestIntegration(unittest.TestCase):
    """統合テスト"""
    
    @patch('rag_qa_chatbot.SentenceTransformer')
    @patch('rag_qa_chatbot.faiss')
    def test_full_pipeline(self, mock_faiss, mock_transformer):
        """完全なパイプラインのテスト"""
        # モック設定
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(5, 384).astype('float32')
        mock_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        # システム初期化
        documents = [
            "経費精算の上限は月額50,000円です。申請は期限内に行ってください。",
            "有給休暇は入社6ヶ月後から取得可能です。年間10日間付与されます。"
        ]
        
        kb_manager = KnowledgeBaseManager()
        kb_manager.build_index(documents)
        
        chatbot = QAChatbot(kb_manager)
        chatbot.generator = None  # テンプレートベース回答を使用
        
        # 質問と回答
        question = "経費精算について教えてください"
        answer = chatbot.answer(question)
        
        # 結果検証
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


def run_performance_test():
    """パフォーマンステスト"""
    print("\n" + "="*50)
    print("パフォーマンステスト実行中...")
    print("="*50)
    
    import time
    
    # 大量のドキュメントでテスト
    large_documents = [
        f"これは{i}番目のテストドキュメントです。" * 10 
        for i in range(100)
    ]
    
    with patch('rag_qa_chatbot.SentenceTransformer') as mock_transformer, \
         patch('rag_qa_chatbot.faiss') as mock_faiss:
        
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(500, 384).astype('float32')
        mock_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        kb_manager = KnowledgeBaseManager()
        
        # インデックス構築時間測定
        start_time = time.time()
        kb_manager.build_index(large_documents)
        build_time = time.time() - start_time
        
        print(f"インデックス構築時間: {build_time:.2f}秒")
        
        # 検索時間測定
        start_time = time.time()
        for _ in range(10):
            kb_manager.search("テスト質問")
        search_time = (time.time() - start_time) / 10
        
        print(f"平均検索時間: {search_time:.4f}秒")


if __name__ == "__main__":
    print("RAG QAチャットボットのテストを開始します...")
    
    # 単体テスト実行
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # パフォーマンステスト実行
    run_performance_test()
    
    print("\n" + "="*50)
    print("すべてのテストが完了しました！")
    print("="*50)