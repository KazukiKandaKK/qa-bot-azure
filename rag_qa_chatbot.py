import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List, Tuple


class KnowledgeBaseManager:
    """ナレッジベースの構築と検索を担当するクラス"""
    
    def __init__(self, embedding_model_name: str = "intfloat/multilingual-e5-small"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.chunk_size = 500
        self.overlap = 50
    
    def _split_document(self, document: str) -> List[str]:
        """ドキュメントを意味のあるチャンクに分割"""
        sentences = re.split(r'[。！？\n]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_index(self, documents: List[str]):
        """インデックス構築"""
        print("ドキュメントを分割中...")
        all_chunks = []
        for doc in documents:
            doc_chunks = self._split_document(doc)
            all_chunks.extend(doc_chunks)
        
        self.chunks = all_chunks
        print(f"総チャンク数: {len(self.chunks)}")
        
        print("テキストをベクトル化中...")
        embeddings = self.embedding_model.encode(self.chunks, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype('float32')
        
        print("FAISSインデックスを構築中...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print("ナレッジベースの構築が完了しました！")
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """質問に関連するチャンクを検索"""
        if self.index is None or not self.chunks:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx])
        
        return relevant_chunks


class QAChatbot:
    """ユーザーとの対話と回答生成のメインロジックを担うクラス"""
    
    def __init__(self, knowledge_base: KnowledgeBaseManager):
        self.knowledge_base = knowledge_base
        print("言語モデルを初期化中...")
        
        model_name = "microsoft/DialoGPT-medium"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"LLMの初期化に失敗しました: {e}")
            print("シンプルなテンプレートベースの回答システムを使用します。")
            self.generator = None
    
    def _build_prompt(self, context: List[str], question: str) -> str:
        """プロンプトを構築"""
        context_text = "\n".join(context) if context else "関連する情報が見つかりませんでした。"
        
        prompt = f"""あなたは、株式会社○○の社内情報を的確に教える、親切なAIアシスタントです。

以下の社内情報を参考にして、質問に答えてください。
---
{context_text}
---

質問: {question}

参考情報に基づいて、質問に日本語で回答してください。もし参考情報に答えがない場合は、推測で答えず、「申し訳ありませんが、関連する情報が見つかりませんでした。」と回答してください。

回答:"""
        
        return prompt
    
    def _simple_answer(self, context: List[str], question: str) -> str:
        """シンプルなテンプレートベースの回答生成"""
        if not context:
            return "申し訳ありませんが、関連する情報が見つかりませんでした。"
        
        combined_context = " ".join(context)
        
        if "経費" in question and "上限" in question:
            if "上限" in combined_context or "限度" in combined_context:
                return f"参考情報によると、{combined_context[:200]}..."
        
        if "有給" in question:
            if "有給" in combined_context or "休暇" in combined_context:
                return f"有給休暇について、参考情報では：{combined_context[:200]}..."
        
        return f"ご質問に関する情報として、以下が見つかりました：{combined_context[:300]}..."
    
    def answer(self, question: str) -> str:
        """質問に対する回答を生成"""
        print(f"質問を処理中: {question}")
        
        relevant_context = self.knowledge_base.search(question, top_k=3)
        print(f"関連する情報を{len(relevant_context)}件見つけました。")
        
        if self.generator is None:
            return self._simple_answer(relevant_context, question)

        try:
            prompt = self._build_prompt(relevant_context, question)
            encoded = self.tokenizer(prompt, return_tensors="pt")
            prompt_tokens = encoded.input_ids.shape[1]
            max_tokens = 512
            max_new_tokens = max(1, max_tokens - prompt_tokens)

            response = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7
            )
            
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            if not answer or len(answer) < 10:
                return self._simple_answer(relevant_context, question)
            
            return answer
            
        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {e}")
            return self._simple_answer(relevant_context, question)


if __name__ == "__main__":
    print("RAG型社内QAチャットボットを起動中...")
    
    dummy_documents = [
        """
        経費精算ルール
        
        1. 経費精算の上限について
        交通費: 月額50,000円まで
        接待費: 1回あたり10,000円まで、月額30,000円まで
        書籍・研修費: 年額100,000円まで
        
        2. 申請方法
        経費精算システムにログインし、レシートの写真を添付して申請してください。
        申請期限は費用発生から1ヶ月以内です。
        
        3. 承認フロー
        5,000円未満: 直属上司の承認のみ
        5,000円以上: 直属上司 + 部長の承認が必要
        """,
        
        """
        有給休暇申請ガイド
        
        1. 有給休暇の取得について
        入社から6ヶ月経過後に10日間付与されます。
        以降、1年ごとに付与日数が増加します。
        有給休暇の有効期限は2年間です。
        
        2. 申請方法
        有給休暇申請システムから申請してください。
        緊急時を除き、取得希望日の3営業日前までに申請が必要です。
        
        3. 連続取得について
        5日以上の連続取得の場合は、2週間前までに申請してください。
        業務の引き継ぎ資料の作成も忘れずに行ってください。
        """,
        
        """
        社内システム利用ガイド
        
        1. ログイン情報
        社員番号とパスワードでログインしてください。
        パスワードは90日ごとに変更が必要です。
        
        2. 利用可能システム
        - 勤怠管理システム
        - 経費精算システム
        - 有給休暇申請システム
        - 社内掲示板
        
        3. トラブル時の対応
        システムに問題が発生した場合は、IT部門（内線：1234）までご連絡ください。
        """
    ]
    
    try:
        kb_manager = KnowledgeBaseManager()
        kb_manager.build_index(dummy_documents)
        
        chatbot = QAChatbot(kb_manager)
        
        test_questions = [
            "経費精算の上限はいくらですか？",
            "有給休暇はいつから取得できますか？",
            "システムにログインできない場合はどうすればいいですか？",
            "会社の創立年はいつですか？"
        ]
        
        print("\n" + "="*50)
        print("QAチャットボットのデモンストレーション")
        print("="*50)
        
        for question in test_questions:
            print(f"\n質問: {question}")
            print("-" * 30)
            answer = chatbot.answer(question)
            print(f"回答: {answer}")
            print()
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("必要なライブラリがインストールされていない可能性があります。")
        print("以下のコマンドでインストールしてください：")
        print("pip install torch transformers sentence-transformers faiss-cpu numpy")