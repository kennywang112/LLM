import os
from llama_cpp import Llama
import json
from typing import List, Dict
import re
from tenacity import retry, stop_after_attempt, wait_exponential
import gc

class DialogueAnalyzer:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        self.model_settings = {
            "n_ctx": 4096,           # Mistral 支援更長的上下文
            "n_batch": 512,
            "n_threads": 8,
            "low_vram": True,
            "seed": 42,
            "n_gpu_layers": 0
        }

        try:
            self.llm = Llama(model_path=model_path, **self.model_settings)
            print("模型載入成功！")
        except Exception as e:
            raise Exception(f"模型載入失敗: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_dialogue(self, text: str) -> List[Dict]:
        # Mistral 特定的提示詞格式
        prompt = f"""<s>[INST] Please carefully analyze the following text and extract the speaker and content for each dialogue. Pay special attention to:
        1. Look for speaker indicators like "said", "asked", "replied", etc.
        2. Look for character names or roles before or after the dialogue
        3. Consider the context to identify who is speaking

        Text:
        {text}

        Requirements:
        - Do not use "Unknown" as speaker unless absolutely necessary
        - Include any context that helps identify the speaker
        - Preserve the exact dialogue content
        - Make sure the output is valid JSON

        Output format:
        [
            {{"speaker": "Character Name/Role", "content": "Exact dialogue content"}},
            {{"speaker": "Character Name/Role", "content": "Exact dialogue content"}}
        ]

        Only output the JSON array, no additional text. [/INST]"""

        
        try:
            response = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["</s>", "[/INST]"],
                echo=False
            )
            
            result_text = response['choices'][0]['text'].strip()
            print(f"\n模型原始輸出:\n{result_text}")  # 調試用
            
            # 嘗試提取JSON部分
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group()
            
            try:
                return json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"\nJSON解析失敗，使用備選方案...")
                # 如果模型輸出解析失敗，使用手動提取的對話
                dialogues = self.extract_dialogues(text)
                formatted_json = self.format_dialogue_json(dialogues)
                return json.loads(formatted_json)
                
        except Exception as e:
            print(f"分析過程發生錯誤: {str(e)}")
            # 使用備選方案
            dialogues = self.extract_dialogues(text)
            formatted_json = self.format_dialogue_json(dialogues)
            try:
                return json.loads(formatted_json)
            except:
                return [{"error": f"分析錯誤: {str(e)}"}]
        finally:
            gc.collect()

    def format_dialogue_json(self, dialogues: List[str]) -> str:
        """手動格式化對話為JSON字符串"""
        json_items = []
        
        for dialogue in dialogues:
            # 嘗試識別說話者
            if "said" in dialogue:
                parts = dialogue.split("said")
                if len(parts) == 2:
                    speaker = parts[0].strip().strip('"').strip()
                    content = parts[1].strip().strip('"').strip()
                    if speaker and content:
                        json_items.append({
                            "speaker": speaker,
                            "content": content.strip('.')
                        })
                    continue
            
            # 如果無法識別說話者，使用默認值
            json_items.append({
                "speaker": "Unknown",
                "content": dialogue.strip('"').strip()
            })

        return json.dumps(json_items, ensure_ascii=False, indent=2)

    def extract_dialogues(self, text: str) -> List[Dict]:
        dialogue_patterns = [
        # Pattern 1: "Speech," speaker said/asked
        r'"([^"]+),"?\s*(?:said|asked|replied|answered)\s+([^\.!?\n]+)',
        
        # Pattern 2: speaker said/asked, "Speech"
        r'([^\.!?\n]+?)(?:said|asked|replied|answered)[^"]*"([^"]+)"',
        
        # Pattern 3: "Speech" (with context before or after)
        r'"([^"]+)"\s*(?:\(([^)]+)\)|(?:[-—]+\s*([^\.!?\n]+)))?',
        
        # Pattern 4: 'Speech' (single quotes)
        r'\'([^\']+)\'\s*(?:\(([^)]+)\)|(?:[-—]+\s*([^\.!?\n]+)))?'
    ]
    
        dialogues = []
        
        for pattern in dialogue_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                if len(match.groups()) >= 2:
                    content = match.group(1).strip()
                    speaker = None
                    
                    # Try to find speaker from different groups
                    for group in match.groups()[1:]:
                        if group and group.strip():
                            speaker = group.strip()
                            break
                    
                    if content:
                        dialogues.append({
                            "speaker": speaker if speaker else "Unknown",
                            "content": content
                        })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dialogues = []
        for d in dialogues:
            dialogue_key = (d["speaker"], d["content"])
            if dialogue_key not in seen:
                seen.add(dialogue_key)
                unique_dialogues.append(d)
        
        return unique_dialogues


def save_results(results: List[Dict], filename: str = "analysis_results.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"結果已儲存到: {filename}")
    except Exception as e:
        print(f"儲存結果時發生錯誤: {str(e)}")

def main():
    MODEL_PATH = r"D:\mistral-7b-instruct-v0.1.Q4_0.gguf"  # Mistral 模型路徑
    
    try:
        analyzer = DialogueAnalyzer(MODEL_PATH)
        
        test_text = """
        "How could I go?" said Trembling. "I have no clothes good enough to
        wear at church; and if my sisters were to see me there, they'd kill me
        for going out of the house."
        
        "I'll give you clothes," said the henwife.
        """
        
        print("開始分析文本...")
        
        dialogues = analyzer.extract_dialogues(test_text)
        print(f"\n找到 {len(dialogues)} 段對話:")
        for d in dialogues:
            print(f"- {d}")
        
        results = analyzer.analyze_dialogue(test_text)
        
        save_results(results)
        
        print("\n分析結果:")
        print(json.dumps(results, ensure_ascii=False, indent=2))
            
    except Exception as e:
        print(f"程式執行過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()