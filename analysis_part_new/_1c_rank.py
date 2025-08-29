# _1c_rank.py  ——  仿照 1b 风格的重写版
from typing import List, Dict, Any
import re, json
import dspy
from dspy import Signature, Predict


class BranchRelevanceRanker(Signature):
    """
    你是“问题相关性评分专家”。请对一组候选分支问题相对于总问题的相关性打分，并给出极简理由。
    - 评分：3=决定性；2=关键性；1=辅助性；0=无关
    - 要求：只输出 JSON，字段结构见下。
    - 输出字段 ranked：按分值降序的列表，每项形如：
        {"sub_question":"...", "score": 0|1|2|3, "rationale":"简要中文理由"}

    评分提示（可酌情参考）：
    - 若只是主问题的同义改写且未引入可检验关键变量，通常为 1
    - 禁止输出多余文本（如客套、解释性段落等）
    """

    main_question: str = dspy.InputField(desc="总问题")
    sub_questions: List[str] = dspy.InputField(desc="候选分支问题列表")
    ranked: List[Dict[str, Any]] = dspy.OutputField(desc="[{sub_question, score, rationale}]（按分值降序）")


def generate_rank(dspy: dspy.dspy, main_question: str, sub_questions: str) -> List[str]:
    """
    输入：简要事件描述 + 用户提问
    输出：适合向量检索的多条细化信息需求（可输入RAG召回模块）
    """
    generator = dspy.Predict(BranchRelevanceRanker)
    result = generator(main_question=main_question, sub_questions=sub_questions)
    return result.ranked
if __name__ == "__main__":
    lm = dspy.LM(
            "ollama_chat/deepseek-r1:14b",
            api_key="",
            api_base="http://localhost:11434"
        )
    dspy.configure(lm=lm)
    mq = "这次地区冲突中，是否会出现外部大国直接军事介入？"
    cands = [
        "是否已有大国在边境集结地面部队？",
        "社交媒体讨论热度是否上升？",
        "周边国家是否临时关闭空域？",
    ]
    result = generate_rank(dspy.dspy,mq, cands)
    print(json.dumps(result, ensure_ascii=False, indent=2))
