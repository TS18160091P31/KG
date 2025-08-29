#QA_part/_2_predict.py
from typing import List
import dspy
import json
#QA_part/_2_predict.py 中新增内容
class ExtractStructuredAnswer(dspy.Signature):
    """
    你是一个严谨的信息提取专家，只允许基于提供的上下文文本进行回答，不得自行想象或编造内容。
    当问题存在多个可能答案（如多个地点、时间、人物等）时，请对每个可能性分别输出完整结构，每个答案为一个 JSON 对象，包含以下字段：
    - "conclusion"：简洁准确地回答用户问题，仅基于原文内容，不得合并多个结论，每条记录只包含一个结论。
    - "evidence"：用原文中的一句或多句话逐字原样引用作为支撑依据，一个符号都不可以修改，直接给出完整一句原文。
    - "reasoning"：说明你是如何从原文中得出该结论的，逻辑清晰，不能扩展原文以外的内容。解释可以分多条：
    格式如下：
    ["解释1","解释二"....]
    - "confidence"：包括推断可信度与信息来源可靠性，每项包括等级与理由，格式如下：
      {
        "推断可信度": {
          "等级": "高",
          "理由": "原文表述直接明确，结论无需多层推理，几乎无歧义。"
        },
        "信息来源可靠性": {
          "等级": "中",
          "理由": "该语句来自网友评论，具有一定主观性，缺乏官方背书。"
        }
      }
    回答结果必须为 JSON 数组；若原文未提及相关信息，请不要生成任何结论项，而是整体返回一个空数组。
    """
    question: str = dspy.InputField(desc="用户提出的问题")
    context: str = dspy.InputField(desc="上下文原始文本")
    json_response: str = dspy.OutputField(desc="严格结构化 JSON 数组格式回答")


def extract_structured_answer(dspy: dspy.dspy, question: str, context: str):
    """
    用于提问并从上下文中抽取多个结构化结论。
    返回格式为 JSON 数组（或固定失败字符串）。
    """
    extractor = dspy.Predict(ExtractStructuredAnswer)
    result = extractor(question=question, context=context)

    if isinstance(result.json_response, str) and "原文未提及" in result.json_response:
        return result.json_response

    try:
        return json.loads(result.json_response) if isinstance(result.json_response, str) else result.json_response
    except json.JSONDecodeError:
        return {
            "error": "模型返回内容非 JSON 格式，请检查模型输出是否规范。",
            "raw": result.json_response
        }
