from typing import  Dict, Any
from config import settings
import json, time, uuid
from steps._1_get_entities import get_entities
from steps._2_get_relations import get_quads
from steps._1b_entity_typing import classify_entities
from steps._1c_entity_attributes import extract_entity_attributes
import dspy, ollama  # pip install ollama
from QA_part._2_predict import extract_structured_answer

# -------- 初始化 LM (一次即可) -------- #
# _ollama = ollama.Client(settings.OLLAMA_BASE)
# lm = dspy.Ollama(
#     model=settings.OLLAMA_MODEL,
#     client=_ollama,
#     temperature=0.2
# )
# lm = dspy.LM(
#     "deepseek/deepseek-chat",
#     api_key="sk-5621f03006ff4d4d87ac6adc98b6f136",
#     api_base="https://api.deepseek.com"
# )
# lm = dspy.LM(
#         "ollama_chat/qwen:14b",
#         api_key="",
#         api_base="http://localhost:11434"
#     )
# dspy.configure(lm=lm)
# -------- 业务主函数 -------- #
def run_kg(text: str) -> Dict[str, str]:
    """
    返回生成的两个文件路径：
      - entity_attr_path  (entities/123.json)
      - relations_path    (relations/123.json)
    """
    lm = dspy.LM(
        model=settings.OLLAMA_MODEL,
        api_key="",
        api_base=settings.OLLAMA_BASE
    )
    dspy.configure(lm=lm)
    # uid = uuid.uuid4().hex[:8]          # 简短唯一标识
    # t0 = time.time()

    # ---------- Step 1: 实体 ---------- #
    entities = get_entities(dspy=dspy, input_data=text, is_conversation=False)
    typed = classify_entities(dspy=dspy, entity_list=entities, context=text)
    attrs = extract_entity_attributes(dspy=dspy, entity_list=typed, context=text)
    print(entities)
    # ---------- Step 2: 关系 ---------- #
    quads = get_quads(dspy=dspy, input_data=text, entities=entities, is_conversation=False)
    print(quads)
    # # ---------- 保存 ---------- #
    # ent_path = settings.ENT_DIR / f"{uid}.json"
    # rel_path = settings.REL_DIR / f"{uid}.json"

    # ent_path.write_text(json.dumps(attrs, ensure_ascii=False, indent=2), encoding="utf-8")
    # rel_path.write_text(json.dumps(quads, ensure_ascii=False, indent=2), encoding="utf-8")

    # return {
    #     "entity_attr_file": str(ent_path),
    #     "relations_file": str(rel_path),
    #     "entities_extracted": len(attrs),
    #     "quads_extracted": len(quads),
    #     "elapsed_sec": round(time.time() - t0, 2)
    # }


    # 直接返回 JSON 内容而不是保存到文件
    return {
        "entities": entities,
        "entity_attributes": attrs,
        "relations": quads
    }

def run_structured_qa(context: str, question: str) -> Dict[str, Any]:
    """
    输入原始文本和问题，返回结构化问答结果：
    - answers: [["结论", "原文", ["解释1", "解释2"]], ...]
    - confidence: [{"推断可信度": {...}, "信息来源可靠性": {...}}, ...]
    - note: 如果无法作答，提供统一说明
    """
    # 使用 ollama 的本地 LLM
    lm = dspy.LM(
        model=settings.OLLAMA_MODEL,
        api_key="",
        api_base=settings.OLLAMA_BASE
    )
    dspy.configure(lm=lm)
    print("连接成功")
    # 提取结构化答案
    result = extract_structured_answer(dspy, question=question, context=context)

    # 可用于日志或调试
    print("结构化问答结果：", result)

    return result
