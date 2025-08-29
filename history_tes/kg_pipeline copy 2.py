from typing import  Dict, Any
from config import settings
import json, time, uuid
from steps._1_get_entities import get_entities
from steps._2_get_relations import get_quads
from steps._2a_get_relations import get_relations
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
import json
from steps._1_get_entities import get_entities
from steps._1b_entity_typing import get_entity_types
from steps._1c_entity_attributes import get_entity_attributes
from steps._1d_entity_aliases import get_entity_aliases
from steps._2_get_relations import get_relations as get_basic_relations
from steps._2a_get_relations import get_relations_with_sentence
from find_alias import find_entity_position


def run_kg(text: str, model) -> dict:
    # Step 1: 实体抽取与处理
    entities = get_entities(model, text)
    entity_types = get_entity_types(model, text, entities)
    entity_attrs = get_entity_attributes(model, text, entities)
    entity_aliases = get_entity_aliases(model, text, entities)
    entity_positions = find_entity_position(text, entities)

    entities_output = []
    for i, ent in enumerate(entities):
        word_c = ent
        word = entity_aliases.get(ent, ent)
        ent_type = entity_types.get(ent, "实体")
        position = entity_positions.get(ent, {})
        start, end = position.get("start", -1), position.get("end", -1)

        entities_output.append({
            "type": ent_type,
            "word": word,
            "word_c": word_c,
            "start": start,
            "end": end
        })

    # Step 2: 三元组抽取，带句子
    relations = get_relations_with_sentence(model, text, entities)

    envent_svos = [(s, p, o, sen) for s, p, o, sen in relations]

    relationships = []
    envent_entity_rel = []
    seen_sentences = set()

    for s, p, o, sen in relations:
        s_alias = entity_aliases.get(s, s)
        o_alias = entity_aliases.get(o, o)
        s_type = entity_types.get(s, "实体")
        o_type = entity_types.get(o, "实体")

        relationships.append([
            s_type, [s, s_alias], [p, p], o_type, [o, o_alias], sen
        ])

        if sen in seen_sentences:
            continue
        seen_sentences.add(sen)

        # 示例构造 envent_entity_rel
        envent_entity_rel.append([
            "事件",
            [sen, sen],
            [s, s],
            "国家/地区",  # 可按需替换为 entity_attrs.get(s, {}).get("location", "国家/地区")
            [o, o],
            sen
        ])

    # 最终打包结果
    result = {
        "string": text,
        "entities": entities_output,
        "envent_svos": envent_svos,
        "relationships": relationships,
        "envent_entity_rel": envent_entity_rel,
        "envent_detail": [],
        "res_re_infer": [],
        "triple_pair_new": []
    }

    return result
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
