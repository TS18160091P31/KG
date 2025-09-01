from typing import  Dict, Any
from config import settings
import json, time, uuid
from steps._1_get_entities import get_entities
from steps._2_get_relations import get_quads
from steps._2a_get_relations import get_relations
from steps._1b_entity_typing import classify_entities
from steps._1c_entity_attributes import extract_entity_attributes
from steps._1d_entity_aliases import extract_aliases_llm
import dspy, ollama  # pip install ollama
from QA_part._2_predict import extract_structured_answer
from QA_part.find_alias import find_alias_positions


# -------- 初始化 LM (一次即可) -------- #
# _ollama = ollama.Client(settings.OLLAMA_BASE)
# lm = dspy.Ollama(
#     model=settings.OLLAMA_MODEL,
#     client=_ollama,
#     temperature=0.2
# )
# lm = dspy.LM(
#         "ollama_chat/qwen:14b",
#         api_key="",
#         api_base="http://localhost:11434"
#     )
# dspy.configure(lm=lm)
# -------- 业务主函数 -------- #




def run_kg(text: str) -> dict:
    # 设置模型接口
    lm = dspy.LM(
        model=settings.OLLAMA_MODEL,
        api_key=settings.API_KEY,
        api_base=settings.OLLAMA_BASE
    )
    # dspy.configure(lm=lm)
    dspy.context(lm=lm)

    # Step 1: 实体抽取与属性处理
    entities = get_entities(dspy=dspy, input_data=text, is_conversation=False)
    entity_types = classify_entities(dspy=dspy, entity_list=entities, context=text)
    entity_attrs = extract_entity_attributes(dspy=dspy, entity_list=entity_types, context=text)
    entity_aliases = {}
    for ent in entity_types:
        name = ent["name"]
        aliases = extract_aliases_llm(dspy, entity_name=name, context=text)
        entity_aliases[name] = aliases
    entity_positions_raw = find_alias_positions(text, entity_aliases)
    quads = get_quads(dspy=dspy, input_data=text, entities=entities, is_conversation=False)
    entity_types_dict = {ent["name"]: ent["type"] for ent in entity_types}

    # 格式化实体输出：实体结构应包含 word、word_c、type、start、end
    entities_output = []
    for ent in entity_types:
        word = ent["name"]
        word_c = word
        # word = entity_aliases.get(ent, ent)  # 别名，默认用自身
        ent_type = ent["type"]
        
        # 取该实体的第一个别名出现位置
        position_list = entity_positions_raw.get(ent["name"], [])
        if position_list:
            first_pos = position_list[0]
            start, end = first_pos.get("start", -1), first_pos.get("end", -1)
        else:
            start, end = -1, -1
        entities_output.append({
            "type": ent_type,
            "word": word,
            "word_c": word_c,
            "start": start,
            "end": end
        })

    # Step 2: 关系抽取
    relations = get_relations(dspy=dspy, input_data=text, entities=entities, is_conversation=False)
    envent_svos = [(s, p, o) for s, p, o, sen in relations]

    # Step 3: relationships 和 envent_entity_rel 构造
    relationships = []
    envent_entity_rel = []
    seen_sentences = set()

    for s, p, o, sen in relations:
        # s_alias = entity_aliases.get(s, s)
        # o_alias = entity_aliases.get(o, o)
        s_type = entity_types_dict.get(s, "实体")
        o_type = entity_types_dict.get(o, "实体")

        envent_entity_rel.append([
            "事件",
            [sen, sen],
            [o_type, o_type],  # 关系词原文与副本
            o_type,
            [o, o],
            sen
        ])
        envent_entity_rel.append([
            "事件",
            [sen, sen],
            [s_type, s_type],  # 关系词原文与副本
            s_type,
            [s, s],
            sen
        ])
        relationships.append([
            s_type,
            [s, s],
            [p, p],  # 关系词原文与副本
            o_type,
            [o, o],
            sen
        ])
        # if sen not in seen_sentences:
        #     seen_sentences.add(sen)
        #     envent_entity_rel.append([
        #         "事件",
        #         [sen, sen],
        #         [s, s],
        #         "国家/地区",  # 如需动态获取类型，可改为 entity_types.get(s, "国家/地区")
        #         [o, o],
        #         sen
        #     ])

    # Step 4: 返回统一结构
    result = {
        "string": text,
        "entities": entities_output,
        "envent_detail": quads,       # 可选：时间、地点、主体结构
        "envent_entity_rel": envent_entity_rel,
        "relationships": relationships,
        "envent_svos": envent_svos,
        "res_re_infer": [],        # 可选：关系/知识推理输出
        "triple_pair_new": []      # 可选：增强版三元组结构
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
        api_key=settings.API_KEY,
        api_base=settings.OLLAMA_BASE
    )
    dspy.configure(lm=lm)
    print("连接成功")
    # 提取结构化答案
    result = extract_structured_answer(dspy, question=question, context=context)

    # 可用于日志或调试
    print("结构化问答结果：", result)

    return result
