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
from find_alias import find_alias_positions

import os

def save_json(obj, filename):
    os.makedirs("cache", exist_ok=True)
    with open(os.path.join("cache", filename), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

print("配置\n")
# 配置 DeepSeek 模型
lm = dspy.LM(
    "deepseek/deepseek-chat",
    api_key="sk-5621f03006ff4d4d87ac6adc98b6f136",
    api_base="https://api.deepseek.com"
)
dspy.configure(lm=lm)
print("完成\n")
# 读取原始文本作为上下文（推荐你保存 text.json 或 hardcode）
with open("source_text.txt", "r", encoding="utf-8") as f:
    text = f.read()
try:
    entities = get_entities(dspy=dspy, input_data=text, is_conversation=False)
    save_json(entities, "entities.json")
except Exception as e:
    print("get_entities出错:", e)
    exit(1)
try:
    entity_types = classify_entities(dspy=dspy, entity_list=entities, context=text)
    save_json(entity_types, "entity_types.json")
except Exception as e:
    print("classify_entities出错:", e)
    exit(1)
try:
    entity_attrs = extract_entity_attributes(dspy=dspy, entity_list=entity_types, context=text)
    save_json(entity_attrs, "entity_attrs.json")
except Exception as e:
    print("extract_entity_attributes出错:", e)
    exit(1)

try:
    entity_aliases = {}
    for ent in entity_types:
        name = ent["name"]
        aliases = extract_aliases_llm(dspy, entity_name=name, context=text)
        entity_aliases[name] = aliases
    # 2. 输出结果
    print("实体别名识别结果：")
    for name, alias_list in entity_aliases.items():
        print(f"  - {name} 的别名：{alias_list}")
    # 3. 保存别名到新文件
    with open("outputs/entity_aliases.json", "w", encoding="utf-8") as f:
        json.dump(entity_aliases, f, ensure_ascii=False, indent=2)
    save_json(entity_aliases, "entity_aliases.json")
except Exception as e:
    print("extract_aliases_llm出错:", e)
    exit(1)
    # Step 1: 实体抽取与属性处理
try:
    entity_positions_raw = find_alias_positions(text, entity_aliases)
    save_json(entity_positions_raw, "entity_positions_raw.json")
except Exception as e:
    print("find_alias_positions出错:", e)
    exit(1)
try:
    quads = get_quads(dspy=dspy, input_data=text, entities=entities, is_conversation=False)
    save_json(quads, "quads.json")
except Exception as e:
    print("get_quads出错:", e)
    exit(1)

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
print("entities_output\n",entities_output)
    # Step 2: 关系抽取
try:
    relations = get_relations(dspy=dspy, input_data=text, entities=entities, is_conversation=False)
    save_json(relations, "relations.json")
except Exception as e:
    print("关系抽取出错:", e)
    exit(1)
envent_svos = [(s, p, o) for s, p, o, sen in relations]
print("envent_svos\n",envent_svos)
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
print("envent_entity_rel\n",envent_entity_rel)
print("relationships\n",relationships)
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
print("result\n",result)
