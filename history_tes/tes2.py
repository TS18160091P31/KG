import json
from steps._2a_get_relations import get_relations
import dspy
import time

# 配置 DeepSeek 模型
lm = dspy.LM(
    "deepseek/deepseek-chat",
    api_key="sk-5621f03006ff4d4d87ac6adc98b6f136",
    api_base="https://api.deepseek.com"
)
dspy.configure(lm=lm)

# 读取实体列表
with open("outputs/entities.json", "r", encoding="utf-8") as f:
    entity_list = json.load(f)

# 读取原始文本作为上下文（推荐你保存 text.json 或 hardcode）
with open("source_text.txt", "r", encoding="utf-8") as f:
    context = f.read()

# Step 2：关系抽取
print("\nStep 2: 开始抽取三元组关系...")
start = time.time()
try:
    relations = get_relations(dspy=dspy, input_data=context, entities=entity_list, is_conversation=False)
    print(f"关系抽取完成，共 {len(relations)} 条关系，用时 {time.time() - start:.2f}s")
    if not relations:
        print("未抽取到任何关系，请检查实体或模型返回格式")
    else:
        for r in relations:
            print("  ", r)

    # 保存三元组关系
    with open("outputs/relations.json", "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
except Exception as e:
    print("关系抽取出错:", e)
    exit(1)

