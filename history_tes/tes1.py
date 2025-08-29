from steps._1_get_entities import get_entities
from steps._2_get_relations import get_quads
from steps._1b_entity_typing import classify_entities
from steps._1d_entity_aliases import extract_aliases_llm
from steps._1c_entity_attributes import extract_entity_attributes
import dspy
import time
import json
import os
from QA_part._2_predict import extract_structured_answer
# # 创建输出目录
# os.makedirs("outputs", exist_ok=True)

# 初始化 DeepSeek 模型
print("正在配置 DeepSeek LLM...")
try:
    # lm = dspy.LM(
    #         "ollama_chat/deepseek-r1:14b",
    #         api_key="",
    #         api_base="http://localhost:11434"
    #     )
    lm = dspy.LM(
        "deepseek/deepseek-chat",
        api_key="sk-5621f03006ff4d4d87ac6adc98b6f136",
        api_base="https://api.deepseek.com"
    )
    dspy.configure(lm=lm)
    print("ollama 模型配置成功")
except Exception as e:
    print("ollama 配置失败:", e)
    exit(1)
if __name__ == "__main__":
    sample_text = """
A轮船作为应邀参展船艇，将于航海日当天停靠琼海，开展开放日、科普讲解与绿色航海倡议发布。
为来年春季运营做准备。
五月中旬将有“我们一起去看海”宣传活动在宁波启动，六月中旬将赴九江展开江河湖海看中国现场推介会([交通运输部][1])。
2025年7月15日——物资供应保障任务实战模仿“天舟九号”快速补给模式，A轮船首次执行类似应急补给任务，从上海火速运送6吨高价值医疗物资到海南，展示三小时内“出港→停靠→卸货”紧急应变能力([维基百科][11])。
2025年11月——冬季内部培训与公众开放11月首月，A轮船在上海港区斜靠，开展全年最后一轮“船员培训+公众开放日”活动，包括模拟应急撤离、安全演练、引擎室探秘等主题。
**第一篇 | 航行与发展：一艘船的一年轮回**2025年1月——新年首航每年年初，A轮船（假设船名）从上海吴淞口国际港出发，开启年度首航，开启“新年巡航”系列活动。
同时发布次年运营计划，开启下一轮周期。
2025年12月——年度总结大会A轮船于本月回港，举办“年度航行总结大会”，披露全年绿色航运、公益影响与技术革新成果，向政府部门与合作方汇报。
2025年5月——航海日活动每年7月11日为中国航海日，今年主题“绿色航海，向新图强”，但相关宣传与论坛活动自5月起即开始预热。
**最新报道概要*** **内河与航海双驱动**：2025年“我们一起去看海”线上线下推广活动，线下5月启动于宁波，6月走进九江，7月航海日在琼海主办论坛([交通运输部][1])。
2025年3月——春季海试春分前后，A轮船启动春季例行海试，从上海出发，前往东海及黄海海域，进行动力系统、导航系统及船员应急安全演练。
2025年9月——秋季货物运输季进入三季度末，A轮船转向货物流业务，从上海经由宁波、广州往返东南亚，执行“一带一路”绿色航运补给任务，展示新能源推进技术在国际航线上的应用。
旨在检测设备稳定性，为全年安全运行奠定基础。
2025年12月——智能船舶发展论坛年底，A轮船主办“智能与绿色船舶发展”小型论坛，赴宁波或上海召开，邀请研究机构、监管机构及航运企业代表出席，展示年度成果：自主导航、网络防护、绿色减排、快速补给。
2025年10月——外援安全协助A轮船接收重庆大学等高校邀请，赴东南亚一港开展航运安全与网络监管友好交流。
2025年4月——网络安全专项演练春末，A轮船进行“海上网络安全”专项部署，模拟GPS欺骗、船舶系统遭遇勒索软件等实战性攻击，完成新型应对机制的演练。
* **崔贤级驱逐舰动态**：朝鲜“崔贤号”驱逐舰于4月25日下水，并于5月21日下水事故后进入修复，6月前恢复浮吊状态([维基百科][13])。
2025年8月——环保性能评估巡航进行一次为期两周的“碳排放测试巡航”，配合环保机构评估船舶新型混合动力减排效果。
* **爱达·魔都号运营回暖**：国产大型游轮“爱达·魔都号”自年初起恢复东北亚航线，具备5G覆盖、水上乐园等设施，并持续发布邮轮绿色航线计划([维基百科][14])。
结合近期红海安全事件，分享安全防护经验([维基百科][6])。
    """

    question = "里根号10月在哪里"

    result = extract_structured_answer(dspy=dspy, question=question, context=sample_text)

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))

# 示例文本
# text = """
# 据美联社2025年06月25日报道，美国海军唯一现役的福特级航空母舰“福特”号于24日启程驶离弗吉尼亚州基地。报道称，此时正值中东局势不稳之际，这艘美国最先进的航母可能将被部署在以色列附近。
#  """
# with open("source_text.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# # Step 1：实体抽取
# print("\nStep 1: 开始抽取实体...")
# start = time.time()
# try:
#     entities = get_entities(dspy=dspy, input_data=text, is_conversation=False)
#     print(f"实体抽取完成，共 {len(entities)} 个实体，用时 {time.time() - start:.2f}s")
#     print("实体列表：", entities)

#     # 保存实体列表
#     with open("outputs/entities.json", "w", encoding="utf-8") as f:
#         json.dump(entities, f, ensure_ascii=False, indent=2)
# except Exception as e:
#     print("实体抽取出错:", e)
#     exit(1)
# # entities=['美联社', '美国', '海军', '福特级航空母舰', '福特号', '弗吉尼亚州', '基地', '中东', '以色列']
# # Step 1.1：实体分类
# try:    
#     # 执行实体分类
#     typed_entities = classify_entities(dspy=dspy, entity_list=entities, context=text)
#     # print(typed_entities)
#     # 输出结果
#     print("实体类型识别结果：")
#     for item in typed_entities:
#         print(f"  - {item['name']} → {item['type']}")
#     # 保存到新文件
#     with open("outputs/typed_entities.json", "w", encoding="utf-8") as f:
#         json.dump(typed_entities, f, ensure_ascii=False, indent=2)
# except Exception as e:
#     print("实体分类出错:", e)
#     exit(1)

# # Step 1.1.1：实体别名
# try:    
#     # 1. 提取别名
#     aliases_all = {}
#     for ent in typed_entities:
#         name = ent["name"]
#         aliases = extract_aliases_llm(dspy, entity_name=name, context=text)
#         aliases_all[name] = aliases
#     # 2. 输出结果
#     print("实体别名识别结果：")
#     for name, alias_list in aliases_all.items():
#         print(f"  - {name} 的别名：{alias_list}")
#     # 3. 保存别名到新文件
#     with open("outputs/entity_aliases.json", "w", encoding="utf-8") as f:
#         json.dump(aliases_all, f, ensure_ascii=False, indent=2)
# except Exception as e:
#     print("实体分类出错:", e)
#     exit(1)
# # Step 1.2：实体属性    
# try:
#     # 调用属性抽取模块

#     attributes = extract_entity_attributes(dspy=dspy, entity_list=typed_entities, context=text)

#     # 输出结果
#     print("实体属性识别结果：")
#     for item in attributes:
#         print(f"- {item['name']}：{item['attributes']}")

#     # 保存
#     with open("outputs/entity_attributes.json", "w", encoding="utf-8") as f:
#         json.dump(attributes, f, ensure_ascii=False, indent=2)
#     # 仅保存所有 name 值到 entity.json

#     names = [item["name"] for item in attributes]  # 提取所有 name
#     with open("outputs/entity.json", "w", encoding="utf-8") as f:
#         json.dump(names, f, ensure_ascii=False, indent=2)
# except Exception as e:
#     print("关系属性分类:", e)
#     exit(1)

# # ---------- Step 2：关系抽取（四元组） ---------- #
# print("\nStep 2: 开始抽取四元组关系...")
# start = time.time()

# try:
#     # 核心调用：将 get_relations → get_quads
#     quads = get_quads(
#         dspy=dspy,
#         input_data=text,
#         entities=entities,
#         is_conversation=False          # 普通文本场景
#     )

#     elapsed = time.time() - start
#     print(f"关系抽取完成，共 {len(quads)} 条四元组，用时 {elapsed:.2f}s")
#     print(dspy.inspect_history(n=1))
#     if not quads:
#         print("未抽取到任何关系，请检查实体或模型返回格式")
#     else:
#         for q in quads:
#             # q = (time, subject, location, predicate)
#             print("  ", q)

#     # 保存四元组到文件
#     with open("outputs/quads.json", "w", encoding="utf-8") as f:
#         json.dump(quads, f, ensure_ascii=False, indent=2)

# except Exception as e:
#     print("关系抽取出错:", e)
#     exit(1)


