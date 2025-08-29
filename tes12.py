import dspy
import time
import json
import os
from analysis_part_new._1_question import generate_analysis_questions
from analysis_part_new._1b_question import generate_detailed_questions
from analysis_part_new._2_answers import generate_analysis_answer
# # 创建输出目录
# os.makedirs("outputs", exist_ok=True)

# 初始化 DeepSeek 模型
print("正在配置 DeepSeek LLM...")
try:
    lm = dspy.LM(
            "ollama_chat/deepseek-r1:14b",
            api_key="",
            api_base="http://localhost:11434"
        )
    dspy.configure(lm=lm)
    print("ollama 模型配置成功")
except Exception as e:
    print("ollama 配置失败:", e)
    exit(1)
if __name__ == "__main__":
    # with open("text.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    # text="佩罗西访问台湾"  
    text="2025年全球气候峰会"
    context="""
NBC新聞在7月30日的報導中提及，佩洛西訪亞行程啟程時，台灣部分還只列為「暫定」。[36]8月1日，美國有線電視新聞網引述台灣和美國政府官員的話報導稱，佩洛西預計在台灣停留一晚[37]。美國政府對佩洛西訪臺的意見有分歧。7月下旬，美國總統拜登在回答有關佩洛西當時傳聞可能會訪問台灣的問題時表示「軍方認為現在這不是一個好主意。在佩洛西可能面對來自北京阻礙的問題上，美國國務院發言人普萊斯對此拒絕討論任何擔憂[39]。佩洛西在一場例行記者會上被問及拜登的說法時，拒絕討論她可能前往台灣的行程，但表明美國對台灣的支援很重要。佩洛西對拜登「軍方認為現在這不是一個好主意」說法表示理解，猜測軍方擔心美國飛機會被擊落或類似的事情。但她補充說，她沒有從拜登那裏聽說過[40]。行政院院長蘇貞昌於7月27日表示非常感謝佩洛西長年以來對臺灣非常支援與友善，對於任何國外友好貴賓訪臺都非常歡迎，政府都會做最好的安排[41]。包括美國國家安全顧問蘇利文在內的多名美國政府官員，皆對佩洛西的出訪表達了擔憂，並就如何勸阻議長不要前往台灣提出建議，甚至派一些官員直接與佩洛西交談。拜登於2022年7月28日與中共中央總書記暨中華人民共和國最高領導人習近平進行了兩個多小時的通話。習近平呼籲拜登勸阻佩洛西訪台。[42]拜登稱對於佩洛西訪臺一事，他無法阻止另一個美國公民的人身自由，總統也不應該干涉國會的運作。因立法院院長游錫堃COVID-19快篩陽性仍需隔離[79]，由立法院副院長蔡其昌接見。蔡其昌表示佩洛西是「台灣的朋友」。佩洛西表示這是很大的讚美，並代表國會的同事接受這項殊榮。[80]佩洛西在立法院致詞時，提到過去曾在天安門拉橫幅抗議，表達對人權的支援，及對中國大陸不公平貿易作法表達抗議的經歷[81][82]。朝野各黨團在立法院的總召集人代表其所屬政黨對佩洛西的來訪皆表達肯定及歡迎[83][84]，而正在隔離照護中的院長游錫堃亦透過視像和佩洛西致意[85]。在佩洛西與蔡其昌的對談中，為佩洛西翻譯的口譯員數度打斷佩洛西。結束時，佩洛西再度談話但被另一名口譯員以時間因素制止。佩洛西於韓國標準時3日夜間21時26分抵達京畿道烏山空軍基地[112]。後續
佩洛西離台後，眾議院議長網站聲明，此次訪問是她們在印太地區更廣泛旅行的一環，重點在關注安全、繁榮和治理，台灣在這三方面都是全球領導者。此時此刻，美國與台灣人民團結一致比以往任何時刻都更加重要，「中國（大陸）無法阻止世界各地的領袖造訪台灣」[113]，佩洛西本人也發出1分半左右的推特影片，以紀錄此次「歷史性」的行程[114]。8月10日，佩洛西返美後召開首場記者會表示，她的訪台行程符合美國政策，北京卻藉此企圖建立新常態，因此不能放任不管，隨團成員也在記者會指改變現狀的是北京而非華盛頓[115][116]。中華民國外交部澄清稱，佩洛西受訪後其辦公室發布逐字稿，說明她本意指的是台灣，稱「中國」屬口誤[117]。[註 4]

據霍士新聞網報道，佩洛西的兒子小保羅隨團訪問台灣，但他並非官方代表團成員，亦不是華府官員或議員顧問，卻乘搭公帑支付的專機，接受中華民國政府款待。小保羅受僱於兩家鋰礦公司，而台灣是亞洲鋰電池生產領先者，因此惹來外界質疑他利用母親影響力謀取商業利益。8月2日，佩洛西抵達台灣後，柯比重申，稱佩洛西此行完全符合美方以台灣關係法、美中三公報與六項保證為指引的一中政策，美方反對任何一方片面改變現狀，期望以和平方式解決兩岸分歧[142]。佩洛西在抵達台灣後也發表聲明，認為此次訪台是「符合台灣關係法、中美三個聯合公報和『六項保證』的」[143]。相關事件及影響
更多資訊：2022年蘭茜·佩洛西訪問台灣的反應與影響 § 相關事件
網絡攻擊
8月2日，中華民國總統府網站受到網絡攻擊，一度未能正常運作[144][145][146]。"""
    question="各国在气候峰会上的立场差异主要源于哪些国内政治因素？"
    dimension="客观原因"
    # answer=generate_analysis_answer(dspy=dspy,event_article=text,question=question)
    # print(answer)
    # # num=5
    results = generate_analysis_questions(dspy=dspy, dimension=dimension, event_article=text)
    print(results)
    # import json
    # with open("qusetions.json", "w", encoding="utf-8") as f:
    #     json.dump(results,f,ensure_ascii=False, indent=2)
    # print(json.dumps(results, ensure_ascii=False, indent=2))
    # with open('qusetions.json', 'r', encoding='utf-8') as file:
    #     results = json.load(file)
    # question_answer={}
    # for dimension,question_list  in results.items():
    #     if dimension == "主观原因":
    #         print(f"维度: {dimension}")
    #         for question in question_list:
    #             print(f"  问题: {question}")
    #             detail_q=generate_detailed_questions(dspy=dspy,event_title=text,question=question)
    #             # print(f"  回答: {answer}")
    #             question_answer[question] = detail_q[question]
    # print(question_answer)
    # with open("detailed_answer.json", "w", encoding="utf-8") as f:
    #     json.dump(question_answer,f,ensure_ascii=False, indent=2)

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


