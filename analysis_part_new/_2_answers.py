import typing 
import dspy

class EventAnswerGenerator(dspy.Signature):
    """
    你是时政/军事分析专家，擅长从多个角度深入分析具体事件。请根据已知的问题相关内容，结合已有的背景信息与常识性知识，提供专业、客观且严谨的分析回答。

    要求：
    - 答案应围绕事件的实际情况进行分析，避免任何推测或无根据的陈述。
    - 每个回答需展现深入的分析与专业知识，确保涵盖事件的多个相关方面。
    - 提供的回答应有清晰的结构，简洁明了，但不失深度，确保能够有效回应提问。
    - 需结合历史、地理、文化等普遍背景，确保回答具有逻辑性和专业性。
    - 在回答最后进行一个整体置信度自评（高 / 中 / 低）,不解释
    【置信度判定规则（只输出最终等级）】
    - 高：材料为权威与多源可交叉验证（≥2个独立来源）；时间、地点、主体明确；数据/文件可查；叙述无明显推测词。
    - 中：材料为权威/半权威或“权威+二手/评论”混合；关键信息部分明确；存在有限推测或样本有限但可被印证。
    - 低：材料主要源自社媒/匿名/未证实/传闻/明显推测；关键事实缺乏来源；时间与主体含糊或存在矛盾。

    """
    event_article: str = dspy.InputField(desc="问题的相关内容")
    question: str = dspy.InputField(desc="分析问题")
    answer: str = dspy.OutputField(desc="问题的详细回答")
    confidence:str = dspy.OutputField(desc="对于回答置信度的自评，仅允许输出高，中，低三档")

def generate_analysis_answer(dspy, event_article: str, question: str) -> str:
    """
    输入：事件文章与分析维度
    输出：回答
    """
    generator = dspy.Predict(EventAnswerGenerator)
    result = generator(event_article=event_article, question=question)
    return {"answer":result.answer, "confidence":result.confidence}
if __name__ == "__main__":
    # === 强烈建议用环境变量管理密钥 ===
    # import os
    # api_key = os.getenv("DEEPSEEK_API_KEY")

    lm = dspy.LM(
            "ollama_chat/deepseek-r1:14b",
            api_key="",
            api_base="http://localhost:11434"
        )
    dspy.configure(lm=lm)

    # ====== 测试样例集：每条包含材料与问题 ======
    cases = [
        {
            "title": "黑海粮食走廊中止后的影响评估",
            "event_article": (
                "背景：自2022年起，黑海粮食倡议允许乌克兰经黑海出口粮食，缓解全球粮价压力。"
                "2023年后，倡议多次受阻并最终中止，黑海航运风险上升，保险费率上涨。"
                "联合国粮农组织（FAO）与国际谷物理事会发布的官方统计数据显示，"
                "部分粮食转经多瑙河及陆路，出口受限导致北非与中东部分国家粮价敏感度显著上升。"
            ),
            "question": "在倡议中止与海上风险上升的情况下，短中期全球粮食供应与价格的主要传导链条是什么？"
        },
        {
            "title": "红海—亚丁湾航运风险与供应链重路由",
            "event_article": (
                "背景：红海与亚丁湾一带近期出现针对商船的袭击与拦截事件。"
                "多家社交媒体账号与匿名船员爆料称，一些国际班轮公司被迫绕行好望角以规避风险。"
                "绕航据称增加了航程、船期与燃油成本，同时引发集装箱周转紧张与运价波动。"
            ),
            "question": "主要行业（如汽车、化工、零售）的供应链受影响机制分别体现在哪些环节？企业可行的缓释策略有哪些？"
        },
        {
            "title": "南海争端中的风险升级与管控",
            "event_article": (
                "背景：南海多方存在海域主张与执法行动分歧。近年来个别海域出现水炮驱离、补给受阻、近距离机动等高风险互动。"
                "中国、菲律宾等国官方声明均强调航行自由与国际法框架，但部分智库与媒体推测，"
                "未来可能出现更多渔业执法冲突与联合演训。"
            ),
            "question": "在避免误判与局势外溢的前提下，各方最可能采取的风险管控与沟通机制有哪些？"
        },
    ]

    print("\n===== EventAnswerGenerator 集成测试（直接调用 LLM）=====\n")

    for i, case in enumerate(cases, start=1):
        print(f"[Case {i}] {case['title']}")
        print("问题（question）:", case["question"])
        # 调用你的封装函数，只返回 answer 文本
        answer_text = generate_analysis_answer(
            dspy=dspy,
            event_article=case["event_article"],
            question=case["question"]
        )
        print("答案（answer）:\n", answer_text)
        print("-" * 80)

    print("\n全部测试结束。\n")

