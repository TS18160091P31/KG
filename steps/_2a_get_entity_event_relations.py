
import re
from typing import List, Tuple, Dict

def extract_context(text: str, s: str, p: str, o: str) -> str:
    """
    从原文中提取包含三元组 s, p, o 的上下文短句。
    若找不到完整三元组，则使用 s+p+o 构造短句。
    """
    pattern = rf"[^。！？]*{re.escape(s)}[^。！？]*{re.escape(p)}[^。！？]*{re.escape(o)}[^。！？]*[。！？]"
    match = re.search(pattern, text)
    if match:
        return match.group().strip()
    # 若未找到完全包含 s,p,o 的句子，则尝试放宽限制
    pattern_loose = rf"[^。！？]*({re.escape(s)}|{re.escape(o)})[^。！？]*{re.escape(p)}[^。！？]*[。！？]"
    match_loose = re.search(pattern_loose, text)
    if match_loose:
        return match_loose.group().strip()
    return f"{s}{p}{o}"

def get_entity_event_relations(input_text: str, spo_list: List[Tuple[str, str, str]]) -> List[List[str]]:
    """
    基于 SPO 三元组构建 envent_entity_rel 格式输出。
    每项结构为：
    [
        事件谓语,
        [事件原文, 事件原文copy],
        [主语/宾语实体, 其copy],
        国家/地区,
        [另一个实体, 其copy],
        事件原文copy
    ]
    """
    results = []
    for s, p, o in spo_list:
        context = extract_context(input_text, s, p, o)
        results.append([
            p,
            [context, context + "copy"],
            [s, s + "copy"],
            "未知国家",  # 若需国家识别，可后续扩展
            [o, o + "copy"],
            context + "copy"
        ])
    return results
