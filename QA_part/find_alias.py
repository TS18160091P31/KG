import json
import re
from typing import Dict, List

def find_alias_positions(text: str, alias_dict: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
    result = {}
    for main_name, alias_list in alias_dict.items():
        matches = []
        for alias in alias_list:
            for match in re.finditer(re.escape(alias), text):
                matches.append({
                    "alias": alias,
                    "start": match.start(),
                    "end": match.end()
                })
        result[main_name] = matches
    return result

# 示例用法
if __name__ == "__main__":
    # 读取别名字典
    with open("outputs/entity_aliases.json", "r", encoding="utf-8") as f:
        alias_dict = json.load(f)

    # 读取原文
    with open("source_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 获取所有别名的位置
    position_dict = find_alias_positions(text, alias_dict)

    # 输出结果
    with open("outputs/entity_alias_positions.json", "w", encoding="utf-8") as f:
        json.dump(position_dict, f, ensure_ascii=False, indent=2)

    print("别名位置提取完毕，保存到 outputs/entity_alias_positions.json")
