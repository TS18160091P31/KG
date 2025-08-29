import dspy
from typing import List, Dict
from typing_extensions import TypedDict, Literal

# 定义实体类型（明确排除 '其他'）
EntityType = Literal["设备"]  # 不允许 "其他"
class Entity(TypedDict):
    name: str
    type: EntityType  # 类型检查时会排除 "其他"

class EntityAttributeExtraction(dspy.Signature):
    """
    目标：仅针对【具体航空母舰】抽取其
          ① 固有属性（基本不变） ② 最新官方状态（文本明确时间或官方表述）。
    若语句含“可能 / 或许 / 预计 / 计划于(无日期)”等，视为推测 → 保留。
    ──────────────────────────────
    ▍ 固有属性（举例）
      • 船级 / 批次            → “福特级首舰”
      • 排水量 / 尺寸          → “满载排水量 10 万吨”
      • 动力装置 / 技术        → “核动力 (A1B 反应堆 ×2)”
      • 服役 / 下水 / 交付时间 → “2017 年正式服役”
      • 母港 / 隶属舰队        → “常驻诺福克，隶属第 2 舰队”
      • 设计搭载量 / 配置     → “可搭载舰载机 75 架”

    ▍ 最新官方状态（举例 — 须满足「带日期 / 官方措辞」）
      • 编队组成        → “[最新 2025-06-24] 与多艘导弹驱逐舰及多个战斗机中队组成航母战斗群”
      • 升级项目        → “2024 年完成 EMALS 系统升级”
      • 最新人事        → “最新舰长：约翰·史密斯上校 (2025)”
      • 部署 / 维修状态 → “2025-06-24 启程驶离弗吉尼亚州，计划部署至地中海”
      • 最新载员        → “[最新 2025-06-24] 约 4500 名美军人员随舰启程”  

    ──────────────────────────────
    输出示例：
    [
      {
        "name": "福特号",
        "attributes": [
          "福特级首舰",
          "满载排水量 10 万吨",
          "核动力 (A1B 反应堆 ×2)",
          "2017 年正式服役",
          "常驻诺福克，隶属第 2 舰队",
          "[最新 2025-06-24] 与多艘导弹驱逐舰及多个战斗机中队组成航母战斗群",
          "[最新 2025-06-24] 约 4500 名美军人员随舰启程"
        ]
      }
    ]

    规则：
    • 当属性直接出处于原文且符合“固有”或“最新”定义时保留。
    • “最新”须包含日期或官方措辞（“最新”“截至××” 等）。
    • 输出列表中不出现历史性或易变的细节。
    """
    entity_list: List[Entity] = dspy.InputField(
        desc="仅包含类型为 '设备' 的具体航空母舰实体"
    )
    context: str = dspy.InputField()
    entity_attributes: list[dict] = dspy.OutputField(
        desc="{'name': 航母名, 'attributes': [固有属性或最新官方状态 ...]}"
    )


def extract_entity_attributes(dspy: dspy.dspy, entity_list: List[str], context: str) -> List[Dict]:
    extractor = dspy.Predict(EntityAttributeExtraction)
    result = extractor(entity_list=entity_list, context=context)
    return result.entity_attributes
