"""
agent/vision_agent.py

视觉子 Agent —— 专职图片理解与分析。
主 Agent 通过工具调用本 Agent，本 Agent 使用独立的系统提示词和视觉模型完成任务。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from model.multimodal_factory import vision_model
from utils.image_handler import build_vision_message
from utils.logger_handler import logger


# ── 视觉子 Agent 系统提示词 ───────────────────────────────────────────────────

_VISION_SYSTEM_PROMPT = """你是一位专业的视觉分析助手，是客服团队中负责图片理解的专家。

你的职责：
1. 准确识别图片中的物体、场景、文字信息
2. 对二手商品图片进行专业鉴定（类别、品牌、新旧程度、潜在问题）
3. 客观、详细地回答关于图片内容的具体问题
4. 当图片内容与用户问题相关时，主动关联两者并给出有价值的洞察

输出要求：
- 语言清晰、简洁、客观
- 避免主观评价，以可观察的事实为准
- 若图片模糊或无法判断，如实说明局限性
""".strip()

_PRODUCT_PROMPT = """
你是一位专业的电商商品鉴定师，请对以下商品图片进行详细分析，按以下结构输出：

1. 【商品类别】判断商品类型
2. 【品牌与型号】如能识别则列出，否则标注"无法识别"
3. 【外观状态】描述新旧程度、磨损/划痕/污渍情况
4. 【关键特征】颜色、材质、配件、尺寸等可见信息
5. 【潜在问题】图片中可见的任何异常或缺陷
6. 【建议标签】3-5 个适合二手平台的描述标签

请保持客观，不做价值判断。
""".strip()


# ── VisionAgent ───────────────────────────────────────────────────────────────

class VisionAgent:
    """
    图片理解子 Agent。

    对外暴露两个主要方法：
      - invoke(image_input, question)  → 通用图片理解
      - analyze_product(image_input)   → 二手商品专项分析
    """

    def __init__(self):
        self.model = vision_model
        self.system_prompt = _VISION_SYSTEM_PROMPT
        logger.info("[VisionAgent] 视觉子 Agent 初始化完成")

    # ── 对外接口 ──────────────────────────────────────────────────────────────

    def invoke(
        self,
        image_input,
        question: str = "请详细描述这张图片的内容。",
    ) -> str:
        """
        通用图片理解。

        image_input：公网 URL / 本地路径 / bytes / Base64 字符串
        question：针对图片的问题或指令
        """
        logger.info(f"[VisionAgent] 收到分析请求，问题: {question[:60]}")
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=build_vision_message(image_input, question)),
            ]
            response = self.model.invoke(messages)
            result = _extract_text(response.content)
            logger.info(f"[VisionAgent] 分析完成，结果长度: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"[VisionAgent] 图片分析失败: {e}")
            return f"图片分析失败：{str(e)}"

    def analyze_product(self, image_input) -> str:
        """二手商品图片专项分析（闲鱼场景）"""
        logger.info("[VisionAgent] 商品专项分析")
        return self.invoke(image_input, _PRODUCT_PROMPT)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """统一提取模型返回的文字内容（兼容 str 和 list[dict]）"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)


# ── 单例 ──────────────────────────────────────────────────────────────────────

vision_agent = VisionAgent()