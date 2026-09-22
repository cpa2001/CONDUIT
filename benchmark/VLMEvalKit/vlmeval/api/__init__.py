from __future__ import annotations

from importlib import import_module


def _missing_api_class(class_name: str, module_name: str, error: Exception):
    class MissingAPI:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                f"Optional API wrapper {module_name}.{class_name} is unavailable in this "
                f"VLMEvalKit checkout: {type(error).__name__}: {error}"
            )

    MissingAPI.__name__ = class_name
    MissingAPI.__qualname__ = class_name
    return MissingAPI


def _load_optional(module_name: str, *class_names: str) -> None:
    try:
        module = import_module(f"{__name__}.{module_name}")
    except Exception as error:
        for class_name in class_names:
            globals()[class_name] = _missing_api_class(class_name, module_name, error)
        return
    for class_name in class_names:
        globals()[class_name] = getattr(module, class_name)


_load_optional("arm_thinker", "ARM_thinker")
_load_optional("bailingmm", "bailingMMAPI")
_load_optional("bedrock", "BedrockAPI")
_load_optional("bluelm_api", "BlueLM_API", "BlueLMWrapper")
_load_optional("claude", "Claude3V", "Claude_Wrapper")
_load_optional("cloudwalk", "CWWrapper")
_load_optional("doubao_vl_api", "DoubaoVL")
_load_optional("gcp_vertex", "GCPVertexAPI")
_load_optional("gemini", "Gemini", "GeminiWrapper")
_load_optional("glm_vision", "GLMVisionAPI")
_load_optional("gpt", "GPT4V", "OpenAIWrapper")
_load_optional("hf_chat_model", "HFChatModel")
_load_optional("hunyuan", "HunyuanVision")
_load_optional("jt_vl_chat", "JTVLChatAPI")
_load_optional("jt_vl_chat_mini", "JTVLChatAPI_2B", "JTVLChatAPI_Mini")
_load_optional("kimivl_api", "KimiVLAPI", "KimiVLAPIWrapper")
_load_optional("lmdeploy", "LMDeployAPI", "LMDeployWrapper")
_load_optional("minimax_api", "MiniMaxAPI")
_load_optional("mug_u", "MUGUAPI")
_load_optional("openai_sdk", "OpenAISDKWrapper")
_load_optional("qwen_api", "QwenAPI")
_load_optional("qwen_vl_api", "Qwen2VLAPI", "QwenVLAPI", "QwenVLWrapper")
_load_optional("rbdashmm_chat3_5_api", "RBdashMMChat3_5_38B_API", "RBdashMMChat3_78B_API")
_load_optional("rbdashmm_chat3_api", "RBdashChat3_5_API", "RBdashMMChat3_API")
_load_optional("reka", "Reka")
_load_optional("sensechat_vision", "SenseChatVisionAPI", "SenseChatVisionV2API")
_load_optional("siliconflow", "SiliconFlowAPI", "TeleMMAPI")
_load_optional("taichu", "TaichuVLAPI", "TaichuVLRAPI")
_load_optional("taiyi", "TaiyiAPI")
_load_optional("telemm", "TeleMM2_API")
_load_optional("telemm_thinking", "TeleMM2Thinking_API")
_load_optional("together", "TogetherAPI")
_load_optional("video_chat_online_v2", "VideoChatOnlineV2API")

__all__ = [
    'OpenAIWrapper', 'HFChatModel', 'GeminiWrapper', 'GPT4V', 'Gemini', 'QwenVLWrapper',
    'QwenVLAPI', 'QwenAPI', 'Claude3V', 'Claude_Wrapper', 'Reka', 'GLMVisionAPI', 'CWWrapper',
    'SenseChatVisionAPI', 'HunyuanVision', 'Qwen2VLAPI', 'BlueLMWrapper', 'BlueLM_API',
    'JTVLChatAPI', 'JTVLChatAPI_Mini', 'JTVLChatAPI_2B', 'bailingMMAPI', 'TaiyiAPI', 'TeleMMAPI',
    'SiliconFlowAPI', 'LMDeployAPI', 'ARM_thinker', 'OpenAISDKWrapper', 'LMDeployWrapper',
    'TaichuVLAPI', 'TaichuVLRAPI', 'DoubaoVL', 'MUGUAPI', 'KimiVLAPIWrapper', 'KimiVLAPI',
    'RBdashMMChat3_API', 'RBdashChat3_5_API', 'RBdashMMChat3_78B_API', 'RBdashMMChat3_5_38B_API',
    'VideoChatOnlineV2API', 'TeleMM2_API', 'TeleMM2Thinking_API', 'TogetherAPI', 'GCPVertexAPI',
    'BedrockAPI', 'SenseChatVisionV2API', 'MiniMaxAPI',
]
