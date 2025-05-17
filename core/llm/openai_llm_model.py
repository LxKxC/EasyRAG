import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Generator
import requests
from dotenv import load_dotenv
import json

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAICompatibleLLM:
    """支持OpenAI格式API的第三方LLM模型的包装类"""
    
    def __init__(self):
        """
        初始化OpenAI兼容的LLM模型
        
        从环境变量中加载配置:
        - OPENAI_API_KEY: API密钥
        - OPENAI_API_BASE: API基础URL (例如: http://localhost:8000/v1)
        - OPENAI_API_MODEL: 模型名称 (例如: gpt-3.5-turbo)
        """
        # 从环境变量获取配置
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model_name = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
        
        # 移除末尾的斜杠
        if self.api_base.endswith("/"):
            self.api_base = self.api_base[:-1]
            
        # 补全API路径，确保以/v1结尾
        if not self.api_base.endswith("/v1"):
            if "/v1/" not in self.api_base:
                self.api_base = f"{self.api_base}/v1"
                
        logger.info(f"初始化OpenAI兼容LLM模型: 基础URL={self.api_base}, 模型={self.model_name}")
    
    def _prepare_headers(self) -> Dict[str, str]:
        """准备请求头"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _build_messages(self, query: str, context: List[str] = None, history: List[List[str]] = None) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        
        # 添加系统消息
        system_content = "你是一个专业的助手，你可以根据提供的上下文信息来回答用户的问题。回答应该准确、有帮助且基于事实。"
        if context and len(context) > 0:
            system_content += "\n\n参考信息：\n" + "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(context)])
        
        messages.append({"role": "system", "content": system_content})
        
        # 添加历史消息
        if history and len(history) > 0:
            for user_query, assistant_response in history:
                messages.append({"role": "user", "content": user_query})
                messages.append({"role": "assistant", "content": assistant_response})
        
        # 添加当前查询
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def generate_response(self, 
                         query: str, 
                         context: List[str] = None, 
                         history: List[List[str]] = None, 
                         temperature: float = 0.1,
                         max_length: int = 2048) -> str:
        """
        生成回复
        
        参数:
            query: 用户查询
            context: 检索到的上下文列表
            history: 聊天历史 [user, assistant, user, assistant, ...]
            temperature: 温度参数，控制回答的随机性
            max_length: 生成的最大长度
            
        返回:
            生成的回答
        """
        try:
            url = f"{self.api_base}/chat/completions"
            headers = self._prepare_headers()
            
            # 构建消息
            messages = self._build_messages(query, context, history)
            
            # 构建请求数据
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_length,
                "stream": False
            }
            
            # 发送请求
            logger.debug(f"发送请求到 {url}")
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            if response.status_code != 200:
                logger.error(f"API请求失败: {response.status_code}, {response.text}")
                return f"抱歉，API请求失败: {response.status_code}"
                
            response_data = response.json()
            
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                logger.error(f"无效的API响应: {response_data}")
                return "抱歉，收到了无效的API响应"
                
            # 提取回复内容
            reply = response_data["choices"][0]["message"]["content"].strip()
            return reply
            
        except Exception as e:
            logger.exception(f"生成回答时出错: {str(e)}")
            return f"抱歉，生成回答时出错: {str(e)}"
    
    def generate_stream(self, 
                       query: str, 
                       context: List[str] = None, 
                       history: List[List[str]] = None, 
                       temperature: float = 0.1,
                       max_length: int = 2048) -> Generator[str, None, None]:
        """
        流式生成回复
        
        参数:
            query: 用户查询
            context: 检索到的上下文列表
            history: 聊天历史 [user, assistant, user, assistant, ...]
            temperature: 温度参数，控制回答的随机性
            max_length: 生成的最大长度
            
        返回:
            生成的回答流
        """
        try:
            url = f"{self.api_base}/chat/completions"
            headers = self._prepare_headers()
            
            # 构建消息
            messages = self._build_messages(query, context, history)
            
            # 构建请求数据
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_length,
                "stream": True
            }
            
            # 发送请求
            logger.debug(f"发送流式请求到 {url}")
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=120)
            
            if response.status_code != 200:
                logger.error(f"API流式请求失败: {response.status_code}, {response.text}")
                yield f"抱歉，API请求失败: {response.status_code}"
                return
            
            # 处理流式响应
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                    
                line = line[5:].strip()
                if line == "[DONE]":
                    break
                    
                try:
                    # chunk = eval(line)  # 解析JSON为Python字典
                    chunk = json.loads(line)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            content = delta["content"]
                            # 下面是为了适配deepseek-r1官方API,临时打的补丁。
                            if content == None:
                                content = ''
                            yield content
                except Exception as e:
                    logger.error(f"解析流式响应出错: {str(e)}, line: {line}")
                    
        except Exception as e:
            logger.exception(f"流式生成回答时出错: {str(e)}")
            yield f"抱歉，流式生成回答时出错: {str(e)}"


# 单例模式的实例
_model_instance = None

def get_openai_model() -> OpenAICompatibleLLM:
    """获取OpenAI兼容的LLM模型实例（单例模式）"""
    global _model_instance
    if _model_instance is None:
        _model_instance = OpenAICompatibleLLM()
    return _model_instance

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试模型
    model = get_openai_model()
    
    # 测试普通生成
    print("普通生成测试:")
    response = model.generate_response("你好，请介绍一下你自己")
    print(response)
    
    # 测试流式生成
    print("\n流式输出测试:")
    for chunk in model.generate_stream("中国有哪些著名的旅游景点？"):
        print(chunk, end="", flush=True) 
