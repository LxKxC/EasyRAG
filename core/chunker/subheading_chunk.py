import re
from typing import List, Dict, Any, Optional, Tuple

class SubheadingTextSplitter:
    """基于小标题的文本分割器，保留大标题和内容结构"""
    
    def __init__(self, 
                 chunk_size=1000, 
                 chunk_overlap=200, 
                 main_headers_level=1, 
                 subheaders_level=[2, 3, 4, 5, 6]):
        """
        初始化小标题文本分割器
        
        参数:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小（字符数）
            main_headers_level: 要保留的主标题级别 (1表示#, 2表示##, 等)
            subheaders_level: 用于分块的子标题级别列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.main_headers_level = main_headers_level
        self.subheaders_level = subheaders_level
        
        # 默认的Markdown标题标记
        self.header_markers = ["#", "##", "###", "####", "#####", "######"]
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        根据小标题将文本分割成块，同时保留大标题结构
        
        参数:
            text: 要分割的Markdown文本
            
        返回:
            包含分割后文本块的列表，每个块包含内容和元数据
        """
        # 分割文本为行
        lines = text.split('\n')
        
        # 初始化结果列表和当前块
        chunks = []
        current_main_headers = {}  # 存储主标题
        current_sub_headers = {}   # 存储子标题
        current_content = []
        
        # 当前标题级别状态
        current_section = None
        is_in_main_header = False
        current_header_level = 0
        
        for line in lines:
            # 检查是否是标题行
            header_match = None
            header_level = 0
            
            # 检测标题级别
            for level, marker in enumerate(self.header_markers, 1):
                if line.strip().startswith(marker + " "):
                    header_match = line.strip()[len(marker) + 1:]
                    header_level = level
                    break
            
            # 处理标题行
            if header_match is not None:
                # 确定标题类型（主标题或子标题）
                is_main_header = header_level <= self.main_headers_level
                
                # 如果是子标题级别标题，处理当前内容块
                if header_level in self.subheaders_level:
                    # 如果有累积的内容，创建一个新块
                    if current_content:
                        chunk_text = "\n".join(current_content).strip()
                        if chunk_text:
                            chunks.append({
                                "content": chunk_text,
                                "metadata": {
                                    **current_main_headers,  # 主标题信息
                                    **current_sub_headers,   # 子标题信息
                                    "main_header": current_main_headers.get(str(self.main_headers_level), ""),
                                    "sub_header": current_section
                                }
                            })
                        current_content = []
                
                # 更新标题信息
                if is_main_header:
                    # 对于主标题，更新主标题信息
                    current_main_headers[str(header_level)] = header_match
                    # 清除所有子标题
                    current_sub_headers = {}
                    is_in_main_header = True
                    current_section = header_match
                else:
                    # 对于子标题，更新子标题信息
                    # 移除所有等级大于等于当前子标题的子标题
                    for level in list(current_sub_headers.keys()):
                        if int(level) >= header_level:
                            current_sub_headers.pop(level)
                    
                    # 添加当前子标题
                    current_sub_headers[str(header_level)] = header_match
                    current_section = header_match
                
                # 将标题行添加到当前内容
                current_content.append(line)
                current_header_level = header_level
            else:
                # 非标题行，直接添加到当前内容
                current_content.append(line)
        
        # 处理最后一个块
        if current_content:
            chunk_text = "\n".join(current_content).strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **current_main_headers,
                        **current_sub_headers,
                        "main_header": current_main_headers.get(str(self.main_headers_level), ""),
                        "sub_header": current_section
                    }
                })
        
        # 处理过大的块
        result_chunks = []
        for chunk in chunks:
            if len(chunk["content"]) > self.chunk_size:
                # 如果块太大，进一步分割
                sub_chunks = self._split_large_chunk(chunk["content"])
                for i, sub_chunk in enumerate(sub_chunks):
                    # 为分割后的块添加序号信息
                    metadata = {**chunk["metadata"]}
                    metadata["is_split"] = True
                    metadata["split_index"] = i + 1
                    metadata["total_splits"] = len(sub_chunks)
                    
                    result_chunks.append({
                        "content": sub_chunk,
                        "metadata": metadata
                    })
            else:
                result_chunks.append({
                    "content": chunk["content"],
                    "metadata": {**chunk["metadata"], "is_split": False}
                })
        
        # 格式化最终结果
        formatted_chunks = []
        for chunk in result_chunks:
            # 标准化为统一格式
            formatted_chunks.append({
                "text": chunk["content"],
                "metadata": chunk["metadata"]
            })
        
        return formatted_chunks
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """
        将大块文本分割成更小的块，保持段落完整性
        
        参数:
            text: 要分割的文本
            
        返回:
            分割后的文本列表
        """
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # 如果当前段落本身就超过了块大小，需要进一步分割
            if para_size > self.chunk_size:
                # 先处理当前累积的块
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # 分割大段落
                words = para.split(' ')
                temp_chunk = []
                temp_size = 0
                
                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if temp_size + word_size > self.chunk_size:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_size = word_size
                    else:
                        temp_chunk.append(word)
                        temp_size += word_size
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
            
            # 如果添加当前段落会超过块大小，创建新块
            elif current_size + para_size + 2 > self.chunk_size:  # +2 for newlines
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for newlines
        
        # 添加最后一个块
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks 




