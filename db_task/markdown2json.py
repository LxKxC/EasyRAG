import re
import logging
from typing import List, Dict, Any, Optional
# from db_task.docx_to_markdown import DocxToMarkdown

logger = logging.getLogger(__name__)

class Markdown2Json:
    """
    将Markdown文本转换为JSON格式，按段落分块
    """
    
    def __init__(self, chunk_size: int = 1000):
        """
        初始化转换器
        
        参数:
            chunk_size (int): 内容分块的最大字符数
        """
        self.chunk_size = chunk_size
    
    def convert(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        将Markdown文本转换为JSON格式
        
        参数:
            markdown_text (str): Markdown格式的文本
            
        返回:
            List[Dict[str, Any]]: 转换后的JSON数据列表，每个元素包含title、bigtitle和content
        """
        try:
            # 分割文本为行
            lines = markdown_text.split('\n')
            
            result = []
            current_bigtitle = ""
            current_title = ""
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检查是否为一级标题
                if line.startswith('# '):
                    # 保存之前的内容
                    if current_content:
                        chunks = self._split_content_into_chunks('\n\n'.join(current_content))
                        for chunk in chunks:
                            result.append({
                                "bigtitle": current_bigtitle,
                                "title": current_title,
                                "content": chunk
                            })
                        current_content = []
                    
                    current_bigtitle = line[2:].strip()
                    current_title = ""
                
                # 检查是否为二级标题
                elif line.startswith('## '):
                    # 保存之前的内容
                    if current_content:
                        chunks = self._split_content_into_chunks('\n\n'.join(current_content))
                        for chunk in chunks:
                            result.append({
                                "bigtitle": current_bigtitle,
                                "title": current_title,
                                "content": chunk
                            })
                        current_content = []
                    
                    current_title = line[3:].strip()
                
                # 普通内容
                else:
                    current_content.append(line)
            
            # 处理最后一块内容
            if current_content:
                chunks = self._split_content_into_chunks('\n\n'.join(current_content))
                for chunk in chunks:
                    result.append({
                        "bigtitle": current_bigtitle,
                        "title": current_title,
                        "content": chunk
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"转换Markdown到JSON失败: {str(e)}")
            raise ValueError(f"转换失败: {str(e)}")
    
    def _split_content_into_chunks(self, text: str) -> List[str]:
        """
        将内容分割成适当大小的块
        
        参数:
            text (str): 要分割的文本
            
        返回:
            List[str]: 分割后的文本块列表
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
            elif current_size + para_size > self.chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # 添加最后一个块
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks


if __name__ == "__main__":
    from docx_to_markdown5 import DocxToMarkdown
    
    docx_to_markdown = DocxToMarkdown()
    title_list = ['']
    for title in title_list:
    #     print(title)
    #     markdown_text = docx_to_markdown.convert(f"/home/user/new/EasyRAG/gov/{title}.docx")
    
    #     markdown2json = Markdown2Json()
    #     json_data = markdown2json.convert(markdown_text)
    #     print(json_data)
    #     for line in json_data:
    #         print(line)
    #         line['text'] = f'标题：{line["title"]}, 内容: {line["content"]}, 内容所属大标题：{line["bigtitle"]}'
    #     print(json_data)
    # for item in json_data:
    #     print(f"大标题: {item['bigtitle']}")
    #     print(f"标题: {item['title']}")
    #     print(f"内容: {item['content'][:100]}...")
    #     print("-" * 50)
        chunks = docx_to_markdown.convert_and_chunk(f"/home/user/new/EasyRAG/gov/{title}.docx")
        print(f"\n总共分成了 {len(chunks)} 个块")
        jsondb = []
        for i, chunk in enumerate(chunks):  # 只打印前3个块作为示例
            print('********************************************')
            print(f"\n块 {i+1}:")
            print(f"一级标题: {chunk['h1']}")
            print(f"二级标题: {chunk['h2']}")
            print(f"当前标题: {chunk['title']}")
            print(f"内容预览: {chunk['content']}...")
            if len(chunk['content']) < 15:
                continue
            json_data = {
                "bigtitle": chunk['h1'],
                "title": chunk['title'],
                "content": chunk['content'],
                "text": f"标题：{chunk['title']}, 内容: {chunk['content']}, 内容所属大标题：{chunk['h1']}"
            }
            jsondb.append(json_data)

            # 写入知识库
            import os
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from main import RAGService
            rag_service = RAGService()
            rag_service.add_documents(kb_name="gov", documents=jsondb)
