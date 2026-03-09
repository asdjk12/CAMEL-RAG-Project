import os
from camel.storages import QdrantStorage
from camel.embeddings import SentenceTransformerEncoder
from camel.storages import VectorRecord


class QdrantDB:
    """简单的Qdrant向量数据库操作类"""
    
    def __init__(self, model_name: str = "TencentBAC/Conan-embedding-v1"):
        """
        初始化Qdrant数据库
        
        任务：
        1. 设置数据存储路径
        2. 初始化embedding模型
        3. 创建QdrantStorage实例
        
        参数:
            model_name: huggingface模型名称
        """
        # TODO: 设置rootpath（数据存储根目录）
        # 提示：使用os.path.dirname(__file__)获取当前文件路径
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        
        # TODO: 初始化SentenceTransformerEncoder
        # 提示：使用model_name参数
        self.embedding_instance = SentenceTransformerEncoder(
            model_name=model_name,       
        )
        
        # TODO: 初始化QdrantStorage
        # 提示：需要vector_dim和path参数
        self.storage_instance = QdrantStorage(
            vector_dim= self.embedding_instance.get_output_dim(),       # 模型输出维度等同于向量库输入维度
            path= os.path.join(self.root_path, "qdrant_data"),  # 新建qdrant_data在根目录下以存放数据
            collection_name="rag_docs",
        )
        
    def save_text(self, text: str, source_file: str = "unknown", idx:int= 0):
        """
        保存单个文本到数据库
        
        任务：
        1. 将文本转换为向量
        2. 创建VectorRecord
        3. 保存到数据库
        
        参数:
            text: 要保存的文本
            source_file: 文本来源文件名
        """
        if not text.strip():                # 空字符text检测
            raise ValueError("text cannot be empty")
        
        # TODO: 使用embedding_instance将文本转换为向量
        # 提示：使用embed_list方法
        text2vector = self.embedding_instance.embed_list(objs=[text])[0]       # 文本转向量
        
        # TODO: 创建payload字典，包含text和source_file信息
        payload = {
            "text": text,
            "source_file": source_file,
            "idx": idx
        }

        # TODO: 创建VectorRecord对象
        vector_record = VectorRecord(
            vector=text2vector,      
            payload=payload,
        )
        
        # TODO: 使用storage_instance.add()保存记录
        self.storage_instance.add([vector_record])

# 实习生任务：
# 完成上述TODO部分，实现一个能够将文本保存到Qdrant向量数据库的功能

    def text2vector(self, text, datbase, json_path:str = "unknown"):
        """
        适配small_ocr_content_list.json 文件类型JSON 输入的文本内容保存
        """
        for idx, content in enumerate(text):
            text = content.get("text","").strip()
            if content.get("type") == "text" and text:
                datbase.save_text(
                    text=text,
                    source_file=json_path,
                    idx=idx,
                )
        return 0


# 1. 创建数据库实例
if __name__ == "__main__":
    db = QdrantDB()

    try:
        # 2. 保存文本
        db.save_text("这是第一段文本", "文档1.txt")
        db.save_text("这是第二段文本", "文档2.txt")

        print("文本保存完成！")
    finally:
        db.storage_instance.close_client()

