import os
import re

import fitz


def _normalize_text(text):
    # 清洗提取出来的文本，减少空格和换行噪声
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text_by_page(pdf_path, ocr_page_fn=None):
    """
    按页提取 PDF 内容。

    两种思路：
    1. 如果 PDF 本身带文字，就直接提取文字。
    2. 如果 PDF 是扫描版，当前页提取不到文字，就转成图片后交给 OCR。

    参数：
    - pdf_path: PDF 文件路径
    - ocr_page_fn: OCR 函数，接收 (page, page_idx)，返回识别出的字符串

    """
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    page_texts = []
    document = fitz.open(pdf_path)

    try:
        for page_idx, page in enumerate(document):
            # 先尝试直接提取文字版 PDF 的内容
            page_text = _normalize_text(page.get_text("text"))

            # 如果这一页没有提取到文字，说明可能是扫描版 PDF
            if not page_text:
                # Todo： 接入OCR model 扫描pdf
                page_text = _normalize_text("")

            page_texts.append(page_text)
    finally:
        document.close()

    return page_texts


def pdf_question_to_text(pdf_path):
    """
    把问题 PDF 解析成一个字符串问题。

    流程：
    1. 按页提取内容
    2. 把所有非空页内容合并
    3. 返回最终的字符串 question
    """
    page_texts = extract_pdf_text_by_page(
        pdf_path=pdf_path,
    )

    question_text = "\n".join(text for text in page_texts if text)      # 按页拼接pdf问题
    question_text = _normalize_text(question_text)              

    if not question_text:
        raise ValueError("PDF 中没有解析出可用的问题内容")

    return question_text


def build_query_bundle(question_text: str):
    """
    构建Query Bundle。

    将已经解析好的pdf问题字符串转成结构化查询对象,对齐向量检索
    """

    raw_question = _normalize_text(question_text)       
    if not raw_question:
        raise ValueError("转化后的pdf问题内容为空,无法构建 Query Bundle")

    lines = [line.strip() for line in raw_question.split("\n") if line.strip()] # 格式化，去点空格与空行

    query_bundle = {
        "raw_question": raw_question,       # 改写后问题
        "clean_question": _normalize_text(raw_question),    
        "query_type": "pdf_question",           # 用于多模态检索方式的区分
        "search_queries": lines if lines else [raw_question],   # 单行/多行 问题
    }

    return query_bundle

if __name__ == "__main__":
    sample_pdf = r"C:\Users\cxc\Desktop\CV\PaymentReceipt_4322472175.pdf"
    if os.path.exists(sample_pdf):
        question = pdf_question_to_text(sample_pdf)
        bundle = build_query_bundle(question)
        print(bundle)
    else:
        print("没有找到示例 PDF 文件")
