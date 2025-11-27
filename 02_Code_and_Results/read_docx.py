import docx
import sys

try:
    doc = docx.Document("Strategic Architecture for Educational Data Mining.docx")
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    with open("docx_content.txt", "w", encoding="utf-8") as f:
        f.write('\n'.join(fullText))
    print("Successfully wrote to docx_content.txt")
except Exception as e:
    print(f"Error reading docx: {e}")
