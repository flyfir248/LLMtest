import fitz

filepath = "./B.pdf"

doc = fitz.open(filepath)
if doc.page_count > 0:
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if len(text.strip()) > 0:
            print(f"Page {page_num + 1}:")
            print(text)
        else:
            print(f"No text found on page {page_num + 1}.")
else:
    print("PDF file is empty.")
doc.close()