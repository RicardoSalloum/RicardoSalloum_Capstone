import fitz  # PyMuPDF

#this script transforms .pdf into series of images to not cause errors in unity
pdf_path = "LectureFood.pdf"
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # Increase resolution

doc = fitz.open(pdf_path)

# save to files
for page in doc:
    pix = page.get_pixmap(matrix=mat)
    output = f"slide_{page.number + 1}.png"
    pix.save(output)
    print(f"Saved: {output}")

doc.close()