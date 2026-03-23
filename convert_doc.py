import pypandoc
import os

print("Downloading pandoc if not installed...")
pypandoc.download_pandoc()

print("Converting Report.md to Report.docx...")
pypandoc.convert_file('Report.md', 'docx', outputfile='Report.docx')
print("Done!")
