import pyttsx3
import PyPDF2
from tkinter.filedialog import *

document = askopenfilename()
docReader = PyPDF2.PdfReader(document)
pages = docReader.numPages

for num in range(0, pages):
    page = docReader.getPage(num)
    text = page.extract_text()
    speaker = pyttsx3.init()
    speaker.say(text)
    speaker.runAndWait()