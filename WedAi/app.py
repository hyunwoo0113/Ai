import pytesseract

image_path = '학뒤.jpg'
language = 'kor'
result = pytesseract.image_to_string(image_path, lang='kor', config='--psm 6')
result=result.replace(" ","")

print(result)

index=result.find("세종장영실고등학교")
if(index!=-1):
    print("장영실고학생")

