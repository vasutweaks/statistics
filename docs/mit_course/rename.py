import glob
import os

for f in glob.glob("*.pdf"):
    strn=f.split(".")[0]
    strn=strn.split("_")[::-1]
    strn1='_'.join(x for x in strn)+'.pdf'
    print(strn1)
    os.rename(f,strn1)
