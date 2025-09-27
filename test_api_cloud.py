import requests

API = "https://veriface-api-2r6izsajpa-uc.a.run.app/"

with open("dfake1.mp4","rb") as f:
    r = requests.post(API, files={"file":("dfake1.mp4", f, "video/mp4")})
print("Status:", r.status_code)
print(str(r.json())[:400], "...")
