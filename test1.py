
# ===== GPIO設定 =====
GPIO.setmode(GPIO.BCM)   # ピン番号の指定方法
pin = 18                 # 使用するGPIOピン番号
GPIO.setup(pin, GPIO.OUT)

# ===== 判定対象画像 =====
img_path = "/home/nisimuramasato/ドキュメント/GitHub/tngchina/coffee_dataset/normal/5.jpg"
img = Image.open(img_path)
img = transform(img).unsqueeze(0)

# ===== 推論 =====
classes = ['defect', 'normal']
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1)

result = classes[pred.item()]
print("判定結果:", result)

# ===== defectなら電気信号を流す =====
if result == "defect":
    print("⚡ defect → 電気信号を送ります")
    GPIO.output(pin, GPIO.HIGH)   # 電気信号 ON
    time.sleep(1)                 # 1秒間信号を流す
    GPIO.output(pin, GPIO.LOW)    # 電気信号 OFF

# ===== 終了処理 =====
GPIO.cleanup()