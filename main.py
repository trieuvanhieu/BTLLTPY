import os
from datetime import datetime
from ultralytics import YOLO
from tkinter import Tk, filedialog
import cv2

# ------------------------- CAU HINH -------------------------
MODEL_PATH = r"E:\Work\KHDL\NHANDANGCACBENHVEDATRENDONGVA\runs\detect\train\weights\best.pt"
THU_MUC_LUU_ANH = "diagnose"
os.makedirs(THU_MUC_LUU_ANH, exist_ok=True)

NHAN_BENH = [
    'viem da do vi khuan (cho)', 'di ung bo chet', 'nhiem nam (cho)',
    'da khoe manh', 'cho khoe manh', 'qua man (cho)', 'hac lao', 'ghe'
]

DE_XUAT_CHUA_TRI = {
    'viem da do vi khuan (cho)': 'Dung thuoc mo khang sinh nhu Neomycin.',
    'di ung bo chet': 'Tam bang sua tam diet ve, dung thuoc chong di ung.',
    'nhiem nam (cho)': 'Su dung thuoc khang nam nhu ketoconazole.',
    'da khoe manh': 'Khong can dieu tri.',
    'cho khoe manh': 'Khong can dieu tri.',
    'qua man (cho)': 'Tranh tiep xuc chat gay di ung, dung thuoc khang histamin.',
    'hac lao': 'Dung thuoc tri nam nhu miconazole, ve sinh da ky.',
    'ghe': 'Tam bang dung dich tri ghe, dung thuoc ivermectin.'
}

# ------------------------- HAM TIEN ICH -------------------------
def chon_tep_anh():
    Tk().withdraw()
    tep = filedialog.askopenfilename(title="Chon anh cho can chan doan", filetypes=[("Anh", "*.jpg *.png *.jpeg")])
    return tep

def hien_thi_anh_va_nhan(tep_anh, ket_qua):
    img = cv2.imread(tep_anh)
    cv2.putText(img, f"Benh: {ket_qua}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Ket qua chan doan", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------- CHUONG TRINH CHINH -------------------------
def main():
    print("🐶 UNG DUNG NHAN DIEN BENH NGOAI DA TREN CHO")

    tep_anh = chon_tep_anh()
    if not tep_anh:
        print("❌ Ban chua chon anh.")
        return

    # Load mo hinh
    model = YOLO(MODEL_PATH)

    # Du doan
    results = model(tep_anh)
    ket_qua = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes.cls.numel() > 0 else "Khong phat hien"

    # Dich ten nhan
    try:
        ten_benh = NHAN_BENH[int(results[0].boxes.cls[0])]
    except:
        ten_benh = "Khong phat hien benh"

    # Goi y cham soc
    de_xuat = DE_XUAT_CHUA_TRI.get(ten_benh, "Khong co goi y dieu tri.")

    # Hien thi va luu
    print(f"🧾 Ket qua: {ten_benh}")
    print(f"💡 Goi y cham soc: {de_xuat}")

    # Luu hinh anh ket qua (neu co)
    ten_tep = os.path.basename(tep_anh)
    ketqua_anh = os.path.join(THU_MUC_LUU_ANH, ten_tep)
    results[0].save(filename=ketqua_anh)

    # Hien thi anh ket qua
    hien_thi_anh_va_nhan(ketqua_anh, ten_benh)

if __name__ == "__main__":
    main()
