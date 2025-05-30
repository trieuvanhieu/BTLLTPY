import os
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from database import connect_db
from datetime import datetime

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = r"E:\\Work\\KHDL\\NHANDANGCACBENHVEDATRENDONGVA\\runs\\detect\\train\\weights\\best.pt"

NHAN_BENH = [
    'viêm da do vi khuẩn (chó)', 'dị ứng bọ chét', 'nhiễm nấm (chó)',
    'da khỏe mạnh', 'chó khỏe mạnh', 'quá mẫn (chó)', 'hắc lào', 'ghẻ'
]

BIỂU_HIỆN_BỆNH = {
    'viêm da do vi khuẩn (chó)': ['mụn mủ', 'da đỏ', 'rụng lông'],
    'dị ứng bọ chét': ['gãi nhiều', 'bọ chét'],
    'nhiễm nấm (chó)': ['vảy gàu', 'ngứa dai dẳng'],
    'ghẻ': ['da đóng vảy', 'ngứa nghiêm trọng'],
    'hắc lào': ['vòng tròn đỏ'],
    'quá mẫn (chó)': ['phát ban', 'mẩn đỏ'],  # Nếu cần, bạn thêm vào form triệu chứng này
    'da khỏe mạnh': [],
    'chó khỏe mạnh': []
}

DE_XUAT_CHUA_TRI = {
    'viêm da do vi khuẩn (chó)': 'Dùng thuốc mỡ kháng sinh như Neomycin.',
    'dị ứng bọ chét': 'Tắm bằng sữa tắm diệt ve, dùng thuốc chống dị ứng.',
    'nhiễm nấm (chó)': 'Sử dụng thuốc kháng nấm như ketoconazole.',
    'da khỏe mạnh': 'Không cần điều trị.',
    'chó khỏe mạnh': 'Không cần điều trị.',
    'quá mẫn (chó)': 'Tránh tiếp xúc chất gây dị ứng, dùng thuốc kháng histamin.',
    'hắc lào': 'Dùng thuốc trị nấm như miconazole, vệ sinh da kỹ.',
    'ghẻ': 'Tắm bằng dung dịch trị ghẻ, dùng thuốc ivermectin.'
}

treatment_plans = {
    "Viêm da do vi khuẩn": [
        {
            "name": "Phác đồ 1: Kháng sinh đường uống",
            "price": "500.000 VNĐ",
            "description": "Sử dụng kháng sinh phù hợp trong 7-10 ngày, kết hợp vệ sinh vùng tổn thương.",
            "steps": [
                "Uống thuốc kháng sinh theo toa bác sĩ.",
                "Rửa sạch vùng da bị viêm mỗi ngày.",
                "Theo dõi tiến triển và tái khám."
            ]
        },
        {
            "name": "Phác đồ 2: Thuốc bôi ngoài da",
            "price": "300.000 VNĐ",
            "description": "Sử dụng thuốc mỡ kháng khuẩn tại chỗ kết hợp với chăm sóc da.",
            "steps": [
                "Bôi thuốc mỡ 2 lần/ngày lên vùng da bị viêm.",
                "Tránh tiếp xúc với chất gây kích ứng.",
                "Giữ vùng da khô ráo."
            ]
        }
    ],
    "Dị ứng bọ chét": [
        {
            "name": "Phác đồ 1: Thuốc diệt bọ chét",
            "price": "400.000 VNĐ",
            "description": "Sử dụng thuốc diệt bọ chét chuyên dụng kết hợp vệ sinh môi trường sống.",
            "steps": [
                "Thoa thuốc diệt bọ chét cho thú cưng theo hướng dẫn.",
                "Giặt giũ chăn, nệm thường xuyên.",
                "Dọn dẹp sạch sẽ môi trường sống."
            ]
        }
    ],
    "Nhiễm nấm": [
        {
            "name": "Phác đồ 1: Thuốc chống nấm toàn thân",
            "price": "600.000 VNĐ",
            "description": "Sử dụng thuốc chống nấm theo toa kết hợp vệ sinh sạch sẽ.",
            "steps": [
                "Uống thuốc chống nấm theo chỉ định.",
                "Vệ sinh và làm sạch vùng da tổn thương.",
                "Tránh để thú cưng tiếp xúc với nguồn lây."
            ]
        }
    ],
    "Hắc lào": [
        {
            "name": "Phác đồ 1: Thuốc bôi chống nấm",
            "price": "350.000 VNĐ",
            "description": "Bôi thuốc chống nấm tại chỗ và giữ vùng da sạch sẽ.",
            "steps": [
                "Bôi thuốc chống nấm 2 lần/ngày.",
                "Giữ vùng da thoáng khí, sạch sẽ.",
                "Không để vật nuôi gãi mạnh gây tổn thương."
            ]
        }
    ],
    "Ghẻ": [
        {
            "name": "Phác đồ 1: Thuốc tẩy ghẻ toàn thân",
            "price": "700.000 VNĐ",
            "description": "Sử dụng thuốc tẩy ghẻ đường toàn thân kết hợp vệ sinh môi trường.",
            "steps": [
                "Tiêm hoặc uống thuốc tẩy ghẻ theo hướng dẫn.",
                "Làm sạch môi trường sống của thú cưng.",
                "Theo dõi tình trạng da sau điều trị."
            ]
        }
    ],
    "Quá mẫn": [
        {
            "name": "Phác đồ 1: Thuốc chống dị ứng",
            "price": "450.000 VNĐ",
            "description": "Sử dụng thuốc chống dị ứng và thay đổi chế độ ăn nếu cần.",
            "steps": [
                "Cho thú cưng dùng thuốc chống dị ứng theo toa.",
                "Theo dõi dấu hiệu dị ứng mới.",
                "Tham khảo ý kiến bác sĩ về thay đổi chế độ ăn."
            ]
        }
    ]
}
BENH_CHUAN_HOA = {
    'viêm da do vi khuẩn (chó)': 'Viêm da do vi khuẩn',
    'dị ứng bọ chét': 'Dị ứng bọ chét',
    'nhiễm nấm (chó)': 'Nhiễm nấm',
    'ghẻ': 'Ghẻ',
    'hắc lào': 'Hắc lào',
    'quá mẫn (chó)': 'Quá mẫn',
    'da khỏe mạnh': 'Da khỏe mạnh',
    'chó khỏe mạnh': 'Chó khỏe mạnh'
}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = YOLO(MODEL_PATH)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def doan_tu_bieu_hien(symptoms):
    max_trung = 0
    benh_chon = None
    for benh, hientuong in BIỂU_HIỆN_BỆNH.items():
        dem = sum(ht in symptoms for ht in hientuong)
        if dem > max_trung:
            max_trung = dem
            benh_chon = benh
    return benh_chon

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        symptoms = request.form.getlist("symptoms")

        if file:
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result_path = os.path.join(app.config['RESULT_FOLDER'], filename)

            file.save(upload_path)
            results = model(upload_path)

            ten_benh_tu_anh = (
                NHAN_BENH[int(results[0].boxes.cls[0])]
                if results[0].boxes.cls.numel() > 0 else "Không phát hiện"
            )

            ten_benh_tu_bieu_hien = doan_tu_bieu_hien(symptoms)

            if ten_benh_tu_bieu_hien:
                ten_benh = ten_benh_tu_bieu_hien
            else:
                ten_benh = ten_benh_tu_anh

            de_xuat = DE_XUAT_CHUA_TRI.get(ten_benh, "Không có gợi ý điều trị.")

            results[0].save(filename=result_path)

            try:
                conn = connect_db()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO BENHAN (TenAnh, DuongDanAnh, TenBenh, PhuongPhapDieuTri, NgayChanDoan)
                    VALUES (?, ?, ?, ?, ?)
                """, (filename, result_path, ten_benh, de_xuat, datetime.now()))
                conn.commit()
                conn.close()
            except Exception as e:
                print("❌ Lỗi cơ sở dữ liệu:", e)

            return render_template("index.html", image_filename=filename, result=ten_benh, advice=de_xuat)

    return render_template("index.html")


@app.route("/dieutri", methods=["POST"])
def dieutri():
    ten_benh = request.form.get("benh")
    ten_benh_chuan = BENH_CHUAN_HOA.get(ten_benh, ten_benh)  # ánh xạ đúng
    lieu_trinh = treatment_plans.get(ten_benh_chuan, [])
    return render_template("dieutri.html", benh=ten_benh_chuan, lieu_trinh=lieu_trinh)


if __name__ == "__main__":
    app.run(debug=True)
