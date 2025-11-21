dataset được lấy từ task 3 liver của medical decathlon

Các bước đã thực hiện:
- Tải dataset
- Tải các thư viện cần thiết: trong folder nnunet, chạy pip install -e .
- Đặt các environment variables: nnunet_raw, nnunet_preprocessed và nnunet_results là đường dẫn đến các folder tương ứng
- Chọn các file sẽ sử dụng và đưa vào imagesTr, imagesTs, labelTr tương ứng trong một thư mục đặt tên là Task03_Liver. Em đã chọn 16 files trong imagesTr để chạy thử
- sử dụng lệnh nnUNetv2_convert_MSD_dataset -i /path/to/Task03_Liver để đưa các file trong imagesTr, imagesTs, labelsTr vào nnunet_raw (theo environment variable), đổi lại tên theo convention của nnunetv2

Preprocess:
- Chạy lệnh nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity để bắt đầu tiền xử lý. Kết quả lưu trong nnunet_preprocessed (theo environment variable). nnUNetPlans.json sẽ chứa thông tin huấn luyện cho 2d, 3d_low_res và 3d_full_res
- Sau khi thử chạy với 3 layer background, liver, cancer không được, em đã chuyển liver vào background và chỉ tìm cancer. Để làm thế, em đã chạy script convert.py (cần chỉnh vị trí file tương ứng với vị trí nnunet_raw, sau khi xong sẽ tạo ra một bản sao của nnunet_raw với labels được sửa lại. Cần thay labels đó vào bản gốc)

Training:
- Chạy lệnh nnUNetv2_train 3 2d 0 --npz
3: task 003
2d: huấn luyện 2d, chọn giữa 2d, 3d_low_res và 3d_full_res.
0: fold 0. Sau khi chạy sẽ tạo 1 file splits_final.json trong nnunet_preprocessed. Có thể thay đổi file này để điều chỉnh thông tin các fold cho k-fold cross validation. Em đã chỉnh cho còn 4 fold

- Kết quả tương ứng các fold sẽ được lưu trong nnunet_results

Vấn đề: Kết quả không tốt, các fold khi tính dice score trong quá trình huấn luyện đều dưới 0.25 (chỉ tính cancer trong trưởng hợp 3 lớp background, liver, cancer) hoặc trong khoảng 0.3 đến 0.5 (2 lớp background và tumor)
