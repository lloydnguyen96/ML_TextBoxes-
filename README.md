# Tiền xử lý dữ liệu
## Tập dữ liệu CTWD
- Bước 1: Tải dữ liệu thô bằng cách
	+ Vào trang chủ: https://ctwdataset.github.io/ của bộ dữ liệu sau đó tới mục dataset.
	+ Hoặc tới thẳng đường link sau: https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ctw.
	+ Bước tiếp theo là điền đầy đủ form: tên, địa chỉ mail và học viện, nơi làm việc.
	+ Tới đường link vào thư mục dataset của ctwd trên Onedrive.
	+ Tải 2 folders là images-trainval và images-test
- Bước 2:
	+ Trong đồ án: images-test không được sử dụng.
	+ Đặt 2 thư mục images-trainval và images-test vào thư mục CTWD trong raw_dataset. Sau đó chạy file preprocess.py trong thư mục CTWD để tiền xử lý dữ liệu thô trong images-trainval. Quá trình tiền xử lý bao gồm: Bước 1: giải nén 26 folders có dạng ctw-trainval-*-of-26.tar. Việc giải nén được thực hiện bởi hàm extract_tarfiles. Bước 2: tách thành hai tập dữ liệu là tập training dataset và tập validation dataset và tách file train.jsonl cũng như file val.jsonl thành các file annotations có dạng .json cho từng ảnh trong bộ dữ liệu tương ứng. Kết quả thu được là hai folders training-dataset và validation-dataset ở cùng thư mục với folder images-test và images-trainval (thư mục CTWD). Hai folders này đều chứa hai folders con là images và annotations. Trong images và annotations đều chứ 26 folders tương ứng với 26 folders gốc. Công việc của bước 2 được thực hiện bởi hàm split_images_trainval. Bước 3: cắt ảnh thô thành các ảnh phù hợp cho mô hình cũng như sửa lại file annotation tương ứng với ảnh sau khi cắt. Bước 3 được thực hiện bởi hàm modify_ctwd_dataset_to_suit_ssd_vgg_model. Vị trí gọi cả 3 hàm trên đều là ở dưới cùng (trong hàm main) của file preprocess.py
	+ Để chạy được file preprocess.py cần sửa một số nội dung sau. Mở file và sửa các đường dẫn cho phù hợp: DATASET_DIRECTORY là đường dẫn tới thư mục CTWD. TARFILES_DIRECTORY là đường dẫn tới thư mục giải nén (ở đây là đường dẫn tới images-trainval). ANNOTATIONS_TRAIN_FILE_PATH là đường dẫn đến file train.jsonl. ANNOTATIONS_VAL_FILE_PATH là đường dẫn tới file val.jsonl. MODIFIED_DATASET_DIRECTORY là đường dẫn tới thư mục chứa các ảnh đã cắt và file annotations. Lưu ý quá trình cắt ảnh chỉ thực hiện tại một thời điểm với một trong hai bộ: training-dataset và validation-dataset. Do đó, cần chạy file preprocess 2 lần, lần 1 chạy 3 hàm extract_tarfiles, split_images_trainval và modify_ctwd_dataset_to_suit_ssd_vgg_model cho training-dataset và lần 2 chỉ chạy modify_ctwd_dataset_to_suit_ssd_vgg_model cho validation-dataset (comment hàm extract_tarfiles và split_images_trainval do 2 hàm này xử lý folder images-trainval nên chỉ cần chạy chúng 1 lần duy nhất). Ở mỗi lần chạy, sửa lại 4 arguments đầu tiên của modify_ctwd_dataset_to_suit_ssd_vgg_model sao cho phù hợp với tập training dataset và tập validation dataset.
	+ Chạy file preprocess.py. Kết quả thu được là 2 folder training-dataset và validation-dataset tại đường dẫn MODIFIED_DATASET_DIRECTORY.
- Bước 3: Convert dữ liệu vừa thu được thành các bản ghi của Tensorflow có đuôi .tfrecord
	+ Bước 1: bật cmd hay bash, di chuyển tới thu mục dataset/ctwd của project
	+ Bước 2: chạy python ctwd_to_tfrecords.py --dataset_directory=MODIFIED_DATASET_DIRECTORY --train_splits='training-dataset' --train_prefix='train' --val_prefix='val' --validation_splits='validation-dataset' --output_directory='./tfrecords'
	+ Bước 3: Nếu có lỗi về đường dẫn thì có thể mở trực tiếp file ctwd_to_tfrecords.py để sửa.
	+ Bước 4: Nếu có thu mục nào chưa tạo (output_directory) thì có thể tạo. Ví dụ tạo thư mục tfrecords trong folder ctwd.
## Tập dữ liệu RCTW
- Bước 1: Tải dữ liệu
	+ Vào http://rctw.vlrlab.net/dataset/
	+ Tải  Training images and annotations_v1.2(7.6G) và Testing images (3.8G) về
	+ Giải nén.
	+ Đặt 2 thư mục icdar2017rctw_test và icdar2017rctw_train_v1.2 vào thư mục RCTW trong raw_dataset
- Bước 2: Tiền xử lý dữ liệu. Ảnh giữ nguyên, chỉ sửa các file annotations. Cách làm là mở file preprocess.py và sửa các đường dẫn cho phù hợp:
	+ DATASET_DIRECTORY: đường dẫn tới thư mục RCTW
	+ Chỉ sử dụng tập huấn luyện làm tập đánh giá cho mô hình vì tập test không chưa file annotation. Do đó không phải sửa gì thêm và tiến hành chạy file proprocess.py
- Bước 3: Convert dữ liệu vừa thu được thành các bản ghi của Tensorflow có đuôi .tfrecord
	+ Bước 1: bật cmd hay bash, di chuyển tới thư mục dataset/rctw của project.
	+ Bước 2: tạo thư mục tfrecords
	+ Bước 3: Chạy python rctw_to_tfrecords.py --dataset_directory=đường dẫn tới RCTW --output_directory=đường dẫn tới thư mục tfrecords vừa tạo.
	+ Bươc 4: Enter chạy rctw_to_tfrecords.py
# Huấn luyện.
- Quá trình huấn luyện được thực hiện bởi hàm main trong file train_textboxes_plusplus.py
- Nếu huấn luyện từ đầu:
	+ Tại dòng 642 trong train_textboxes_plusplus.py sửa scaffold=tf.train.Scaffold(init_fn=get_init_fn()) thành scaffold=None.
	+ Mở cmd hoặc bash, di chuyển tới thư mục của project
	+ Chạy python train_textboxes_plusplus.py --multi_gpu=False (=True nếu dùng GPU) --data_dir='./dataset/ctwd/tfrecords' --model_dir='./logs/' (vị trí lưu model được huấn luyện) --data_format='channels_last' ('channels_first' nếu dùng GPU)
- Nếu fine tune một pretrained model:
	+ scaffold=tf.train.Scaffold(init_fn=get_init_fn()) giữ nguyên không thay đổi
	+ đặt pretrained model (file .ckpt file checkpoint, graph, ...) vào thư mục models của project
	+ Chạy python train_textboxes_plusplus.py --multi_gpu=False (=True nếu dùng GPU) --data_dir='./dataset/ctwd/tfrecords' --model_dir='./logs/' (vị trí lưu model được huấn luyện) --data_format='channels_last' ('channels_first' nếu dùng GPU)
- Nếu huấn luyện mô hình đề xuất: tại dòng thứ 11 sửa from config import textboxes_plusplus_config as config thành from config import modified_textboxes_plusplus_config as config và sửa dòng thứ 6 trong textboxes_plusplus_net.py từ from config import textboxes_plusplus_config as config thành from config import modified_textboxes_plusplus_config as config
- Nếu huấn luyện mô hình Textboxes++: giữ nguyên from config import textboxes_plusplus_config as config tại cả hai file.
- Enter để chạy huấn luyện.
# Đánh giá.
- File eval.py để đánh giá mô hình Textboxes++ và mô hình đề xuất.
- Nếu huấn luyện mô hình Textboxes++:
	+ Tại dòng thứ 11 của eval.py sửa thành from config import textboxes_plusplus_config as config
	+ sửa dòng thứ 6 trong textboxes_plusplus_net.py thành config import textboxes_plusplus_config as config
- Nếu huấn luyện mô hình đề xuất:
	+ Tại dòng thứ 11 của eval.py sửa thành from config import modified_textboxes_plusplus_config as config
	+ sửa dòng thứ 6 trong textboxes_plusplus_net.py từ from config import textboxes_plusplus_config as config thành from config import modified_textboxes_plusplus_config as config
- Đặt mô hình cần đánh giá (file .ckpt file checkpoint, graph, ...) vào thư mục models hoặc thư mục logs. Nếu để trong thư mục models thì thư mục logs phải không được chứa các model khác vì folder logs được ưu tiên trước.
- Mở cmd hoặc bash, gõ python eval.py --data_dir='./dataset/ctwd/tfrecords' ('./dataset/rctw/tfrecords' nếu dùng RCTW để đánh giá) 
- Nếu đánh giá bằng tập RCTW thì cần sửa dòng 602 từ dataset_pattern='val-*' thành dataset_pattern='train-*' bởi vì ta dùng tập train của RCTW để đánh giá mô hình.