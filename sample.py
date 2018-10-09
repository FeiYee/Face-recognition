import face_recognition
import cv2
import os

from tqdm import tqdm

def dirName(path):
    '''
    读取路径下所有子文件夹名与文件名
    :param path: 根目录
    :return: [文件夹1名，文件夹2名，......]，[文件1名，文件2名，......]
    '''
    for _, dirs, files in os.walk(path):
        return dirs,files


def start(all_name,face_path,base):
    '''
    :param all_name: 所有用户名称
    :param face_path: 所有人脸照片路径
    :param base: 每人几张照片
    :return: None
    '''
    # 通过阀值
    pass_base = int(base * 0.8)
    video_capture = cv2.VideoCapture(0)
    face_all = []
    for _path in tqdm(face_path,"读取人脸库"):
        face_all.append(face_recognition.face_encodings(face_recognition.load_image_file(_path))[0])

    face_locations = []
    face_names = []
    process_this_frame = True

    while True:
        # 读取摄像头画面
        ret, frame = video_capture.read()

        # 改变摄像头图像的大小，图像小，所做的计算就少
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # 判断是否为同一人
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # 默认为unknown
                match = face_recognition.compare_faces(face_all, face_encoding, tolerance=0.37)
                name = "Unknow"
                temp = []
                for (i,_bool) in enumerate(match):
                    if _bool:
                        temp.append(i // base)
                        name = str(all_name[i // base])
                if name in all_name:
                    num = temp.count(all_name.index(name))
                    name += ": " + str(num / base * 100) + "%"
                    if num >= pass_base:
                        print("通过")

                face_names.append(name)

        process_this_frame = not process_this_frame

        # 将捕捉到的人脸显示出来
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 矩形框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            #加上标签
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display
        cv2.imshow('Video', frame)

        # 按Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

FACE_ROOT = "facelib/"
IMAGE_DIR = []
# 读取所有人名
NAME,_ = dirName(FACE_ROOT)
# 读取所有图像路径
for path in NAME:
    for _,image_name in dirName(FACE_ROOT + path):
        IMAGE_DIR.append(FACE_ROOT + path + "/" + image_name)
# 开始检测识别
start(NAME,IMAGE_DIR,2)
