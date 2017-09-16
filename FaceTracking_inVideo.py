import face_recognition
import cv2
import array

VIDEO_DIR = "test.mp4" #视频的路径
TRACKING_FACE_DIR = ["face1.jpg" ,"face2.jpg","face3.jpg", "face4.jpg", "face5.jpg", "face6.jpg"] #在这里输入识别的脸的图片
TRACKING_FACE_LABEL = ["Sheldon", "Leonard", "Penny", "Howard", "Rajesh", "Amy"]    #在这里输入识别的脸的名称
VIDEO_SCALE = 1	#视频识别时的缩放倍数
FRAMES_READ = 1 #每次读取帧数的间隔
UNKNOWN_NAME = "Unknown"

capture = cv2.VideoCapture(VIDEO_DIR) #数字0是调用摄像头

'''
以下是将检测结果写入视频
'''

#获得码率及尺寸
fps = capture.get(cv2.CAP_PROP_FPS)
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#size = (704, 576)

videoWriter = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('F', 'L', 'V', '1'), fps, size)
 
'''
以上是将检测结果写入视频
'''


tracking_image_encodings = []
for face_dir in TRACKING_FACE_DIR:
    tracking_image = face_recognition.load_image_file(face_dir)   #此处输入跟踪的脸部目标
    tracking_image_encodings.append(face_recognition.face_encodings(tracking_image)[0])

#对变量初始化
face_locations = []
face_encodings = []
face_labels = []
do_this_frame = True

while True:
    #对帧的抓取
    for i in range(FRAMES_READ):
        ret, frame = capture.read()
    #print(frame)
    #缩小所截取的帧以减少运算
    cutdown_frame = cv2.resize(frame, (0, 0), fx = 1/VIDEO_SCALE, fy = 1/VIDEO_SCALE)#这里的fx，fy是缩小的倍数
    #print("OKOK")
    if do_this_frame:
        face_locations = face_recognition.face_locations(cutdown_frame)
        face_encodings = face_recognition.face_encodings(cutdown_frame, face_locations)

        face_labels = []
        #print("OKOK2")
        
            #for tracking_face in tracking_image_encodings:
            #print("OKOK3")
        for face_rec in face_encodings:
            match = face_recognition.compare_faces(tracking_image_encodings, face_rec, tolerance = 0.6)
            #print(match)
            has_match = False
            for if_match in match:
                if if_match :
                    name = TRACKING_FACE_LABEL[match.index(if_match)]
                    face_labels.append(name)
                    has_match = True
            if not has_match:
                name = UNKNOWN_NAME
                face_labels.append(name)
            #index = tracking_image_encodings.index(tracking_face)
            #print(face_labels)
            

    do_this_frame = not do_this_frame
    #print(face_labels)
    #显示结果
    for(top, right, bottom, left), name in zip(face_locations, face_labels):
        top *= VIDEO_SCALE
        right *= VIDEO_SCALE
        bottom *= VIDEO_SCALE
        left *= VIDEO_SCALE

        

        #if name != "unknown"
        cv2.rectangle(frame, (left, top), (right,bottom), (0,69,255), 2)  #框出人脸
        cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 59, 255), cv2.FILLED)    #画出标签的底色
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 29), font, 1.0, (0,255,255), 1)   #标出标签

    cv2.imshow('Playing...', frame)
    videoWriter.write(frame) #写视频帧

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows