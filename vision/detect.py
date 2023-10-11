import argparse
import time
from pathlib import Path

import pymysql
import pandas as pd
import datetime
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os.path
from gtts import gTTS

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    # 안전모 미착용, 쓰러짐 시간, 전우조를 위한 변수
    head_count = 0
    lay_count = 0
    person_count = 0
    h_count = 0
    l_count = 0
    p_count = 0
    vest_count = 0
    v_count = 0
    
    # db연동
    conn = pymysql.connect(host="127.0.0.1", user = "root", password = "1234", db ="harbor", charset ="utf8")
    cur = conn.cursor()

    # 빈 데이터프레임 생성
    df = pd.DataFrame(columns=["case", "casetime", "casetype", "caseplace", "caseimage"])
    # 여기 부분이 계속해서 반복되면서 진행되는 것
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                ''' 고치기 시작한 부분 '''
                s_list = []
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n}{names[int(c)]}"  # add to string
                    s_list.append(s)

                count = []
                extract = []

                for i in range(len(s_list)):
                    print(f'\n{s_list[i]}')
                    count.append(s_list[i][:1])
                    extract.append(s_list[i][1:])


                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H시")
                
                # 안전모 미착용시
                if 'head' in extract:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    head_count += 1
                else:
                    head_count = 0

                # 안전조끼 미착용시
                if ('head' in extract and 'vest' not in extract) or ('safety hat' in extract and 'vest' not in extract):
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    vest_count += 1
                else:
                    vest_count = 0
                
                # 쓰러짐 감지
                if 'lay' in extract:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    lay_count += 1
                else:
                    lay_count = 0       

                # 전우조
                if 'safety hat' in extract:       
                    if count[0]=='1' and 'head' not in extract:     
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        person_count += 1  
                        print('\n=========안전모 하나만 있는 경우 전우조 안 지킴==========', end='')
                    else:
                        person_count = 0
                        print('\n=========전우조 잘 지키네 굳!!!!!!!!!!!============', end='')
                elif 'head' in extract:
                    if count[0]=='1':     
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        person_count += 1  
                        print('\n========헤드 하나만 있는 경우 전우조 안 지킴==========', end='')
                    else:
                        person_count = 0
                else:
                    person_count = 0

                print(f'\n현재 시간 : {current_time}')
                print(count)
                print(extract)
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # 이미지 있으면 실행
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 안전모 미착용 2초
                    if head_count == 130:              
                        print('============= 현재 2초로 체크==============')
                        print('=============하이바 써라==============')
                        print('=============하이바 써라==============')
                        print('=============하이바 써라==============')
                        print('=============하이바 써라==============')
                        print('=============하이바 써라==============')

                        # 데이터프레임 형태로 csv파일 생성
                        head_count = 0
                        h_count += 1
                        df = pd.concat([df, pd.DataFrame({"case": f'안전모사건{h_count}', "casetime": current_time,
                                        "casetype": "안전모 미착용", "caseplace": "선내",
                                        "caseimage": f"C:/Users/admin/Desktop/dataframe/safetyhat{h_count}.jpg"}, index=[h_count])], ignore_index=True)
                        
                        # 이미지 저장
                        save = f'C:/xampp/htdocs/myimages/safetyhat{h_count}.jpg'
                        cv2.imwrite(save, im0)
                        image_url = f"http://localhost/myimages/safetyhat{h_count}.jpg"

                        # 데이터베이스 삽입
                        query = "INSERT INTO precaution (`case`, casetime, casetype, caseplace, caseimage) VALUES (%s, %s, %s, %s, %s)"
                        values = (f'안전모 사건 {h_count}', current_time, '안전모 미착용', 'sector1', image_url)
                        cur.execute(query, values)
                        conn.commit()

                        # text = "안전모 착용법 위반 안전모 착용법 위반 즉시 안전모를 착용하시기 바랍니다."
                        # # gTTS를 사용하여 텍스트를 한국어로 음성으로 변환
                        # tts = gTTS(text, lang='ko')  # 'ko'는 한국어를 나타냅니다. 원하는 언어 코드로 변경할 수 있습니다.            
                        # # 음성 출력
                        # tts.play()

                    # 안전조끼 미착용 2초
                    if head_count == 130:              
                        print('============= 현재 2초로 체크==============')
                        print('=============조끼 껴라==============')
                        print('=============조끼 껴라==============')
                        print('=============조끼 껴라==============')
                        print('=============조끼 껴라==============')
                        print('=============조끼 껴라==============')

                        # 데이터프레임 형태로 csv파일 생성
                        vest_count = 0
                        v_count += 1
                        df = pd.concat([df, pd.DataFrame({"case": f'안전조끼사건{v_count}', "casetime": current_time,
                                        "casetype": "안전조끼 미착용", "caseplace": "선내",
                                        "caseimage": f"C:/Users/admin/Desktop/dataframe/safetyhat{v_count}.jpg"}, index=[v_count])], ignore_index=True)
                        
                        # 이미지 저장
                        save = f'C:/xampp/htdocs/myimages/vest{v_count}.jpg'
                        cv2.imwrite(save, im0)
                        image_url = f"http://localhost/myimages/vest{v_count}.jpg"

                        # 데이터베이스 삽입
                        query = "INSERT INTO precaution (`case`, casetime, casetype, caseplace, caseimage) VALUES (%s, %s, %s, %s, %s)"
                        values = (f'안전조끼 미착용 사건 {v_count}', current_time, '안전조끼 미착용', 'sector1', image_url)
                        cur.execute(query, values)
                        conn.commit()

                        # text = "안전모 착용법 위반 안전모 착용법 위반 즉시 안전모를 착용하시기 바랍니다."
                        # # gTTS를 사용하여 텍스트를 한국어로 음성으로 변환
                        # tts = gTTS(text, lang='ko')  # 'ko'는 한국어를 나타냅니다. 원하는 언어 코드로 변경할 수 있습니다.            
                        # # 음성 출력
                        # tts.play()

                    #쓰러짐 2초
                    if lay_count == 130:
                        print('============= 현재 2초로 체크==============')
                        print('==============일어나라=================')
                        print('==============일어나라=================')
                        print('==============일어나라=================')
                        print('==============일어나라=================')
                        print('==============일어나라=================')

                        lay_count=0
                        l_count+=1
                        df = pd.concat([df, pd.DataFrame({"case": f'쓰러짐사건{l_count}', "casetime": current_time,
                                        "casetype": "쓰러짐", "caseplace": "선외",
                                        "caseimage": f"C:/xampp/htdocs/myimages/lay{l_count}.jpg"}, index=[l_count])], ignore_index=True)
                        
                        # 이미지 저장
                        save = f'C:/xampp/htdocs/myimages/lay{l_count}.jpg'
                        cv2.imwrite(save, im0)
                        image_url = f"http://localhost/myimages/lay{l_count}.jpg"

                        # # 텍스트 생성
                        # text = "쓰러짐"
                        # # gTTS를 사용하여 텍스트를 한국어로 음성으로 변환
                        # tts = gTTS(text, lang='ko')  # 'ko'는 한국어를 나타냅니다. 원하는 언어 코드로 변경할 수 있습니다.           
                        # # 음성 출력
                        # tts.play()

                        # 데이터베이스 삽입
                        query = "INSERT INTO precaution (`case`, casetime, casetype, caseplace, caseimage) VALUES (%s, %s, %s, %s, %s)"
                        values = (f'쓰러짐 사건 {l_count}', current_time, '쓰러짐', 'sector2', image_url)
                        cur.execute(query, values)
                        conn.commit()

                    # 전우조 안지킴 2초
                    if person_count == 130:
                        print('============= 현재 2초로 체크==============')
                        print('==============전우조 지켜라=================')
                        print('==============전우조 지켜라=================')
                        print('==============전우조 지켜라=================')
                        print('==============전우조 지켜라=================')
                        print('==============전우조 지켜라=================')

                        person_count=0
                        p_count+=1
                        df = pd.concat([df, pd.DataFrame({"case": f'전우조사건{p_count}', "casetime": current_time,
                                        "casetype": "전우조", "caseplace": "야적",
                                        "caseimage": f"C:/xampp/htdocs/myimages/two{p_count}.jpg"}, index=[p_count])], ignore_index=True)
                        # 이미지 저장
                        save = f'C:/xampp/htdocs/myimages/two{p_count}.jpg'
                        cv2.imwrite(save, im0)
                        image_url = f"http://localhost/myimages/two{p_count}.jpg"
                        
                        # text = "전우조 안지킴"
                        # # gTTS를 사용하여 텍스트를 한국어로 음성으로 변환
                        # tts = gTTS(text, lang='ko')  # 'ko'는 한국어를 나타냅니다. 원하는 언어 코드로 변경할 수 있습니다.     
                        # # 음성 출력
                        # tts.play()

                        # 데이터베이스 삽입
                        query = "INSERT INTO precaution (`case`, casetime, casetype, caseplace, caseimage) VALUES (%s, %s, %s, %s, %s)"
                        values = (f'2인1조 미준수 {p_count}', current_time, '2인1조 미준수', 'sector3', image_url)
                        cur.execute(query, values)
                        conn.commit()

                    vid_writer.write(im0)

    print(f'\n\n2초동안 안전모 미착용 감지 개수 : {h_count}')
    print(f'2초동안 쓰러져있는 감지 개수 : {l_count}')
    print(f'2초동안 혼자있는 경우 개수 : {p_count}')
    print('\n\n=================== 데이터 프레임 =================')
    print(df)
    conn.close()

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
