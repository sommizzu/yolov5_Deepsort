# 필요한 라이브러리들을 임포트합니다.
import argparse
import os
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# CPU 사용량을 제한합니다.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 파일의 절대 경로를 찾습니다.
FILE = Path(__file__).resolve()

# 파일의 상위 디렉토리를 찾습니다. 이는 코드의 루트 디렉토리입니다.
ROOT = FILE.parents[0]  # yolov5 strongsort root directory

# 가중치가 저장된 디렉토리를 설정합니다.
WEIGHTS = ROOT / 'weights'

# sys.path에 코드의 루트 디렉토리와 yolov5, strong_sort 디렉토리를 추가합니다.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH

# 현재 작업 디렉토리를 기준으로 상대 경로로 ROOT를 설정합니다.
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 필요한 라이브러리와 모듈들을 임포트합니다.
import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from collections import deque,Counter
from add_reid import *

# 로깅 핸들러를 제거하여 중복 로깅을 방지합니다.
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

# 데코레이터 torch.no_grad()는 이 함수 내부에서의 연산들이 backpropagation을 위한 연산 기록을 남기지 않도록 합니다.
# 이는 추론시에 메모리 사용량을 줄이고 속도를 향상시킵니다.
# 객체 추적 실행 함수 정의
@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # YOLO 모델의 가중치
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # StrongSORT 모델의 가중치
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',  # StrongSORT 설정 파일 경로
        imgsz=(640, 640),  # 추론 이미지 크기 (높이, 너비)
        conf_thres=0.25,  # 신뢰도 임계값
        iou_thres=0.45,  # NMS IOU 임계값
        max_det=1000,  # 이미지당 최대 검출 수
        device='',  # CUDA 장치, 예: 0 또는 0,1,2,3 또는 CPU
        show_vid=False,  # 결과 표시 여부
        save_txt=False,  # 결과를 *.txt로 저장할지 여부
        save_conf=False,  # --save-txt 레이블에 신뢰도를 저장할지 여부
        save_crop=False,  # 예측된 박스를 자를지 여부
        save_vid=False,  # --save-txt 레이블에 신뢰도를 저장할지 여부
        nosave=False,  # 이미지/비디오를 저장하지 않을지 여부
        classes=None,  # 클래스로 필터링: --class 0 또는 --class 0 2 3
        agnostic_nms=False,  # 클래스에 구애받지 않는 NMS
        augment=False,  # 증강 추론
        visualize=False,  # 기능 시각화
        update=False,  # 모든 모델 업데이트
        project=ROOT / 'runs/track',  # 결과를 project/name에 저장
        name='exp',  # 결과를 project/name에 저장
        exist_ok=False,  # 기존 project/name이 있어도 괜찮음, 증가하지 않음
        line_thickness=3,  # 바운딩 박스 두께 (픽셀)
        hide_labels=False,  # 레이블 숨기기
        hide_conf=False,  # 신뢰도 숨기기
        hide_class=False,  # ID 숨기기
        half=False,  # FP16 반 정밀도 추론 사용
        dnn=False,  # ONNX 추론에 OpenCV DNN 사용
):

    source = str(source)  # 소스 파일의 경로를 문자열로 변환
    save_img = not nosave and not source.endswith('.txt')  # 저장할 이미지가 있는지 확인
    is_file = Path(source).suffix[1:] in (VID_FORMATS)  # 입력 소스가 파일인지 확인
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # 입력 소스가 URL인지 확인
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 웹캠을 사용할지 확인
    if is_url and is_file:
        source = check_file(source)  # 다운로드할 파일이 있는지 확인

    # Directories
    # 단일 YOLO 모델 사용 시
    if not isinstance(yolo_weights, list):  
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    # --yolo_weights 이후 단일 모델 사용 시
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  
        exp_name = yolo_weights[0].split(".")[0]
    # --yolo_weights 이후 여러 모델 사용 시
    else:  
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 결과를 저장할 디렉토리 생성

    # Load model
    device = select_device(device)  # 사용할 장치 선택 (CPU 또는 GPU)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)  # 모델 로드
    stride, names, pt = model.stride, model.names, model.pt  # 모델에서 필요한 정보 추출
    imgsz = check_img_size(imgsz, s=stride)  # 이미지 크기 확인

    # Dataloader
    # 웹캠이나 비디오 스트림 사용 시
    if webcam:  
        show_vid = check_imshow()
        cudnn.benchmark = True  # 동일한 이미지 크기 추론을 빠르게 하기 위해 설정
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:  # 단일 이미지나 비디오 파일 사용 시
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # 각 비디오 소스마다 StrongSORT 인스턴스 생성
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # 모델 웜업
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

# 각각의 객체(트랙)에 대한 경로를 추적하는 dictionary입니다. 키는 객체의 id이고, 값은 deque입니다.
paths = {}

# 추적 중인 객체의 클래스를 저장합니다. 이 경우 사람을 추적하므로 항상 0입니다.
track_cls = 0

# 마지막으로 추적된 객체의 ID를 저장합니다.
last_track_id = -1

# 현재 프레임의 인덱스
idx_frame = 0

# 결과를 저장하는 리스트
results = []

# 이미 계산된 객체 ID를 저장하는 데크 (최대 길이 50)
already_counted = deque(maxlen=50)  

# 이미 계산된 객체 ID를 저장하는 데크 (최대 길이 50)
already_counted2 = deque(maxlen=50)  

# 총 트랙 수
total_track2 = 0

# 감지된 각 클래스의 수를 저장하는 카운터
class_counter = Counter()  

# 총 감지 수
total_counter = 0

# 위로 이동한 객체의 수
up_count = 0

# 아래로 이동한 객체의 수
down_count = 0

# 도착한 사람을 저장하는 딕셔너리
arrived_person = {}

# 새로운 ID 딕셔너리와 인덱스
new_id_dict, new_id_index = dict(), 0

# 데이터셋의 각 프레임에 대해 반복
for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
    # 동기화된 시간을 가져옵니다.
    t1 = time_sync()

    # 이미지를 토치 텐서로 변환하고 GPU로 보냅니다.
    im = torch.from_numpy(im).to(device)
    
    # 이미지를 fp16 또는 fp32로 변환합니다.
    im = im.half() if half else im.float()  
    
    # 0 - 255의 값을 0.0 - 1.0으로 스케일링합니다.
    im /= 255.0  
    
    # 배치 차원을 확장합니다.
    if len(im.shape) == 3:
        im = im[None]  
    
    t2 = time_sync()
    dt[0] += t2 - t1

    # 추론을 수행합니다.
    visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
    pred = model(im, augment=opt.augment, visualize=visualize)
    
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS를 적용합니다.
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
    dt[2] += time_sync() - t3

# 각각의 감지에 대해 처리를 수행합니다.
for i, det in enumerate(pred):  # 각 이미지에 대한 감지
    seen += 1
    if webcam:  # 웹캠 사용 시
        p, im0, _ = path[i], im0s[i].copy(), dataset.count
        p = Path(p)  # 경로 객체로 변환
        s += f'{i}: '
        txt_file_name = p.name
        save_path = str(save_dir / p.name)  # 이미지 또는 비디오 파일의 저장 경로
    else:
        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # 경로 객체로 변환
        # 비디오 파일의 경우
        if source.endswith(VID_FORMATS):
            txt_file_name = p.stem
            save_path = str(save_dir / p.name)  # 이미지 또는 비디오 파일의 저장 경로
        # 이미지 폴더의 경우
        else:
            txt_file_name = p.parent.name  # 현재 이미지를 포함하고 있는 폴더의 이름
            save_path = str(save_dir / p.parent.name)  # 이미지 또는 비디오 파일의 저장 경로
    curr_frames[i] = im0

    txt_path = str(save_dir / 'tracks' / txt_file_name)  # 텍스트 파일의 저장 경로
    s += '%gx%g ' % im.shape[2:]  # 출력 문자열
    imc = im0.copy() if save_crop else im0  # 이미지 복사 (save_crop 옵션이 True인 경우)

    # Annotator 객체 생성
    annotator = Annotator(im0, line_width=2, pil=not ascii)
    
    if cfg.STRONGSORT.ECC:  # 카메라 움직임 보상
        strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

    # 이미지의 중앙에 위치한 가상의 경계선
    line = [(0, int(0.55 * im0.shape[0])), (int(im0.shape[1]), int(0.55 * im0.shape[0]))]

    # 가상의 경계선을 이미지에 그립니다.
    cv2.line(im0, line[0], line[1], (0, 255, 255), 4)
    annotator = Annotator(im0, line_width=2, pil=not ascii)

    if det is not None and len(det):
        # Bounding box를 이미지의 크기에 맞게 재조정합니다.
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # 결과를 출력합니다.
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # 각 클래스에 대한 감지 수
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 문자열에 추가

        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]

        # 감지 결과를 strongsort에 전달합니다.
        t4 = time_sync()
        outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
        t5 = time_sync()
        dt[3] += t5 - t4

        # 통과한 사람의 수를 계산합니다.
        for track in outputs[0]:
            bbox = track[:4]
            track_id = track[4]
            midpoint = tlbr_bottom(bbox)
            origin_midpoint = (
                midpoint[0], im0.shape[0] - midpoint[1])  # get midpoint respective to botton-left
            cv2.circle(im0, midpoint, 10, (0, 0, 255), -1)

            if track_id not in paths:
                paths[track_id] = deque(maxlen=2)
                total_track = track_id

            paths[track_id].append(midpoint)
            previous_midpoint = paths[track_id][0]
            origin_previous_midpoint = (previous_midpoint[0], im0.shape[0] - previous_midpoint[1])
            if intersect(midpoint, previous_midpoint, line[0], line[1]) and track_id not in already_counted:
                class_counter[track_cls] += 1
                total_counter += 1
                last_track_id = track_id
                total_track2 += 1
                cv2.line(im0, line[0], line[1], (0, 0, 255), 10)

                already_counted.append(track_id)  # Set already counted for ID to true.

                angle = vector_angle(origin_midpoint, origin_previous_midpoint)
                if angle > 0:
                    up_count += 1
                if angle < 0:
                    down_count += 1

        # 감지 결과를 시각화하기 위해 bounding box를 그립니다.
        if len(outputs[i]) > 0:
            for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                if save_txt:
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                       bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                if save_vid or save_crop or show_vid:  # Add bbox to image
                    c = int(cls)  # integer class
                    id = int(id)  # integer id
                    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                        (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                    annotator.box_label(bboxes, label, color=colors(c, True))
                    if save_crop:
                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                        save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

        LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

    else:
        strongsort_list[i].increment_ages()
        LOGGER.info('No detections')

    # 결과를 스트리밍합니다.
    im0 = annotator.result()

    # "통과한 사람의 수: 총합 (위로 이동: 수, 아래로 이동: 수)" 라벨을 추가합니다.
    label = "통과한 사람의 수: {} (위로 이동: {}, 아래로 이동: {})".format(str(total_counter), str(up_count), str(down_count))
    t_size = get_size_with_pil(label, 25)
    x1 = 20
    y1 = 100
    im0 = put_text_to_cv2_img_with_pil(im0, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))

    # 마지막으로 통과한 사람에 대한 정보를 추가합니다.
    if last_track_id >= 0:
        label = "사람 {}번 {} 방향으로 통과".format(str(last_track_id), str("위로") if angle >= 0 else str('아래로'))
        t_size = get_size_with_pil(label, 25)
        x1 = 20
        y1 = 150
        im0 = put_text_to_cv2_img_with_pil(im0, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))

    # 비디오를 보여줍니다.
    if show_vid:
        cv2.putText(im0, f"{n} {names[int(c)]}{'s' * (n > 1)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 255), 2)
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond

    # 감지된 이미지를 저장합니다.
    if save_vid:
        if vid_path[i] != save_path:  # 새 비디오
            vid_path[i] = save_path
            if isinstance(vid_writer[i], cv2.VideoWriter):
                vid_writer[i].release()  # 이전 비디오 writer를 해제합니다.
            if vid_cap:  # 비디오
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # 스트림
                fps, w, h = 30, im0.shape[1], im0.shape[0]
            save_path = str(Path(save_path).with_suffix('.mp4'))  # 결과 비디오에 강제로 *.mp4 확장자를 사용합니다.
            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer[i].write(im0)

    prev_frames[i] = curr_frames[i]

# 결과를 출력합니다.
t = tuple(x / seen * 1E3 for x in dt)  # 이미지당 속도
LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
if save_txt or save_vid:
    s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
if update:
    strip_optimizer(yolo_weights)  # 모델 업데이트 (SourceChangeWarning 수정)

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # YOLO 모델 가중치 경로
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5s.pt', help='model.pt path(s)')
    
    # StrongSort 모델 가중치 경로
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pth')
    
    # StrongSort 설정 파일 경로
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    
    # 웹캠, 파일, 디렉토리, URL 또는 glob이 될 수 있는 입력 소스
    parser.add_argument('--source', type=str, default=r'D:\my_job\DATA\data/test.mp4', help='file/dir/URL/glob, 0 for webcam')
    
    # 추론 크기 (h, w)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    
    # 신뢰도 임계값
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    
    # NMS IoU 임계값
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    
    # 이미지당 최대 감지 수
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    
    # CUDA 장치
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # 추적 비디오 결과를 표시
    parser.add_argument('--show-vid',default=True, action='store_true', help='display tracking video results')
    
    # 결과를 *.txt로 저장
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    
    # --save-txt 레이블에 신뢰도 저장
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    
    # 예측 상자를 잘라내어 저장
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    
    # 비디오 추적 결과를 저장
    parser.add_argument('--save-vid',default=True, action='store_true', help='save video tracking results')
    
    # 이미지/비디오를 저장하지 않음
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    
    # 필터링할 클래스
    parser.add_argument('--classes', default='0',nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    
    # 클래스에 구애받지 않는 NMS를 사용
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    
    # 확장된 추론을 사용
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    # 특징을 시각화
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    
    # 모든 모델을 업데이트
    parser.add_argument('--update', action='store_true', help='update all models')
    
    # 결과를 저장할 경로
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    
    # 결과를 저장할 경로의 이름
    parser.add_argument('--name', default='exp', help='save results to project/name')
    
    # 기존 프로젝트/이름을 허용하고 증가시키지 않음
    parser.add_argument('--exist-ok',default=True, action='store_true', help='existing project/name ok, do not increment')
    
    # 바운딩 박스의 두께 (픽셀)
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    
    # 레이블 숨김
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')

    # 신뢰도 숨김
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')

    # ID 숨김
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')

    # FP16 half-precision 추론 사용
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    # ONNX 추론을 위해 OpenCV DNN 사용
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # 명령줄 인수 파싱
    opt = parser.parse_args()

    # imgsz 값을 늘림 (len(opt.imgsz) == 1 인 경우 2배로, 그 외의 경우 1배로)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
return opt

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


