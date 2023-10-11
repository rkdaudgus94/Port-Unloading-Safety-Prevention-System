<div align = "center" >

![header](https://capsule-render.vercel.app/api?type=waving&&color=gradient&height=100&section=header&fontSize=100)
</div>

<div align="center">
    <h1>  ⛵Port-Unloading-Safety-Prevention-System⛵ </h1>
</div>
<div align ="left">
    
<details> 
    
<summary><h2>🗺️ 포스터</summary>
<img width="906" alt="image" src="https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/5977c213-e623-4b2c-9eba-895f5ab1189a">

</details>
</div>

<div align ="left">

## 🚩 목적
- 안전부주의 예방 
 크레인 붕괴, 컨테이너 깔림과 같은 장비의 문제 뿐만 아니라, 안전 부주의로 인한 헬멧 미착용 사고도 발생하고 있다는 사례들이 있었다. 
 이러한 사고들은 사람의 목숨과 직접적으로 맞닿아 있기 때문에 중대한 문제이지만 실상은 잘 지켜지지 않고 있다. 
 그렇기에 CCTV를 통해 헬멧이나 안전조끼 미착용, 2인1조 작업등을 잘 지키고 있는지 확인하여 사고를 예방하고자 한다.

- 사고발생시 즉시 대처
 사고가 발생한 경우 작업자들은 대부분 쓰러져 있는 상황이었다. 그러한 경우 cctv로 쓰러진 사람을 확인하고 신속히 알려 빠른 대처를 통해 피해를 최소화 하고자 한다.

## 💭 수행 내용
### 🥇 통계기반 머신러닝
 - 한국항만물류협회 사이트에 항만하역 재해통계 및 사례 자료가 2011년~2021년까지 있어서 이 자료를 바탕으로 근 5년간의 사고 발생 시간, 근속 연수, 사고 장소, 사고 유형에 대한 데이터셋을 제작했다.</br>
 - 시간, 장소, 유형 뿐만 아니라 근속 연수를 넣은 이유는 근속 연수가 짧은 초보자의 사고 발생 사례가 많긴 하지만 베테랑들도 익숙한 일에 방심하고 안전 부주의로 인해 사고 발생 사례가 많기 때문이다.</br>
 - 랜덤포레스트, XGBoost, LSBM, 의사결정나무를 통해 학습을 진행하였기에 우선 문자열 데이터를 원핫인코딩으로 전처리 해주었다. 타겟으로는 사고가 자주 발생하는 시간대나 장소로 정했고, 각각 학습을 돌렸을 때 정확도가 비슷했지만 그 중 가장 높은 XGBoost를 최종적으로 선택했다.</br>
</br>
</br>
 <div align ="center">
     
### 한국항만물류협회 사이트에 항만하역 재해통계 및 사례 자료(2011~2021)     
![image](https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/3ed94c05-e3c0-45aa-941d-e76190c82a32)</br>

 </div>
 <div align ="center">
     
### 사건/사고 장소 예측 정확도     
![image](https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/469c3a27-7023-4386-bf36-10a1fe54c925)
</br>
</br>
</br>
</div>

### 🥈 비전
- 안전모, 안전조끼, 안전고리, 쓰러진 자세를 학습시키고 그걸 바탕으로 안전모, 안전조끼 미착용 및 쓰러진 자세를 식별한다. 이것이 비전 처리 부분에서의 전체적인 큰 틀이다. 
 학습에 필요한 이미지 데이터는 AI 허브, 유튜브 동영상에서 발췌해서 LabelImg라는 툴을 이용해 라벨링 작업을 진행
 - 총 만 장 정도 라벨링 작업을 진행했는데 항만 하역의 특성상 대부분은 날이 밝거나 밤에 조명을 비춰 밝은 화면의 이미지이다. 하지만 날이 저물어가고 조명을 틀지 않은 경우 CCTV 화면 상 회색의 가까운 색을 표현하기에 회색으로 전처리를 한 이미지를 추가
 - CCTV를 통해 객체 탐지를 진행해야 하므로 영상 객체 인식을 하기 쉽고 결과가 좋은 yolo를 이용하기로 결정
 yolov5, yolov7, yolov8을 후보군으로 잡았는데, 최종적으로 선택한 버전은 yolov7이다.
 - 이 부분에 대해 자세히 설명하자면 yolov5는 학습 속도는 나쁘지 않았지만 정확도 부분에서 아쉬운 점이 보였다. 안전고리는 거의 잡지 못했고, 객체가 아닌 곳에서 탐지되는 경우가 상당히 많이 발생해서 우선 후보군에서 제외하기로 했다.
 - yolov8 같은 경우 학습 속도와 정확도가 상당히 괜찮은 편이었지만, 최종적인 목표를 이루기 위해 탐지된 클래스를 추출해야 하는 작업에서 난항을 겪었다.
 yolov5, yolov7과 달리 detect.py 파일이 존재하지 않고, 구조적으로 많은 부분이 바뀌어서 클래스 추출 뿐만 아니라 threshold 설정 등 세부 설정 부분에서 코드를 뜯어서 진행해야 했기에 시간이 부족하다고 판단하여 후보군에서 제외 
 - 최종적으로 선택한 yolov7은 학습 속도는 느린 편이지만 정확도 부분에서 괜찮았고, 탐지된 클래스 추출, threshold 설정 등 세부 설정도 하기 편해서 선택
 모델은 yolov7-x를 통해 진행했는데 기본 모델인 yolov7보다 정확도가 높고, yolov7-x 보다 높은 모델은 속도가 너무 느리고, 정확도 부분에서 크게 이득이 없어서 최종적으로 yolov7-x로 결정
 - threshold는 기본적으로 0.25로 설정 되어 있는데 객체가 아닌 곳에서 탐지되는 경우가 종종 있어서 0.3으로 조정하였고, 그 결과 객체 탐지가 괜찮아졌지만, 안전고리 부분을 잡기가 힘들어졌다.
 안전고리의 모양 특성상 동그라미 모양의 앞면을 라벨링 했지만, 옆면으로 틀어지는 순간 일자 모양이 되어버려서 학습시키는데 어려움이 있다고 판단해서 안전고리 부분은 추출할 때 빼기로 결정
 - 추출한 부분은 안전모, 안전조끼 미착용, 쓰러진 자세가 5초 동안 감지될 때 추출되게끔 코드로 만들었다. 사건 발생 시간, 사건 종류, 장소, 그리고 영상 속 장면을 이미지 파일로 만들어서 데이터베이스에 저장되게 진행

![image](https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/5c78f7b4-8de7-40e3-98d4-cac6f3c89318)

<div align = "center">

### 🔘yolo5 result
![image](https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/d658121c-c4b3-484f-959b-1e5a60b58d6c)
</br>
</br>
</br>
### 🔘yolo7 result
![image](https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/6e420cc9-98dd-4a3f-a1a6-8f2fc3b20e9f)
</br>
</br>
</br>
### 🔘yolo8 result
![image](https://github.com/rkdaudgus94/Port-safety-prevent-system/assets/76949032/1745ea6c-a63c-4475-864c-da3a03f6910a)

</div>
