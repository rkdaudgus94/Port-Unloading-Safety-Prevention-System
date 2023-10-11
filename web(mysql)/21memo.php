<?php
    include "21Lib.php";
?>

<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
    @font-face {
            font-family: 'COOPBL';
            src: url('C:/Windows/Fonts/COOPBL.TTF') format('truetype');
    }
        table {
    width: 80%;
    margin: 20px auto;
    border-collapse: collapse;
}
body {
    font-family: COOPBL, sans-serif;
    background-color: #2A2A2A;  
    color: #fff;             
}

table, th, td {
    border: 1px solid black;
    color: #fff;             /* 글자색을 흰색으로 설정 */
    background-color: #333;  /* 배경색을 검은색으로 설정 */
}


th, td {
    padding: 10px;
    text-align: center;
}

nav a {
    color: #fff;
    padding: 5px 10px;
    margin: 0 5px;
    background-color: #007BFF;
    color: white;
    text-decoration: none;
    border-radius: 3px;
}

nav a:hover {
    background-color: #2A2A2A;
}

.top-bar {
            background-color: #2A2A2A; /* 검은색 배경 */
            padding: 10px 20px;     /* 바 내부의 패딩 */
            text-align: center;       
        }

h1.title {
            color: #888; /* 얕은 회색 빛 */
            font-size: 45px;
            margin-left: 20px; /* 왼쪽 간격 설정 */
            margin-top: 30px; /* 상단 간격 설정 */
            font-family: 'COOPBL';
        }

.table-container {
    border-radius: 15px; /* 굴곡의 크기 */
    overflow: hidden; /* 굴곡 부분 밖의 내용을 숨김 */
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2); /* 선택적: 그림자 효과 추가 */
    background-color: white; /* 배경색 설정 */
    margin: 20px auto; /* 중앙 정렬 */
}
.grid-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 50px; /* 사각형 간의 간격 */
    margin-top: 50px;
    margin-bottom: 60px;
}

.box {
    width: calc(25% - 10px); /* 전체 너비의 50%에서 간격의 반을 뺌 */
    height: 300px; /* 높이 설정 */
    background-color: #f5f5f5; /* 배경색 설정 */
    border-radius: 20px; /* 굴곡 설정 */
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2); /* 그림자 효과 */
    flex-basis: calc(40% - 400px); /* flexbox의 기본 너비 설정 */
    text-align: center; /* 내용 중앙 정렬 */
    line-height: 100px; /* 수직 중앙 정렬 */
    font-family: 'HMKMRHD', sans-serif; 
}


 .box{
    background-color: #333;  /* 배경색을 검은색으로 설정 */
    color: #fff;          /* 글자색을 흰색으로 설정 */
    font-family: 'HMKMRHD', sans-serif; 
}
.table-container, .grid-container{
    background-color: #2A2A2A;  /* 배경색을 검은색으로 설정 */
    color: #fff; 
}
    </style>
</head>
<body>

<div class="top-bar">
    <h1 class="title">안전사고기록</h1>
</div>
<?php
// 1. 현재 페이지 번호 받기
$page = isset($_GET['page']) ? (int)$_GET['page'] : 1;

// 2. 한 페이지에 표시할 데이터 수
$items_per_page = 10;

// 3. 총 데이터 수 계산
$total_query = "SELECT COUNT(*) as total FROM precaution";
$total_result = mysqli_query($connect, $total_query);
$total_row = mysqli_fetch_assoc($total_result);
$total_items = $total_row['total'];
$total_pages = ceil($total_items / $items_per_page);

// 4. LIMIT 구문 사용
$offset = ($page - 1) * $items_per_page;
$query = "SELECT * FROM precaution LIMIT $offset, $items_per_page";
$result = mysqli_query($connect, $query);
?>


<div class="grid-container">
    <div class="box">위반사항 1위</div>
    <div class="box">사고율 1위 장소</div>
    <div class="box">사건 사고 발생 수</div>
</div>

<div class="table-container">
    <table border="1">
    <tr>
        <td style="width:150px; text-align:center;"> Type </td>
        <td style="width:150px; text-align:center;"> 사건일시 </td>
        <td style="width:150px; text-align:center;"> 사건종류 </td>
        <td style="width:150px; text-align:center;"> 사건장소 </td>
        <td style="width:150px; text-align:center;"> 사건이미지 </td>
    </tr>

    <?php
    while($data = mysqli_fetch_array($result)){
    ?>
    <tr>
        <td style="width:150px; text-align:center;"><?=$data['case']?></td>
        <td style="width:150px; text-align:center;"><?=$data['casetime']?></td>
        <td style="width:150px; text-align:center;"><?=$data['casetype']?></td>
        <td style="width:150px; text-align:center;"><?=$data['caseplace']?></td>       
        <td style="width:150px; text-align:center;"><a href="<?=$data['caseimage']?>" target="_blank">사건 이미지 보기</a></td>
    </tr>
    <?php
    }
    ?>
    </table>
</div>
<!-- 5. 페이지 링크 생성 -->
<nav>
    <?php
    for ($i = 1; $i <= $total_pages; $i++) {
        echo "<a href='?page=$i'>$i</a> ";
    }
    ?>
</nav>

</body>
</html>
