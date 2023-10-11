<?php


        error_reporting(E_ALL);
        ini_set("display_errors",1);

        $connect = mysqli_connect("127.0.0.1","root","1234","harbor");

        if(mysqli_connect_error()){
            echo "mysql 접속중 오류가 발생했습니다.";
            echo mysqli_connect_error();
        }
?>