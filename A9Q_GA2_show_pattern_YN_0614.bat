@echo on
title A9Q_GA2_show
rem timeout /t 2

pushd .
cd /d %~dp0

SET SWITCH=%1
SET IMG=%2
SET STAGE=%3 rem MTF or IQT 

SET ADB=adb
SET IM_PATH=.\Resoure_data\A9Q_RAW\
SET SD_PATH=/sdcard/Pictures/Display_Image/
%ADB% root
rem %ADB% remount
rem %ADB% shell mount -o remount,rw /

if "%SWITCH%" == "p" SET q=T
if "%SWITCH%" == "U" SET q=T
if "%q%" == "T" (
	if "%IMG%" == "MTFH" (
		SET Temp=MTF_H_2px_1080_1440
	) ^
	else if "%IMG%" == "MTFV" (
		SET Temp=MTF_V_2px_1080_1440
	) ^
	else if "%IMG%" == "White" (
		SET Temp=White_1080_1440
	) ^
	else if "%IMG%" == "Black" (
		SET Temp=Black_1080_1440
	) ^
	else if "%IMG%" == "CBCW" (
		SET Temp=CB3X3B_1080_1440
	) ^
	else if "%IMG%" == "CBCB" (
		SET Temp=CB3X3A_1080_1440
	) ^
	else if "%IMG%" == "W50D" (
		SET Temp=white_point_image_4px_50gap_0value_1080_1440
	) ^
	else if "%IMG%" == "B50D" (
		SET Temp=black_point_image_4px_50gap_96value_1080_1440
	) ^
	else if "%IMG%" == "W30D" (
		SET Temp=white_point_image_4px_30gap_0value_1080_1440
	) ^
	else if "%IMG%" == "B30D" (
		SET Temp=black_point_image_4px_30gap_96value_1080_1440
	) ^
	else if "%IMG%" == "Cross" (
		SET Temp=Cross_1080_1440
	) ^
	else if "%IMG%" == "MTF3H" (
		SET Temp=MTF_H_3px_1080_1440
	) ^
	else if "%IMG%" == "MTF3V" (
		SET Temp=MTF_V_3px_1080_1440
	) ^
	else if "%IMG%" == "MTF4H" (
		SET Temp=MTF_H_4px_1080_1440
	) ^
	else if "%IMG%" == "MTF4V" (
		SET Temp=MTF_V_4px_1080_1440
	) ^

	
	echo %IMG%
	echo %Temp%
)


rem put image folder
if "%SWITCH%" == "p" (
	rem %ADB% push .\Resoure_data\Push_IM\%IMG% %SD_IMG_PATH%
	rem %ADB% push %IM_PATH%%Temp%\. /data/display/displayfiltertest/	
	
	rem A9Q
	%ADB% push %IM_PATH%%Temp% /data/display/displayfiltertest/
)

rem UI show image button function
if "%SWITCH%" == "U" (
	
	rem A9Q
	%ADB% shell /usr/bin/qc2displayfilterunittest --foldername %Temp% --format C8_LINEAR
	rem --sleep_sec 10
	
	rem A93
	rem %ADB% push %IM_PATH%\%Temp% %SD_PATH%
	rem %ADB% shell am start -d %SD_PATH%%Temp% -t image/png -a android.intent.action.VIEW
)

if "%SWITCH%" == "L" (
	start set-current_default.bat
)

rem colse image or go home, etc.
if "%SWITCH%" == "E" (
	start kill_all.bat
)

popd
echo on