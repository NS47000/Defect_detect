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


rem put image folder
if "%SWITCH%" == "p" (
	rem %ADB% push .\Resoure_data\Push_IM\%IMG% %SD_IMG_PATH%
	rem %ADB% push %IM_PATH%%Temp%\. /data/display/displayfiltertest/	
	
	rem A9Q
	%ADB% push %IM_PATH%%Temp% /data/display/displayfiltertest/
)