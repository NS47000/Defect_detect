@echo on
title A9Q_GA2_show
rem timeout /t 2

pushd .
cd /d %~dp0

SET SWITCH=%1
SET IM_PATH=%2
SET IMG=%3
SET ADB=adb
SET SD_PATH=/sdcard/Pictures/Display_Image/
%ADB% root
rem %ADB% remount
rem %ADB% shell mount -o remount,rw /


rem put image folder
if "%SWITCH%" == "P" (
	rem =============Push Image=========================
	rem %ADB% %ADB% push %IM_PATH%%IMG% /data/display/displayfiltertest/	
	%ADB% push %IM_PATH%%IMG% /data/display/displayfiltertest/
)


if "%SWITCH%" == "U" (
	rem =========use image=================================
	rem %ADB% shell /usr/bin/qc2displayfilterunittest --foldername %IMG% --format C8_LINEAR
	%ADB% shell /usr/bin/qc2displayfilterunittest --foldername %IMG% --format C8_LINEAR
	rem --sleep_sec 10
	
)

if "%SWITCH%" == "L" (
	start set-current_default.bat
)


if "%SWITCH%" == "E" (
	rem colse image or go home, etc.
	start kill_all.bat
)

popd
echo on