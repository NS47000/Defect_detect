@echo off
::A9Q
adb root
rem taskkill /fi "WindowTitle eq  A9Q_GA2_show"
rem taskkill /fi "WindowTitle eq 系統管理員:  showImage"
taskkill /f /fi "WindowTitle eq 系統管理員:  showImage"
taskkill /f /fi "WindowTitle eq showImage"
taskkill /f /fi "WindowTitle eq showImage"
taskkill /fi "WindowTitle eq setBacklight"
rem taskkill /fi "WindowTitle eq C:\Windoes\system32\cmd.exe"
exit
