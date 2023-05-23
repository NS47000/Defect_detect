@echo off
title showImage
timeout /t 2

cd /d %~dp0
SET ADB=adb

SET IMG=%1
%ADB% root
%ADB% shell /usr/bin/qc2displayfilterunittest --foldername %IMG% --format C8_LINEAR
rem start A9Q_GA2_show_pattern_YN_0614.bat U %IMG%