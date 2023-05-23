@echo on
title [FA2-3] FPC Test

rem timeout /t 2
SET IMG=%1
start /min showImage.bat %IMG%
rem start setBacklight.bat
rem start A9Q_GA2_show_pattern_YN_0614.bat U %IMG%
rem start set_dxdy_default.bat
::timeout /t 10
::taskkill /fi "WindowTitle eq showImage"

