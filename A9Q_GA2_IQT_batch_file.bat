@echo off

pushd .
cd /d %~dp0

SET SWITCH=%1
SET SN=%2
SET TIME=%3
SET LR=%4
SET BAT=A9Q_GA2_show_pattern_YN_0614.bat

rem IQT push image folder(A9Q)
if "%SWITCH%" == "P" ( 
	rem color image
	call %BAT% p White
	call %BAT% p Black
	rem CBC image
	call %BAT% p CBCB
	call %BAT% p CBCW
	rem dummy image
	call %BAT% p B50D 
	call %BAT% p B30D
	call %BAT% p W50D
	call %BAT% p W30D
	rem MTF image
	call %BAT% p MTFH
	call %BAT% p MTFV
	rem Cross image
	call %BAT% p Cross
	
	rem call %BAT% p MTF3H
	rem call %BAT% p MTF3V
	rem call %BAT% p MTF4H
	rem call %BAT% p MTF4V


)

popd
echo on