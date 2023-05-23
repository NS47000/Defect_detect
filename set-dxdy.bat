@echo off

pushd .
cd /d %~dp0

SET /A L_DX=470+%1
SET /A L_DY=%2
SET /A R_DX=470-%3
SET /A R_DY=%4

SET /A ABS_L_DY=-%2
SET /A ABS_R_DY=-%4

SET ADB=adb

::%ADB% root
%ADB% shell "echo "%L_DX%" > /sys/kernel/rtimd/rtimd_eye/left_eye_shift_right"
if %L_DY% geq 0 (
    %ADB% shell "echo "%L_DY%" > /sys/kernel/rtimd/rtimd_eye/left_eye_shift_bottom"
) else (
	echo %L_DY%
	echo %ABS_L_DY%
    %ADB% shell "echo "%ABS_L_DY%" > /sys/kernel/rtimd/rtimd_eye/left_eye_shift_top"
)
%ADB% shell "echo "%R_DX%" > /sys/kernel/rtimd/rtimd_eye/right_eye_shift_left"
if %R_DY% geq 0 (
    %ADB% shell "echo "%R_DY%" > /sys/kernel/rtimd/rtimd_eye/right_eye_shift_bottom"
) else (
    %ADB% shell "echo "%ABS_R_DY%" > /sys/kernel/rtimd/rtimd_eye/right_eye_shift_top"
)

popd
echo on