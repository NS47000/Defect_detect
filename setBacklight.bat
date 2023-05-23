@echo on
title setBacklight
::timeout /t 10

cd /d %~dp0
SET ADB=adb
SET Lr=%1
SET Lg=%2
SET Lb=%3
SET Rr=%4
SET Rg=%5
SET Rb=%6

%ADB% root
rem %ADB% shell "echo 240 > /sys/bus/i2c/devices/0-0028/red_led"
rem %ADB% shell "echo 220 > /sys/bus/i2c/devices/0-0028/green_led"
rem %ADB% shell "echo 100 > /sys/bus/i2c/devices/0-0028/blue_led"
rem %ADB% shell "echo 120 > /sys/bus/i2c/devices/0-0029/red_led"
rem %ADB% shell "echo 110 > /sys/bus/i2c/devices/0-0029/green_led"
rem %ADB% shell "echo 50 > /sys/bus/i2c/devices/0-0029/blue_led"
%ADB% shell "echo %Lr% > /sys/bus/i2c/devices/0-0028/red_led"
%ADB% shell "echo %Lg% > /sys/bus/i2c/devices/0-0028/green_led"
%ADB% shell "echo %Lb% > /sys/bus/i2c/devices/0-0028/blue_led"
%ADB% shell "echo %Rr% > /sys/bus/i2c/devices/0-0029/red_led"
%ADB% shell "echo %Rg% > /sys/bus/i2c/devices/0-0029/green_led"
%ADB% shell "echo %Rb% > /sys/bus/i2c/devices/0-0029/blue_led"
%ADB% shell sync

