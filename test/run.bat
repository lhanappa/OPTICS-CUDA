 for /L %%i in (50, 50, 300) do for /L %%n in (1,1,6) do ..\optics_test.exe 0.1 6 %%i %%n > time_%%i_%%n.txt
pause