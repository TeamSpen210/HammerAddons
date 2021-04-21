@echo off

SET games=momentum p2ce
SET game=%1

:: Make sure game isn't empty
:while
IF [%game%]==[] (echo Games: %games% & echo Enter game to build. Use ALL to build every game. & SET /P game= & GOTO :while)

IF /I %game%==ALL (
  CALL :copy_hammer_files
  (FOR %%i in (%games%) do (
    CALL :build_game "%%i"
  ))
  EXIT
) ELSE (
  (FOR %%i in (%games%) do (
    IF /I %game%==%%i (
      CALL :copy_hammer_files
      CALL :build_game %game%
      EXIT
    )
  ))
  echo Unknown game. Exitting. & EXIT /B 1
)

:build_game
  echo Building FGD for %1...
  py unify_fgd.py exp %1 srctools -o "build/%1.fgd"
  IF %ERRORLEVEL% NEQ 0 (echo Building FGD for %1 has failed. Exitting. & EXIT)
  EXIT /B

:copy_hammer_files
  echo Copying Hammer files...
  IF %ERRORLEVEL% LSS 8 robocopy hammer build/hammer /S /PURGE
  IF %ERRORLEVEL% LSS 8 robocopy instances build/instances /XF *.vmx /S /PURGE
  IF %ERRORLEVEL% LSS 8 robocopy transforms build/postcompiler/transforms /PURGE
  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying Hammer files. Exitting. & EXIT
