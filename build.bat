@echo off
SETLOCAL enabledelayedexpansion

SET games=momentum p2ce

SET "build_dir=build"
SET bin_dir=bin/win64

:: Setup hammer folder copy exclusions (*_momentum, *_p2ce, etc)
SET "robocopy_exclusions=scripts "
FOR %%i in (%games%) do (
  CALL SET "game_exclude=*_%%i "
  SET "robocopy_exclusions=!robocopy_exclusions!!game_exclude!"
)

SET game=%1
:: Make sure game isn't empty by asking the user for what game to build
:while
IF [%game%]==[] (echo Games: %games% & echo Enter game to build. Use ALL to build every game. & SET /P game= & GOTO :while)

echo Removing previous build in ./%build_dir%/
rmdir /S /Q "%build_dir%"

IF /I %game%==ALL (
  :: Modify build directory to not have directories clash
  CALL SET "main_build_dir=!build_dir!"
  FOR %%i in (%games%) do (
    CALL SET "game_build_dir=!main_build_dir!"
    SET "build_dir=!game_build_dir!/%%i"
    CALL :build_%%i
  )
  EXIT /B
) ELSE (
  FOR %%i in (%games%) do (
    IF /I %game%==%%i (
      CALL :build_%%i
      EXIT /B
    )
  )
  echo Unknown game. Exitting. & EXIT /B 1
)

:build_p2ce
  CALL :copy_hammer_files p2ce
  CALL :copy_vscript_files
  CALL :copy_postcompiler_files
  CALL :build_game_fgd p2ce
  EXIT /B

:build_momentum
  CALL :copy_hammer_files momentum
  CALL :build_game_fgd momentum
  EXIT /B

:build_game_fgd
  echo Building FGD for %1...
  mkdir "%build_dir%/%1"
  python unify_fgd.py exp %1 srctools -o "%build_dir%/%1/%1.fgd"

  IF %ERRORLEVEL% NEQ 0 (echo Building FGD for %1 has failed. Exitting. & EXIT /B)
  EXIT /B

:copy_hammer_files
  echo Copying Hammer files...
  robocopy hammer %build_dir%/hammer /S /PURGE /XD %robocopy_exclusions%
  IF %ERRORLEVEL% LSS 8 robocopy hammer/cfg_%1 %build_dir%/hammer/cfg /S

  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying Hammer files for %1. Exitting. & EXIT /B

:copy_vscript_files
  echo Copying VScript files (hammer/scripts)...
  robocopy hammer/scripts %build_dir%/hammer/scripts /S /PURGE

  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying VScript files (hammer/scripts). Exitting. & EXIT /B

:copy_postcompiler_files
  echo Copying postcompiler transforms...
  robocopy transforms %build_dir%/%bin_dir%/postcompiler/transforms /S /PURGE
  
  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying postcompiler transforms. Exitting. & EXIT /B
