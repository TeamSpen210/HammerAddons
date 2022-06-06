@echo off
SETLOCAL enabledelayedexpansion

SET games=momentum p2ce
SET modes=fgd md

SET "build_dir=build"
SET "build_md_dir=build_md"
SET bin_dir=bin/win64

:: Setup hammer folder copy exclusions (*_momentum, *_p2ce, etc)
SET "robocopy_exclusions=scripts "
FOR %%i in (%games%) do (
  CALL SET "game_exclude=*_%%i "
  SET "robocopy_exclusions=!robocopy_exclusions!!game_exclude!"
)

SET mode=%1
:: Make sure mode isn't empty by asking the user
:while_mode
IF [%mode%]==[] (echo Modes: %modes% & echo Enter mode. Use ALL to build everything. & SET /P mode= & GOTO :while_mode)

SET game=%2
:: Make sure game isn't empty by asking the user for what game to build
:while_game
IF [%game%]==[] (echo Games: %games% & echo Enter game to build. Use ALL to build every game. & SET /P game= & GOTO :while_game)

IF /I %mode%==ALL (
  CALL :build_fgd_cleanup
  CALL :build_md_cleanup
  GOTO :main_build_step
)
IF /I %mode%==FGD (
  CALL :build_fgd_cleanup
  GOTO :main_build_step
)
IF /I %mode%==MD (
  CALL :build_md_cleanup
  GOTO :main_build_step
)

:main_build_step
IF /I %game%==ALL (
  :: Modify build directory to not have directories clash
  CALL SET "main_build_dir=!build_dir!"
  FOR %%i in (%games%) do (
    CALL SET "game_build_dir=!main_build_dir!"
    SET "build_dir=!game_build_dir!/%%i"
    CALL :build %%i
  )
  EXIT
) ELSE (
  FOR %%i in (%games%) do (
    IF /I %game%==%%i (
      CALL :build %%i
      EXIT
    )
  )
  echo Unknown game. Exitting. & EXIT /B 1
)

:build_fgd_cleanup
  echo Removing previous FGD build in ./%build_dir%/
  rmdir /S /Q "%build_dir%"
  EXIT /B

:build_md_cleanup
  echo Removing previous markdown build in ./%build_md_dir%/
  rmdir /S /Q "%build_md_dir%"
  EXIT /B

:build
  IF /I %mode%==ALL (
    CALL :build_fgd_%%1
    CALL :build_game_markdown %1
    EXIT /B
  )
  IF /I %mode%==FGD (
    CALL :build_fgd_%%1
    EXIT /B
  )
  IF /I %mode%==MD (
    CALL :build_game_markdown %1
  )
  EXIT /B

:build_fgd_p2ce
  CALL :copy_hammer_files p2ce
  CALL :copy_vscript_files
  CALL :copy_postcompiler_files
  CALL :build_game_fgd p2ce
  EXIT /B

:build_fgd_momentum
  CALL :copy_hammer_files momentum
  CALL :build_game_fgd momentum
  EXIT /B

:build_game_fgd
  echo Building FGD for %1...
  mkdir "%build_dir%/%1"
  python unify_fgd.py exp %1 srctools -o "%build_dir%/%1/%1.fgd"

  IF %ERRORLEVEL% NEQ 0 (echo Building FGD for %1 has failed. Exitting. & EXIT)
  EXIT /B

:build_game_markdown
  echo Generating markdown from FGD for %1...
  mkdir "%build_md_dir%/%1"
  python unify_fgd.py expmd %1 srctools -o "%build_md_dir%/%1"
  
  IF %ERRORLEVEL% NEQ 0 (echo Building markdown for %1 has failed. Exitting. & EXIT)
  EXIT /B

:copy_hammer_files
  echo Copying Hammer files...
  robocopy hammer %build_dir%/hammer /S /PURGE /XD %robocopy_exclusions%
  IF %ERRORLEVEL% LSS 8 robocopy hammer/cfg_%1 %build_dir%/hammer/cfg /S

  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying Hammer files for %1. Exitting. & EXIT

:copy_vscript_files
  echo Copying VScript files (hammer/scripts)...
  robocopy hammer/scripts %build_dir%/hammer/scripts /S /PURGE

  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying VScript files (hammer/scripts). Exitting. & EXIT

:copy_postcompiler_files
  echo Copying postcompiler transforms...
  robocopy transforms %build_dir%/%bin_dir%/postcompiler/transforms /S /PURGE
  
  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying postcompiler transforms. Exitting. & EXIT
