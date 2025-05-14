@echo off
setlocal EnableDelayedExpansion

SET games=p2 p1 hl2 ep1 ep2 gmod csgo tf2 asw l4d l4d2 infra mesa dods

:: If set, override the FGD filename generated.
SET filename.p2=portal2
SET filename.p1=portal
SET filename.hl2=halflife2
SET filename.ep1=episodic
SET filename.gmod=garrysmod
SET filename.tf2=tf
SET filename.asw=swarm
SET filename.l4d=left4dead
SET filename.l4d2=left4dead2
SET filename.mesa=blackmesa
SET filename.dods=dod

:: Other games supported by tags, but not currently included in builds
SET filename.asrd=reactivedrop
SET filename.css=cstrike
SET filename.ez1=EntropyZero
SET filename.ez2=EntropyZero2
SET filename.hl2dm=hl2mp
SET filename.kz=kreedzclimbing
:: No official name for this, but don't want it to just be "psa.fgd"
SET filename.psa=portal_stillalive
:: No official name for these either
SET filename.p2sixense=portal2_sixense
SET filename.p2edu=portal2_edu
:: No official Half-Life: Source FGD exists; using game dir name
SET filename.hls=hl1
SET filename.hldms=hl1mp

SET game=%1

:: Make sure game isn't empty
:while
IF [%game%]==[] (echo Games: %games% & echo Enter game to build. Use ALL to build every game. & SET /P game= & GOTO :while)

IF /I %game%==ALL (
  CALL :copy_hammer_files
  (FOR %%i in (%games%) do (
    CALL :build_game %%i
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
  SET tag=%1
  IF DEFINED filename.%tag% (SET fname=!filename.%tag%!) ELSE (SET fname=%tag%)
  echo Building FGD for %1 as "%fname%.fgd"...
  python src/hammeraddons/unify_fgd.py exp "%tag%" srctools -o "build/%fname%.fgd"
  IF %ERRORLEVEL% NEQ 0 (echo Building FGD for %tag% has failed. Exitting. & EXIT)
  EXIT /B

:copy_hammer_files
  echo Copying Hammer files...
  IF %ERRORLEVEL% LSS 8 robocopy hammer build/hammer /S /PURGE
  IF %ERRORLEVEL% LSS 8 robocopy instances build/instances /XF *.vmx /S /PURGE
  IF %ERRORLEVEL% LSS 8 EXIT /B 0
  echo Failed copying Hammer files. Exitting. & EXIT
