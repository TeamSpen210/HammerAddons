#!/bin/sh
games="p2 p1 hl2 ep1 ep2 gmod csgo tf2 asw l4d l4d2 infra mesa"
game=$1
if [ $# -eq 0 ]; then
  echo Games: "${games[*]}" & echo Enter game to build. Use ALL to build every game. & read -p "" game
fi

copy_hammer_files() {
  echo "Copying Hammer files..."
  mkdir -p build/postcompiler &&
  cp -rf hammer build/hammer &&
  cp -rf instances build/instances &&
  cp -rf examples build/examples &&
  cp -rf transforms build/postcompiler/transforms &&
  find ./build/instances -iname "*.vmx" -delete # Yes, I know that we could use rsync with a ton of options to do this instead of using cp and then deleting unwanted files. This is FAR nicer imo.
  
  if [ $? -ne 0 ]; then
    echo "Failed copying Hammer files. Exitting." & exit 1
  fi
  return 0
}

build_game() {
  echo "Building FGD for $1..."
  python3 src/hammeraddons/unify_fgd.py exp $1 srctools -o "build/$1.fgd"
  
  if [ $? -ne 0 ]; then
    echo "Building FGD for $1 has failed. Exitting." & exit 1
  fi
  return 0
}

if [ "${game^^}" = "ALL" ]; then 
  copy_hammer_files
  for i in $games 
    do
    build_game $i
  done
else
  for i in $games
    do
    if [ "$i" = "$game" ]; then 
      copy_hammer_files
      build_game $game
      exit
    fi
    echo "Unknown game. Exitting." & exit 1
  done
fi
