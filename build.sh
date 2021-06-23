#!/bin/bash

games="momentum p2ce"

build_dir="build"
bin_dir="bin/win64" # Hammer is windows only

# Setup hammer folder copy exclusions (*_momentum, *_p2ce, etc)
copy_exclusions=(--exclude="scripts")
for i in $games
  do
  copy_exclusions+=(--exclude="*_$i")
done

game=$1
# Make sure game isn't empty by asking the user for what game to build
if [ $# -eq 0 ]; then
  echo Games: "${games[*]}" && echo Enter game to build. Use ALL to build every game. && read -p "" game
fi

echo "Removing previous build in $build_dir"
rm -rf "$build_dir"

build_p2ce() {
  copy_hammer_files p2ce
  copy_vscript_files
  copy_postcompiler_files
  build_game_fgd p2ce
}

build_momentum() {
  copy_hammer_files momentum
  build_game_fgd momentum
}

copy_hammer_files() {
  echo "Copying Hammer files..."
  mkdir -p "$build_dir"
  rsync -a "${copy_exclusions[@]}" "hammer" "$build_dir"
  rsync -a hammer/cfg_$1/* $build_dir/hammer/cfg

  if [ $? -ne 0 ]; then
    echo "Failed copying Hammer files. Exitting." && exit 1
  fi
  return 0
}

build_game_fgd() {
  echo "Building FGD for $1..."
  mkdir -p "$build_dir/$1"
  python3 unify_fgd.py exp $1 srctools -o "$build_dir/$1/$1.fgd"

  if [ $? -ne 0 ]; then
    echo "Building FGD for $1 has failed. Exitting." && exit 1
  fi
  return 0
}

copy_vscript_files() {
  echo "Copying VScript files (hammer/scripts)..."
  mkdir -p "$build_dir/hammer"
  cp -rf hammer/scripts "$build_dir/hammer"

  if [ $? -ne 0 ]; then
    echo "Failed copying VScript files (hammer/scripts). Exitting." && exit 1
  fi
  return 0
}

copy_postcompiler_files() {
  echo "Copying postcompiler transforms..."
  mkdir -p "$build_dir/$bin_dir/postcompiler"
  cp -rf transforms "$build_dir/$bin_dir/postcompiler"

  if [ $? -ne 0 ]; then
    echo "Failed copying postcompiler transforms. Exitting." && exit 1
  fi
  return 0
}

if [ "${game^^}" = "ALL" ]; then
  # Modify build directory to not have directories clash
  main_build_dir=$build_dir
  for i in $games
    do
    unset build_dir
    build_dir=$main_build_dir/$i
    build_$i
  done
else
  for i in $games
    do
    if [ "$i" = "$game" ]; then
      build_$game
      exit
    fi
  done
  echo "Unknown game. Exitting." && exit 1
fi
