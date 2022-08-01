rem Script used to transfer across postcompiler code from the srctools repo.
cd ../
rmdir /S /Q postcompiler/
git clone --no-local "srctools/.git" "postcompiler"
cd postcompiler
git filter-repo --paths-from-file HammerAddons/srctools-files.txt --commit-callback "commit.message += b'\nTransferred from TeamSpen210/srctools @ ' + commit.original_id"
