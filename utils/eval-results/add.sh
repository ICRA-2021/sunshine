ID=$(git log -n 1 --date=short --pretty=format:%ad-%H -- ../../sunshine)
mkdir -p $ID
cd $ID
