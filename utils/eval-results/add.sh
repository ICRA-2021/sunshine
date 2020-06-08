mkdir -p $(git log -n 1 --pretty=format:%H -- ../../sunshine)
cd $(git log -n 1 --pretty=format:%H -- ../../sunshine)
