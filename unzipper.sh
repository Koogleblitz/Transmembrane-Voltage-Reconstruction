#!/bin/bash
#[+] default script doesnt work so I made my own. 
cd intracardiac_dataset

for i in {1..22}
do
   echo "[+]--------------- File $i ------------->>"
   wget https://library.ucsd.edu/dc/object/bb29449106/_${i}_1.tgz/download
   echo "Unzipping the file"
   tar -xvzf download
   rm -f download
done

cd -