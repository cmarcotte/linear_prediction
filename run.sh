#!/bin/zsh

savedir=./fhn_pred/
N=8192

moddir=./fhn_data/
L=300.0

u0file=$moddir/u0

Ufile=$moddir/U

ufile=$moddir/u

w1file=$moddir/w1
w2file=$moddir/w2
v1file=$moddir/v1
v2file=$moddir/v2

for ((theta = 0 ; theta <= 0 ; theta+=5)); do
	echo $theta
	savedir=./fhn_pred/${theta}
	mkdir -p $savedir
	python3 linpred.py --savedir=$savedir --ufile=$ufile --v1file=$v1file --v2file=$v2file --w1file=$w1file --w2file=$w2file --Ufile=$Ufile --L=$L --N=$N --u0file=$u0file --theta=$theta
	#ffmpeg -framerate 1 -height 1376 -i ${savedir}/%03d.svg -c:v libsvtav1 -r 10 -crf 22 ${savedir}/${theta}.mp4
done
