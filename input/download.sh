#!/bin/bash

function download {
	link=https://zenodo.org/record/4537209/files/DataSet${1}.zip?download=1
	wget $link
	unzip DataSet${1}.zip?download=1;
}

for num in {1..11}
do
	download $num
done

rm DataSet*.zip*
