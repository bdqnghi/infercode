#!/bin/bash
cd $HOME/node_modules > /dev/null
find . -name node-types.json | while read f; do
	lang=$(basename $(dirname $(dirname $f)) | sed -e 's/tree-sitter-//g')
	cat $f | jq '.'| grep '"type": "' | awk '{print $2}'| sed -e 's/,//g' -e 's/"//g' | sort | uniq | while read l; do
	md=$(echo $l | md5sum)
	echo $md,$l | sed -e 's/ -//g'
done > node_types_$lang.txt
done
cd - > /dev/null

