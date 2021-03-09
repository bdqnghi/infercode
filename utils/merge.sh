#!/bin/bash
cd $HOME/node_modules > /dev/null
find . -name grammar.json | while read f; do
	cat $f | jq '.'| grep '"name": "' | awk '{print $2}'| sed -e 's/,//g' -e 's/"//g' | sort | uniq
done | sort | uniq | while read l; do
	md=$(echo $l | md5sum)
	echo $md,$l
done | sed -e 's/ -//g' 
cd - > /dev/null

