#!/bin/bash
case "$1" in
fast|bash|flatc|ggnn|gumtree|live_test|make|main_ggnn|protoc|unzip|wget|c2rust|rust2pickle|rustc|srcml|cargo|rustup|sccache|txl|nicad6|tree-sitter|sccache|cargo-geiger|rust2json|rust2xml|dctags)
	cmd=$1
	shift
	$cmd $@
	;;
tokei)
	# A POSIX variable
	OPTIND=1         # Reset in case getopts has been used previously in the shell.
	c2rust=
	dctags=
	fast=
	llvm=
	pycparser=
	syn=
	treesitter=
	unsafe=
	txl=
	shift
	while getopts "cdflpstux" opt; do
	case "$opt" in
	c)  
		c2rust=-$opt
		;;
	d)  
		dctags=-$opt
		;;
	f)  
		fast=-$opt
		;;
	l)  
		llvm=-$opt
		;;
	p)  
		pycparser=-$opt
		;;
	s)  
		syn=-$opt
		;;
	t)  
		treesitter=-$opt
		;;
	u)  
		unsafe=-$opt
		;;
	x)  
		txl=-$opt
		;;
	esac
	done
	shift $((OPTIND-1))
	if [ "$1" == "" ]; then
		tokei --streaming | awk -v output=$(pwd) '{if (NF>1) print $1, $2, output}' | xargs -n3 -P0 proc.sh ${c2rust:-} ${fast:-} ${llvm:-} ${pycparser:-} ${syn:-} ${treesitter:-} ${txl:-} ${unsafe:-} ${dctags:-}
	else
		cd $1 > /dev/null
		if [ "$2" == "" ]; then
			tokei --streaming | awk -v output=$(pwd) '{if (NF>1) print $1, $2, output}' | xargs -n3 -P0 proc.sh ${c2rust:-} ${fast:-} ${llvm:-} ${pycparser:-} ${syn:-} ${treesitter:-} ${txl:-} ${unsafe:-} ${dctags:-}
		else
			tokei --streaming | awk -v output=$2 '{if (NF>1) print $1, $2, output}' | xargs -n3 -P0 proc.sh ${c2rust:-} ${fast:-} ${llvm:-} ${pycparser:-} ${syn:-} ${treesitter:-} ${txl:-} ${unsafe:-} ${dctags:-}
		fi
		cd - > /dev/null
	fi
	;;
model)
	ls /model | grep -v .txt
	;;
classes)
	ls /model/*.txt | sed -e 's/.txt//'
	;;
subtree)
	if [ ! -f node_types_python.csv ]; then
		cp /usr/share/lib/node_types_*.csv .
	fi
	tokei --streaming=simple -t=C,C++,Java,Python,Rust | grep -v "#" | awk '{print tolower($1), $2}' | xargs -n 2 python3 /usr/share/lib/subtree.py
	;;
*)
	/home/docker/.cargo/bin/tokei $@
	;;
esac
