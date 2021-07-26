#!/bin/bash
command -v curl >/dev/null 2>&1 || {
	# make sure apt is up to date
	sudo apt-get update --fix-missing
	sudo apt-get install -y curl
	sudo apt-get install -y build-essential libssl-dev
}
command -v pip >/dev/null 2>&1 || {
	# install python
	sudo apt install python3 python3-pip -y
}

# Install infercode
#pip3 install infercode

command -v cargo >/dev/null 2>&1 || {
	# install Rust
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
	chmod +x rustup.sh
	./rustup.sh -y
	export RUSTC_WRAPPER=
	export RUST_BACKTRACE=1
	source $HOME/.cargo/env 
}

command -v git  >/dev/null 2>&1 || {
	# install tree-sitter command
	sudo apt-get install -y git
}

command -v tree-sitter >/dev/null 2>&1 || {
	git clone https://github.com/tree-sitter/tree-sitter $HOME/tree-sitter
	cd $HOME/tree-sitter/cli && cargo install --path .
}

command -v npm >/dev/null 2>&1 || {
	# Install nvm with node and npm
	export NVM_DIR=/usr/local/nvm
	export NODE_VERSION=16.5.0
	curl https://raw.githubusercontent.com/creationix/nvm/v0.30.1/install.sh | bash \
	    && source $NVM_DIR/nvm.sh \
	    && nvm install $NODE_VERSION \
	    && nvm alias default $NODE_VERSION \
	    && nvm use default
	export NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
	export PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
	source $NVM_DIR/nvm.sh 
}

npm-parser() {
   lang=$1
   if [ ! -d node_modules/tree-sitter-$lang ]; then
	   npm install tree-sitter-$lang
   fi
}
export -f npm-parser
add-parser() {
   lang=$1
   if [ -d node_modules/tree-sitter-$lang ]; then
	   cd node_modules/tree-sitter-$lang > /dev/null
	   if [ ! -f grammar.js ]; then
		   tree-sitter generate
	   fi
	   if [ ! -f $HOME/.cache/tree-sitter/lib/$lang.so ]; then
		   tree-sitter parse grammar.js
	   fi
	   cd -  > /dev/null
   fi
}
export -f add-parser

command -v parallel >/dev/null 2>&1 || {
	sudo apt-get install parallel -y
}
# vhdl embedded_template systemrdl toml typescript markdown wat eno funnel vue r haskell wast tsx
#parallel add-parser ::: bash c cpp c-sharp css elm go html java javascript kotlin lua php python ruby rust scala solidity verilog yaml
for lang in bash c cpp c-sharp css elm go html java javascript kotlin lua php python ruby rust scala solidity verilog yaml; do
	npm-parser $lang
done
for lang in bash c cpp c-sharp css elm go html java javascript kotlin lua php python ruby rust scala solidity verilog yaml; do
	add-parser $lang
done
mkdir -p $HOME/.tree-sitter/bin
cp $HOME/.cache/tree-sitter/lib/*.so $HOME/.tree-sitter/bin

