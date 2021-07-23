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
pip install infercode

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
	cd $HOME/tree-sitter/cli && source $HOME/.cargo/env && cargo install --path .
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

# vhdl embedded_template systemrdl toml typescript markdown wat eno funnel vue r haskell wast tsx
for lang in bash c cpp c-sharp css elm go html java javascript kotlin lua php python ruby rust scala solidity verilog yaml; do
   if [ ! -d node_modules/tree-sitter-$lang ]; then
	   npm install tree-sitter-$lang
   fi
   cd node_modules/tree-sitter-$lang > /dev/null
   if [ ! -f $lang/grammar.js ]; then
	   tree-sitter generate
   fi
   if [ ! -f $HOME/.cache/tree-sitter/lib/$lang.so ]; then
	   tree-sitter parse grammar.js
   fi
   cd -  > /dev/null
done
mkdir -p $HOME/.tree-sitter/bin
cp $HOME/.cache/tree-sitter/lib/*.so $HOME/.tree-sitter/bin
