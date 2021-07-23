#!/bin/bash
# make sure apt is up to date
apt-get update --missing
apt-get install -y curl
apt-get install -y build-essential libssl-dev

# install python
apt install python3 python3-pip -y
# Install infercode
pip install infercode

# install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
chmod +x rustup.sh
./rustup.sh -y
export RUSTC_WRAPPER=
export RUST_BACKTRACE=1

# install tree-sitter command
apt-get install -y git
git clone https://github.com/tree-sitter/tree-sitter $HOME/tree-sitter
cd $HOME/tree-sitter/cli && source $HOME/.cargo/env && cargo install --path .

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

# install tree-sitter parsers' NPM packages
cd && rm -rf node_modules && source $NVM_DIR/nvm.sh \
 && npm install \
	tree-sitter-bash \
        tree-sitter-c \
	tree-sitter-cpp \
	tree-sitter-c-sharp \
	tree-sitter-css \
	tree-sitter-elm \
	tree-sitter-go \
	tree-sitter-html \
	tree-sitter-java \
	tree-sitter-javascript \
	tree-sitter-kotlin \
	tree-sitter-lua \
	tree-sitter-php \
	tree-sitter-python \
	tree-sitter-ruby \
        tree-sitter-rust \
	tree-sitter-scala \
	tree-sitter-solidity \
	tree-sitter-verilog \
	tree-sitter-yaml

# tree-sitter-vhdl
# tree-sitter-embedded_template
# tree-sitter-systemrdl
# tree-sitter-toml
# tree-sitter-typescript
# tree-sitter-markdown
# tree-sitter-wat
# tree-sitter-eno
# tree-sitter-funnel
# tree-sitter-vue
# tree-sitter-r
# tree-sitter-haskell
# tree-sitter-wast
# tree-sitter-tsx

# Generate tree-sitter parsers' shared libraries
source $HOME/.cargo/env && for lang in $HOME/node_modules/tree-sitter-*; do cd $lang; tree-sitter generate; tree-sitter parse grammar.js; cd - ; done
mkdir -p $HOME/.tree-sitter/bin
cp $HOME/.cache/tree-sitter/lib/*.so $HOME/.tree-sitter/bin

