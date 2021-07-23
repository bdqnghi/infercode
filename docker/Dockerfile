FROM ubuntu
# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# make sure apt is up to date
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y build-essential libssl-dev

# Install nvm with node and npm
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 16.5.0
RUN curl https://raw.githubusercontent.com/creationix/nvm/v0.30.1/install.sh | bash \
    && source $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# install python
run apt install python3 python3-pip -y

# install Rust
run curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh
run chmod +x rustup.sh
run ./rustup.sh -y
ENV RUSTC_WRAPPER=
ENV RUST_BACKTRACE=1

# install tree-sitter command
run apt-get install -y git
run git clone https://github.com/tree-sitter/tree-sitter /root/tree-sitter
run cd /root/tree-sitter/cli && source $HOME/.cargo/env && cargo install --path .

# install tree-sitter parsers' NPM packages
run cd /root && rm -rf node_modules && source $NVM_DIR/nvm.sh \
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

# Install infercode
run pip install infercode

# Generate tree-sitter parsers' shared libraries
run source $HOME/.cargo/env && for lang in /root/node_modules/tree-sitter-*; do cd $lang; tree-sitter generate; tree-sitter parse grammar.js; cd - ; done
run mkdir -p /root/.tree-sitter/bin
run cp /root/.cache/tree-sitter/lib/*.so /root/.tree-sitter/bin

