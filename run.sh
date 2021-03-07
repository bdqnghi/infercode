docker build -t yijun/tree-sitter docker
docker run -v $(pwd):/e -it yijun/tree-sitter subtree
