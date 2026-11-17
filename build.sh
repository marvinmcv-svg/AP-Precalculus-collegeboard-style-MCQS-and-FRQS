#!/bin/sh
# Wraps index.html (written as a page fragment) into a full HTML document in dist/.
set -e
mkdir -p dist
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<meta name="description" content="Interactive AP Precalculus Units 1-2 study guide with College Board style MCQs and FRQs.">\n<style>[hidden]{display:none!important}img{max-width:100%%}body{margin:0}</style>\n</head>\n<body>\n'
  cat index.html
  printf '\n</body>\n</html>\n'
} > dist/index.html
