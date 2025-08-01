#!/bin/bash
uvx python -m http.server 20284 -d package_cache/index &
sleep .2
command=$1
shift 
$command "$@"

kill -- -$$
