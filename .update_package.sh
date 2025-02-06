#!/bin/bash

uv lock --P "$1" && uv sync --all-groups
