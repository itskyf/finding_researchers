#!/usr/bin/env bash

for f in $(curl https://s3-us-west-2.amazonaws.com/ai2-s2ag/samples/MANIFEST.txt); do
	echo "$f"
	curl --create-dirs "https://s3-us-west-2.amazonaws.com/ai2-s2ag/$f" --output "$f"
	if [[ $f == *.gz ]]; then
		gzip --decompress "$f"
	fi
done
