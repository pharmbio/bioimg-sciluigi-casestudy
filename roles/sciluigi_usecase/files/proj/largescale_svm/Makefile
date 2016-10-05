bin/binaries.tar.gz:
	wget -O $@ https://ndownloader.figshare.com/files/6330402

bin/runjar: bin/binaries.tar.gz
	tar -C bin -zxvf $<

download_binaries: bin/runjar
