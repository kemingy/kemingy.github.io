sync:
	git submodule update --init --recursive

build:
	zola build

serve:
	zola serve
