sync:
	git submodule update --remote --merge

build:
	zola build

serve:
	zola serve